"""Per-process job bodies for the multi-seed replication check.

Each function here is meant to run inside its own OS process (spawned by
joblib/loky), never in the parent. That's what makes the "isolate cwd +
Cython cache dir per process" trick work: HSP_Phase.characterize_cycle() and
friends hard-code `net.restore('baseline', filename='baseline_2hz.bri')` --
a path relative to the process's cwd -- so giving each process its own cwd
with its own seed-specific baseline_2hz.bri lets us reuse those functions
completely unmodified for independent network realizations.
"""
import shutil
import sys
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EPILEPSY_DIR = THIS_DIR.parent
for _p in (str(EPILEPSY_DIR), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _seed_cache_from_template(cache_dir, template_cache_dir):
    """Populate a fresh per-process cache dir from an already-compiled
    template cache, via hardlinks (instant, no recompilation) rather than a
    real copy. Every (seed, condition) job builds the SAME network structure
    -- only runtime weight values and drug-set state variables differ, never
    the generated Cython/C++ source -- so the compiled .so's from one warm-up
    build are valid for every job. Safe to hardlink because brian2 never
    mutates a cached file in place: a structural change produces a new
    content hash and therefore a new filename."""
    import os
    template_cache_dir = Path(template_cache_dir)
    if not template_cache_dir.exists():
        return
    # Only the compiled extensions matter for a cache hit; the cache dir also
    # accumulates a "tmp/<full source path>/" tree of intermediate .cpp/.o
    # build files (a pyximport quirk when the cache dir lives under /tmp) and
    # 0-byte .lock marker files -- skip those, we don't need them and some
    # are directories os.link()/copy2() can't handle anyway.
    for src in template_cache_dir.glob("*.so"):
        dst = Path(cache_dir) / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _isolate(work_dir, cache_dir, template_cache_dir=None):
    """chdir into `work_dir` and point brian2's cython cache at `cache_dir`,
    both private to this process. Must run before any brian2 code generation
    (i.e. before the first net.run()). If `template_cache_dir` is given (see
    `prewarm_cache_job`), hardlink its already-compiled extensions in first
    so this process doesn't pay to recompile identical code from scratch."""
    import os
    from brian2 import prefs

    work_dir = Path(work_dir)
    cache_dir = Path(cache_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if template_cache_dir is not None:
        _seed_cache_from_template(cache_dir, template_cache_dir)
    os.chdir(work_dir)
    prefs.codegen.runtime.cython.cache_dir = str(cache_dir)
    prefs.codegen.runtime.cython.multiprocess_safe = False


def prewarm_cache_job(template_cache_dir, work_dir, network_seed=0, tau_s=50.0):
    """Compile every code object the network needs exactly once, into
    `template_cache_dir`. Run this single-process, before fanning out the
    (seed, condition) grid, so each of those parallel processes can hardlink
    the results instead of independently recompiling ~dozens of Cython
    extensions (which is what made the very first build in this environment
    take the better part of an hour)."""
    try:
        _isolate(work_dir, template_cache_dir)
        from brian2 import second
        from seeded_network import build_network_seeded
        build_network_seeded(
            tau_ins_AMPA=tau_s * second, tau_int_AMPA=tau_s * second,
            tau_ins_GABA=tau_s * second, tau_int_GABA=tau_s * second,
            network_seed=network_seed,
        )
        return dict(ok=True, template_cache_dir=str(template_cache_dir))
    except Exception as exc:  # noqa: BLE001 - report, don't kill the batch
        return dict(ok=False, error=str(exc), traceback=traceback.format_exc())


def calibrate_seed_job(seed, baseline_path, work_dir, cache_dir,
                       calib_time_s=15000.0, dt_us=1000, tau_s=50.0,
                       input_freq=40, template_cache_dir=None, trace_out_path=None):
    """Build a fresh network for `seed` and run it to the ~2 Hz HSP fixed
    point, storing the result at `baseline_path`. Runs in its own process.
    If `trace_out_path` is given, also dumps the full settling trace (see
    seeded_network.calibrate_baseline) for the pre-drug stability sanity
    check plotted by plot_traces.plot_calibration()."""
    try:
        _isolate(work_dir, cache_dir, template_cache_dir=template_cache_dir)
        from seeded_network import calibrate_baseline
        path = calibrate_baseline(seed, baseline_path, calib_time_s=calib_time_s,
                                  dt_us=dt_us, tau_s=tau_s, input_freq=input_freq,
                                  trace_out_path=trace_out_path)
        return dict(seed=seed, baseline_path=path, ok=True)
    except Exception as exc:  # noqa: BLE001 - report, don't kill the batch
        return dict(seed=seed, ok=False, error=str(exc), traceback=traceback.format_exc())


def extract_calibration_trace_job(seed, baseline_path, trace_out_path,
                                  work_dir, cache_dir, template_cache_dir=None,
                                  tau_s=50.0, input_freq=40):
    """Backfill a calibration trace for a seed that's already calibrated --
    net.restore() + read monitors, no simulation. Cheap (seconds, not the
    calib_time_s of a real calibration): the full history is already inside
    `baseline_path` (see seeded_network.extract_calibration_trace)."""
    try:
        _isolate(work_dir, cache_dir, template_cache_dir=template_cache_dir)
        from seeded_network import extract_calibration_trace
        extract_calibration_trace(seed, baseline_path, trace_out_path,
                                  tau_s=tau_s, input_freq=input_freq)
        return dict(seed=seed, baseline_path=str(baseline_path), ok=True)
    except Exception as exc:  # noqa: BLE001 - report, don't kill the batch
        return dict(seed=seed, ok=False, error=str(exc), traceback=traceback.format_exc())


def compute_reference_job(seed, baseline_path, grid, work_dir, cache_dir,
                          out_path, template_cache_dir=None,
                          reference_freq_hz=40.0, tau_s=50.0):
    """Compute the untreated `reference_freq_hz` phase (the healthy, non-
    seizure fixed point -- 40 Hz by default) ONCE per seed, drug={}. Every
    condition job then reuses this same reference instead of each drug
    separately recomputing its own 40Hz+drug run -- there's no reason to
    apply a seizure treatment to a state that isn't seizing, and it was pure
    redundant compute (7 conditions x the same ~40Hz work each). Runs in its
    own process; pickles the raw result (has numpy arrays, not meant to be
    read by anything other than run_condition_job)."""
    tag = f"seed{seed}/reference"
    try:
        _isolate(work_dir, cache_dir, template_cache_dir=template_cache_dir)
        shutil.copy2(baseline_path, Path(work_dir) / "baseline_2hz.bri")

        from brian2 import second
        from seeded_network import build_network_seeded
        from coarse_phase import phase_capacity_seeded_coarse

        print(f"[{tag}] starting", flush=True)
        net = build_network_seeded(
            tau_ins_AMPA=tau_s * second, tau_int_AMPA=tau_s * second,
            tau_ins_GABA=tau_s * second, tau_int_GABA=tau_s * second,
            input_freq=reference_freq_hz, network_seed=seed,
        )
        res = phase_capacity_seeded_coarse(
            net, reference_freq_hz, grid["probe_seeds"], drug={},
            phases=grid["phases"], n_steps=grid["n_steps"],
            amp_hz=grid["amp_hz"], step_s=grid["step_s"],
            max_delay=grid["max_delay"], char_kw=grid["char_kw"],
            snapshot_warmup_s=grid["snapshot_warmup_s"],
            snapshot_chunk_s=grid["snapshot_chunk_s"],
            snapshot_max_scan_s=grid["snapshot_max_scan_s"],
            progress_tag=tag,
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(out_path, "wb") as fh:
            pickle.dump(res, fh)
        print(f"[{tag}] done", flush=True)
        return dict(seed=seed, ok=True, out_path=str(out_path))
    except Exception as exc:  # noqa: BLE001 - report, don't kill the batch
        print(f"[{tag}] FAILED: {exc}", flush=True)
        return dict(seed=seed, ok=False, error=str(exc), traceback=traceback.format_exc())


def _jsonify(x):
    import numpy as np
    if isinstance(x, dict):
        return {k: _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonify(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _save_traces(trace_out_path, burst, reference, seed, condition_name):
    import numpy as np
    Path(trace_out_path).parent.mkdir(parents=True, exist_ok=True)
    cb, cs = burst["cycle"], reference["cycle"]
    np.savez_compressed(
        trace_out_path,
        seed=seed, condition=str(condition_name),
        burst_freq_hz=burst["input_freq_hz"],
        steady_freq_hz=reference["input_freq_hz"],
        t_burst=cb["t"], rE_burst=cb["rE"], rI_burst=cb["rI"],
        kA_burst=cb["k_AMPA"], kG_burst=cb["k_GABA"], r_est_burst=cb["r_est"],
        is_bursting_burst=cb["is_bursting"],
        burst_onsets_burst=cb["burst_onsets"], burst_offsets_burst=cb["burst_offsets"],
        t_steady=cs["t"], rE_steady=cs["rE"], rI_steady=cs["rI"],
        kA_steady=cs["k_AMPA"], kG_steady=cs["k_GABA"], r_est_steady=cs["r_est"],
        is_bursting_steady=cs["is_bursting"],
        burst_onsets_steady=cs["burst_onsets"], burst_offsets_steady=cs["burst_offsets"],
    )


def run_condition_job(seed, condition_name, drug, baseline_path, grid,
                      work_dir, cache_dir, raw_out_path, reference_path,
                      burst_freq_hz=36.0, tau_s=50.0,
                      template_cache_dir=None, trace_out_path=None):
    """Restore the per-seed baseline, run the phase-resolved IPC+PR
    replication at `burst_freq_hz` (36 Hz) under `drug` --
    coarse_phase.phase_capacity_seeded_coarse(), which calls HSP_Phase's own
    characterize_cycle/snapshot_at_kphase/_ipc_probe unmodified -- then merge
    in the shared untreated reference phase (see compute_reference_job,
    loaded from `reference_path`) instead of computing this condition's own
    40Hz+drug run. Derives an intervention-style summary (ipc_ref/
    ipc_retained/tradeoff_score, same formula as
    HSP_Phase.intervention_comparison) from the merged result. Runs in its
    own process.

    If `trace_out_path` is given, also dumps the raw cycle traces (E/I
    population rate, k_AMPA/k_GABA, r_est vs time, at both burst_freq_hz and
    the shared reference frequency) to that .npz path -- the same signals
    characterize_cycle() already computes internally for phase-finding, just
    not normally returned past this function. plot_traces.py reads these for
    the auto-generated per-condition, all-seeds sanity plots."""
    tag = f"seed{seed}/{condition_name}"
    try:
        _isolate(work_dir, cache_dir, template_cache_dir=template_cache_dir)
        shutil.copy2(baseline_path, Path(work_dir) / "baseline_2hz.bri")

        import pickle
        import numpy as np
        from brian2 import second
        from seeded_network import build_network_seeded
        from HSP_Phase import paired_phase_stats
        from coarse_phase import phase_capacity_seeded_coarse

        with open(reference_path, "rb") as fh:
            reference = pickle.load(fh)

        print(f"[{tag}] starting", flush=True)
        net = build_network_seeded(
            tau_ins_AMPA=tau_s * second, tau_int_AMPA=tau_s * second,
            tau_ins_GABA=tau_s * second, tau_int_GABA=tau_s * second,
            input_freq=burst_freq_hz, network_seed=seed,
        )

        burst = phase_capacity_seeded_coarse(
            net, burst_freq_hz, grid["probe_seeds"], drug=drug,
            phases=grid["phases"], n_steps=grid["n_steps"],
            amp_hz=grid["amp_hz"], step_s=grid["step_s"],
            max_delay=grid["max_delay"], char_kw=grid["char_kw"],
            snapshot_warmup_s=grid["snapshot_warmup_s"],
            snapshot_chunk_s=grid["snapshot_chunk_s"],
            snapshot_max_scan_s=grid["snapshot_max_scan_s"],
            progress_tag=tag,
        )

        # Same merge order HSP_Phase.run_phase_replication() itself uses:
        # burst's phases first, reference's 'steady' on top (a collision would
        # mean the reference frequency also burst, which the 40Hz fixed point
        # isn't designed to do).
        by_phase = {k: v for k, v in burst["by_phase"].items() if v.get("found")}
        by_phase.update({k: v for k, v in reference["by_phase"].items() if v.get("found")})

        ipc_stats = paired_phase_stats(by_phase, metric="ipc_total")
        pr_stats = paired_phase_stats(by_phase, metric="participation_ratio")

        if trace_out_path is not None:
            _save_traces(trace_out_path, burst, reference, seed, condition_name)

        cyc = burst["cycle"]
        ref = by_phase.get("mid") or by_phase.get("steady")
        post = by_phase.get("post_ictal")
        ipc_ref = float(np.mean(ref["ipc_total"])) if ref else float("nan")
        ipc_post = float(np.mean(post["ipc_total"])) if post else ipc_ref
        retained = (ipc_post / ipc_ref) if (ref and ipc_ref > 0) else float("nan")
        score = ipc_ref / (1.0 + cyc["cycle_lf_power"]) if ref else float("nan")

        raw = dict(
            seed=seed, condition=condition_name, drug=drug,
            cycle=dict(is_bursting=cyc["is_bursting"], period_s=cyc["period_s"],
                      recovery_s=cyc["recovery_s"], cycle_lf_power=cyc["cycle_lf_power"],
                      E_rate=cyc["E_rate"], I_rate=cyc["I_rate"]),
            ipc_stats=ipc_stats, pr_stats=pr_stats,
            ipc_ref=ipc_ref, ipc_post_ictal=ipc_post, ipc_retained=retained,
            tradeoff_score=score,
            by_phase_raw={ph: {k: v for k, v in d.items()
                               if k in ("ipc_total", "participation_ratio", "E_rate_probe")}
                         for ph, d in by_phase.items()},
            ok=True,
        )
        raw_json = _jsonify(raw)
        Path(raw_out_path).parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(raw_out_path, "w") as fh:
            json.dump(raw_json, fh, indent=2)
        print(f"[{tag}] done (ipc_ref={ipc_ref:.3f})", flush=True)
        return raw_json
    except Exception as exc:  # noqa: BLE001 - report, don't kill the batch
        print(f"[{tag}] FAILED: {exc}", flush=True)
        return dict(seed=seed, condition=condition_name, ok=False,
                   error=str(exc), traceback=traceback.format_exc())
