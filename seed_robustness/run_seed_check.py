"""
Repeat the phase-resolved IPC (+ participation-ratio) intervention pipeline
from IPC_only.ipynb / DBS_intervention.ipynb across 2-3 independent network
seeds and check that the orderings (phase contrasts, intervention ranking)
hold.

Nothing in the existing notebooks or HSP_Phase.py/hsp_capacity.py/
hsp_metrics.py is modified. Instead:

  - seeded_network.py duplicates build_network()/run_sim() with a
    `network_seed` argument threaded through brian2.seed(), so each seed
    gets an independent draw of the recurrent weights and Poisson streams.
  - Each network seed needs its own calibrated ~2 Hz HSP fixed point (the
    notebooks share one `baseline_2hz.bri` on disk); we regenerate one per
    seed with the same recipe (all four HSP tau's = 50 s, 15000 s settling).
  - HSP_Phase.characterize_cycle() and friends hard-code
    `net.restore('baseline', filename='baseline_2hz.bri')`, a path relative
    to the process's cwd. Every (seed, condition) job runs in its own OS
    process (joblib/loky), chdir'd into its own scratch dir holding just
    that seed's baseline_2hz.bri -- so the unmodified HSP_Phase functions
    pick up the right network without any code changes. Each such process
    also gets its own brian2 Cython cache dir, so N parallel Brian2 code
    generations never collide/lock on one shared cache.
  - Each condition job only computes the 36 Hz (seizure) phase under its own
    drug. The 40 Hz reference phase (the healthy, non-seizure fixed point --
    there's no reason to apply a drug/DBS dose to a state that isn't
    seizing) is computed ONCE per seed at drug={} (Step A.5 below) and shared
    across every condition, instead of each condition redundantly
    recomputing its own ~40Hz+drug run.

Usage:
    python run_seed_check.py
    python run_seed_check.py --seeds 1,2,3 --n-jobs 6
    python run_seed_check.py --conditions untreated,"benzo (k=1.0)"
    python run_seed_check.py --fine   # use the notebook's full-resolution grid
"""
import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from joblib import Parallel, delayed  # noqa: E402

import conditions as cond_mod  # noqa: E402
import orderings  # noqa: E402
import plot_ipc_summary  # noqa: E402
import plot_seizure_onset  # noqa: E402
import plot_traces  # noqa: E402
import worker  # noqa: E402


def available_cores():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="1,2,3,4,5",
                  help="comma-separated network seeds (default: 1,2,3,4,5)")
    p.add_argument("--conditions", default=None,
                  help="comma-separated condition names to run (default: all in "
                       "conditions.CONDITIONS); quote names with spaces")
    p.add_argument("--out-dir", default=str(THIS_DIR / "results"))
    p.add_argument("--n-jobs", type=int, default=None,
                  help="default: all cores minus 2")
    p.add_argument("--calib-time-s", type=float, default=15000.0,
                  help="HSP settling time before storing each seed's baseline "
                       "(same value the notebooks used: 15000 s)")
    p.add_argument("--force-recalibrate", action="store_true",
                  help="rebuild per-seed baselines even if cached")
    p.add_argument("--force-rerun", action="store_true",
                  help="rerun (seed, condition) jobs AND the per-seed reference "
                       "phase even if a saved result exists")
    p.add_argument("--fine", action="store_true",
                  help="use the notebooks' full-resolution probe grid instead of "
                       "the coarse seed-check grid")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    all_names = [c["name"] for c in cond_mod.CONDITIONS]
    if args.conditions:
        wanted = {c.strip() for c in args.conditions.split(",")}
        conditions = [c for c in cond_mod.CONDITIONS if c["name"] in wanted]
        missing = wanted - {c["name"] for c in conditions}
        if missing:
            raise SystemExit(f"unknown condition name(s): {missing}. "
                            f"available: {all_names}")
    else:
        conditions = cond_mod.CONDITIONS

    grid = cond_mod.FINE_GRID if args.fine else cond_mod.COARSE_GRID
    # Resolve to absolute now: each (seed, condition) job chdir's into its own
    # scratch dir in a subprocess, so any relative path computed here would
    # be interpreted relative to the wrong cwd once it gets there.
    out_dir = Path(args.out_dir).resolve()
    n_jobs = args.n_jobs if args.n_jobs is not None else max(1, available_cores() - 2)

    print(f"network seeds : {seeds}")
    print(f"conditions    : {[c['name'] for c in conditions]}")
    print(f"grid          : {'FINE' if args.fine else 'COARSE'}  "
        f"(probe_seeds={len(grid['probe_seeds'])}, n_steps={grid['n_steps']}, "
        f"max_delay={grid['max_delay']})")
    print(f"n_jobs        : {n_jobs}  (of {available_cores()} available cores, "
        f"leaving {max(0, available_cores() - n_jobs)} free)")
    print(f"out_dir       : {out_dir}")

    # ---- Step 0: compile every code object ONCE, single-process ----------
    # Every (seed, condition) job builds the identical network structure --
    # only weight values and drug-set state differ, never the generated
    # Cython/C++ source -- so compiling it here and hardlinking the result
    # into each job's own cache dir (see worker._isolate) avoids recompiling
    # the same ~dozens of extensions once per job. On this machine a cold
    # compile took the better part of an hour; a hardlink-seeded cache hit
    # is instant.
    template_cache_dir = out_dir / "cython_cache" / "_template"
    if not any(template_cache_dir.glob("*.so")):
        print(f"[prewarm] compiling code objects once into {template_cache_dir} ...")
        pw = worker.prewarm_cache_job(
            template_cache_dir=str(template_cache_dir),
            work_dir=str(out_dir / "work" / "_prewarm"),
        )
        if not pw["ok"]:
            raise SystemExit(f"prewarm compile failed:\n{pw['traceback']}")
    else:
        print(f"[prewarm] using cached compiled extensions in {template_cache_dir}")

    # ---- Step A: one calibrated ~2 Hz baseline per network seed ----------
    baselines_dir = out_dir / "baselines"
    calib_traces_dir = out_dir / "traces" / "calibration"
    full_calib_seeds, extract_only_seeds = [], []
    for seed in seeds:
        baseline_path = baselines_dir / f"seed{seed}.bri"
        trace_path = calib_traces_dir / f"seed{seed}.npz"
        if args.force_recalibrate or not baseline_path.exists():
            full_calib_seeds.append(seed)
        elif not trace_path.exists():
            # net.store() keeps each monitor's full recorded history, not
            # just final state, so the trace is already inside baseline_path
            # -- just restore + read, no simulation needed.
            extract_only_seeds.append(seed)
        else:
            print(f"[calibrate] seed {seed}: using cached {baseline_path} (+ trace)")

    calib_calls = (
        [delayed(worker.calibrate_seed_job)(
            seed=seed,
            baseline_path=str(baselines_dir / f"seed{seed}.bri"),
            work_dir=str(out_dir / "work" / f"calib_seed{seed}"),
            cache_dir=str(out_dir / "cython_cache" / f"calib_seed{seed}"),
            calib_time_s=args.calib_time_s,
            template_cache_dir=str(template_cache_dir),
            trace_out_path=str(calib_traces_dir / f"seed{seed}.npz"),
        ) for seed in full_calib_seeds]
        + [delayed(worker.extract_calibration_trace_job)(
            seed=seed,
            baseline_path=str(baselines_dir / f"seed{seed}.bri"),
            trace_out_path=str(calib_traces_dir / f"seed{seed}.npz"),
            work_dir=str(out_dir / "work" / f"extract_seed{seed}"),
            cache_dir=str(out_dir / "cython_cache" / f"extract_seed{seed}"),
            template_cache_dir=str(template_cache_dir),
        ) for seed in extract_only_seeds]
    )

    if calib_calls:
        if full_calib_seeds:
            print(f"[calibrate] running {len(full_calib_seeds)} full calibration(s) "
                f"({args.calib_time_s:.0f}s HSP settling each)...")
        if extract_only_seeds:
            print(f"[calibrate] backfilling traces for {len(extract_only_seeds)} "
                f"already-calibrated seed(s) (cheap: restore + read, no rerun)...")
        calib_n_jobs = max(1, min(n_jobs, len(calib_calls)))
        calib_results = Parallel(n_jobs=calib_n_jobs, backend="loky", verbose=10)(calib_calls)
        for r in calib_results:
            if not r["ok"]:
                print(f"[calibrate] seed {r['seed']} FAILED:\n{r['traceback']}")
        if any(not r["ok"] for r in calib_results):
            raise SystemExit("one or more baseline calibrations failed; see above")

    # ---- Step A.5: one untreated reference phase per seed, SHARED ---------
    # The reference frequency (40 Hz, the healthy non-seizure fixed point) is
    # not a seizure state, so there's no reason to apply a drug/DBS to it --
    # compute it once per seed at drug={} and every condition below reuses
    # it, instead of each of N conditions redundantly recomputing its own
    # ~40Hz+drug run (this used to be 1/8th useful work, 7/8ths waste).
    reference_dir = out_dir / "reference"
    reference_seeds = [s for s in seeds
                      if args.force_rerun or not (reference_dir / f"seed{s}.pkl").exists()]
    for s in seeds:
        if s not in reference_seeds:
            print(f"[reference] seed {s}: using cached {reference_dir / f'seed{s}.pkl'}")

    if reference_seeds:
        print(f"[reference] computing untreated {cond_mod.STEADY_FREQ_HZ:g}Hz reference "
            f"for {len(reference_seeds)} seed(s) (shared across all conditions)...")
        ref_n_jobs = max(1, min(n_jobs, len(reference_seeds)))
        ref_results = Parallel(n_jobs=ref_n_jobs, backend="loky", verbose=10)(
            delayed(worker.compute_reference_job)(
                seed=seed,
                baseline_path=str(baselines_dir / f"seed{seed}.bri"),
                grid=grid,
                work_dir=str(out_dir / "work" / f"reference_seed{seed}"),
                cache_dir=str(out_dir / "cython_cache" / f"reference_seed{seed}"),
                out_path=str(reference_dir / f"seed{seed}.pkl"),
                template_cache_dir=str(template_cache_dir),
                reference_freq_hz=cond_mod.STEADY_FREQ_HZ,
                inspect_dir=str(out_dir / "ipc_probe_inspection" / "runs"),
            )
            for seed in reference_seeds
        )
        for r in ref_results:
            if not r["ok"]:
                print(f"[reference] seed {r['seed']} FAILED:\n{r['traceback']}")
        if any(not r["ok"] for r in ref_results):
            raise SystemExit("one or more reference computations failed; see above")

    # ---- Step B: every (seed, condition) as its own process --------------
    raw_dir = out_dir / "raw"
    traces_dir = out_dir / "traces"
    jobs = []
    for seed in seeds:
        for c in conditions:
            slug = worker._slug(f"seed{seed}_{c['name']}")
            raw_path = raw_dir / f"{slug}.json"
            trace_path = traces_dir / f"{slug}.npz"
            jobs.append((seed, c, raw_path, trace_path))

    # A job also reruns if its raw result predates the trace-export feature
    # (raw_path exists but trace_path doesn't) -- backfills traces for
    # results computed before this was added, without rerunning everything.
    todo = [(s, c, rp, tp) for s, c, rp, tp in jobs
           if args.force_rerun or not rp.exists() or not tp.exists()]
    cached = [(s, c, rp, tp) for s, c, rp, tp in jobs
             if not args.force_rerun and rp.exists() and tp.exists()]
    for s, c, rp, tp in cached:
        print(f"[run] seed {s} / {c['name']}: using cached {rp}")

    if todo:
        print(f"[run] running {len(todo)} (seed, condition) job(s), "
            f"{n_jobs} at a time...")
        Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(worker.run_condition_job)(
                seed=seed, condition_name=c["name"], drug=c["drug"],
                weights=c.get("weights"),
                baseline_path=str(baselines_dir / f"seed{seed}.bri"),
                grid=grid,
                work_dir=str(out_dir / "work" / worker._slug(f"seed{seed}_{c['name']}")),
                cache_dir=str(out_dir / "cython_cache" / worker._slug(f"seed{seed}_{c['name']}")),
                raw_out_path=str(raw_path),
                trace_out_path=str(trace_path),
                reference_path=str(reference_dir / f"seed{seed}.pkl"),
                burst_freq_hz=cond_mod.BURST_FREQ_HZ,
                template_cache_dir=str(template_cache_dir),
                inspect_dir=str(out_dir / "ipc_probe_inspection" / "runs"),
            )
            for seed, c, raw_path, trace_path in todo
        )

    # ---- Analyze -----------------------------------------------------
    results = orderings.load_results(raw_dir)
    failed = [r for r in results if not r.get("ok")]
    for r in failed:
        print(f"[run] seed {r.get('seed')} / {r.get('condition')} FAILED:\n"
            f"{r.get('traceback')}")

    orderings.save_summary_csv(results, out_dir / "summary.csv")
    orderings.print_report(results)
    print(f"\nsummary CSV: {out_dir / 'summary.csv'}")
    print(f"raw per-job results: {raw_dir}/*.json")

    plot_paths = plot_traces.plot_all(out_dir)
    for p in plot_paths:
        print(f"trace plot (all seeds, sanity check): {p}")

    ipc_plot = plot_ipc_summary.plot_ipc_across_conditions(
        results, out_dir / "plots" / "ipc_summary.png")
    if ipc_plot:
        print(f"IPC summary (all conditions, every seed): {ipc_plot}")
    ipc_v3 = plot_ipc_summary.plot_ipc_v3(results, out_dir / "plots" / "ipc_summary_v3.png")
    if ipc_v3:
        print(f"IPC summary v3 (steady-state IPC, selected conditions): {ipc_v3}")
    for p in plot_ipc_summary.plot_ipc_v4(results, out_dir / "plots" / "ipc_summary_v4.png"):
        print(f"IPC summary v4 (untreated, benzo, DBS 10-40 pA): {p}")

    onset_paths = plot_seizure_onset.plot_all(out_dir)
    for p in onset_paths:
        print(f"onset-aligned plot (0-5Hz, bifurcation view): {p}")


if __name__ == "__main__":
    main()
