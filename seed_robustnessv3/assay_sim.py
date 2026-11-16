"""One (seed, condition) job for the HSP-active functional assays.

Native trajectory: the seed's baseline is restored, HSP time constants set,
the condition applied (weights, benzo, stimulation proxy) exactly as in the
v2 cycle runs; the native drive `inp` is replaced by an equivalent Poisson
drive group (assay_stimulus.add_probe_drive, A = 0: rate b on every channel)
that can also play probe patterns. The run is never frozen: HSP, its rate
estimator and the treatment stay active everywhere.

At every checkpoint (uniform grid after settle_s) the full network state is
stored in memory. Four branches run from it, each after restoring that state:
  sham  -- native input, the native segment's documented seed: must replay
           the native continuation (see _native_seed for why the random state
           itself is not restored)
  enc   -- encoding presentations (left/right +/-1 pattern)
  noise -- clean-trained readout test: corrupted presentations
  mem   -- iid binary symbol per 50 ms bin
Probe branches reseed Brian's RNG with a branch-specific seed (independent
Poisson realisations; the symbol sequences are shared across conditions).
Then the checkpoint is restored once more and the native run continues to
the next checkpoint, so the native trajectory never sees a probe.

Saved per job (.npz): native 1 s E/I rates, HSP gains, seizures; per
checkpoint and branch the spike counts in the scoring windows, the symbols,
drift (k_AMPA/k_GABA/r_est at branch start and end, E/I spikes, peak 1 s E
rate) and the sham spike counts for the replay check; resolved config.
"""
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np

import worker

THIS_DIR = Path(__file__).resolve().parent
HEAVY_MONITORS = ("V_mon", "act_mon", "act_mon_I", "rate_pop_mon_E", "rate_pop_mon_I",
                  "k_AMPA_E_hsp", "k_GABA_E_hsp", "sp_exc", "sp_inh", "r_est")
KINDS = ("sham", "enc", "noise", "mem")


def _code_version():
    h = hashlib.sha1()
    for f in ("assay_sim.py", "assay_stimulus.py", "seeded_network.py", "conditions.py"):
        h.update((THIS_DIR / f).read_bytes())
    return h.hexdigest()[:12]


def _native_seed(seed, j):
    """Seed of the native segment starting at checkpoint j. Brian2 2.5's
    restore_random_state restores a raw C pointer to the Cython random
    buffer (a use-after-free once that buffer has been replaced), so the
    random state is never restored: the native trajectory is reseeded at
    every checkpoint with this documented seed instead, and the sham branch
    uses the same seed to replay it."""
    return 7_000_000 + 10_000 * int(seed) + int(j)


def _branch_seed(seed, cond_name, j, kind):
    h = hashlib.sha1(f"{seed}|{cond_name}|{j}|{kind}".encode()).hexdigest()
    return int(h[:8], 16)


def run_assay_job(seed, condition, baseline_path, grid, A_hz, work_dir, cache_dir, out_path,
                  template_cache_dir=None, n_checkpoints=None):
    name = condition["name"]
    tag = f"assay seed{seed}/{name}"
    try:
        worker._isolate(work_dir, cache_dir, template_cache_dir=template_cache_dir)
        import brian2
        from brian2 import second, ms, Hz, SpikeMonitor, StateMonitor, seed as b2seed
        from seeded_network import build_network_seeded, tau_HSP
        tau_hsp_s = float(tau_HSP / second)
        from HSP_Phase import _apply_drugs, _runkw
        from interventions import apply_weight_intervention
        import assay_stimulus as ast

        g = dict(grid)
        f = float(condition["freq_hz"])
        net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second,
                                   input_freq=f, network_seed=seed)
        net.restore("baseline", filename=str(baseline_path))
        P_E = net["P_E"]
        dt_s = float(P_E.clock.dt / second)
        g["dt_ms"] = dt_s * 1000
        for nm in worker.HSP_TAUS:
            setattr(P_E, nm, g["hsp_tau_s"] * second)
        if condition.get("weights"):
            w = condition["weights"]
            apply_weight_intervention(net, w["kind"], scale=w.get("scale", 1.0))
        _apply_drugs(net, condition.get("drug") or {})
        w_drive = float(np.mean(net["PG_E"].w[:]))
        for nm in HEAVY_MONITORS:                         # calibration history out of every checkpoint
            net.remove(net[nm])
        net["inp"].rates = 0 * Hz                         # the drive group below carries the drive

        cps = ast.checkpoint_times(g)
        if n_checkpoints is not None:
            cps = cps[:n_checkpoints]
        sch = ast.build_schedule(g, len(cps))
        ast.check_schedule(sch, [f], A_hz)
        drive, _ = ast.add_probe_drive(net, f, w_drive, sch, g["sched_ms"])
        spE = SpikeMonitor(P_E, name="spE")
        spI = SpikeMonitor(net["P_I"], name="spI")
        kmon = StateMonitor(P_E, ["k_AMPA_E_hsp", "k_GABA_E_hsp"], record=True, dt=1 * second, name="kmon")
        net.add(spE, spI, kmon)
        rk = _runkw(net)
        b2seed(5000 + int(seed))
        t_start = float(net.t / second)
        dur = ast.branch_durations(g)
        tpb = {k: int(round(v / g["dt_ms"])) for k, v in
               dict(pres=g["enc_present_ms"], mbin=g["mem_bin_ms"]).items()}
        nE = int(P_E.N)
        n_cp = len(cps)
        out = dict(
            enc_counts=np.zeros((n_cp, g["enc_trials"], nE), np.uint8),
            noise_counts=np.zeros((n_cp, sch["noise_q"].shape[1], nE), np.uint8),
            mem_counts=np.zeros((n_cp, g["mem_bins"], nE), np.uint8),
            mem_counts_alt=np.zeros((n_cp, g["mem_bins"], nE), np.uint8),
            sham_counts=np.zeros((n_cp, nE), np.uint16),
            **{f"{k}_{v}": np.zeros(n_cp) for k in KINDS
               for v in ("kA0", "kA1", "kG0", "kG1", "rest1", "nE", "nI", "maxE1s")})
        print(f"[{tag}] start: {g['run_s']:g} s native, {n_cp} checkpoints every {g['cp_spacing_s']:g} s, "
              f"A {A_hz:g} Hz, b {f:g} Hz", flush=True)
        t_wall = time.time()

        def window_counts(i0, t0_tick, t1_tick):
            ti = np.round(np.asarray(spE.t[i0:] / second) / dt_s).astype(np.int64) - t0_tick
            ii = np.asarray(spE.i[i0:])
            keep = (ti >= 0) & (ti < t1_tick - t0_tick)
            return ti[keep], ii[keep]

        for j, tcp in enumerate(cps):
            gap = tcp - (float(net.t / second) - t_start)
            if gap > 1e-9:
                net.run(gap * second, **rk)
            net.store("cp")
            for kind in KINDS:
                net.restore("cp")
                if kind == "sham":
                    b2seed(_native_seed(seed, j))       # the same seed the native continuation uses
                    drive.A = 0 * Hz
                else:
                    b2seed(_branch_seed(seed, name, j, kind))
                    drive.A = A_hz * Hz
                    drive.tb = net.t
                    drive.off = sch["offsets"][j, ("enc", "noise", "mem").index(kind)] * second
                i0E, i0I = spE.num_spikes, spI.num_spikes
                t0 = float(net.t / second)
                t0_tick = int(round(t0 / dt_s))
                n_ticks = int(round(dur[kind] / dt_s))
                out[f"{kind}_kA0"][j] = float(np.mean(P_E.k_AMPA_E_hsp[:]))
                out[f"{kind}_kG0"][j] = float(np.mean(P_E.k_GABA_E_hsp[:]))
                net.run(dur[kind] * second, **rk)
                ti, ii = window_counts(i0E, t0_tick, t0_tick + n_ticks)
                out[f"{kind}_kA1"][j] = float(np.mean(P_E.k_AMPA_E_hsp[:]))
                out[f"{kind}_kG1"][j] = float(np.mean(P_E.k_GABA_E_hsp[:]))
                out[f"{kind}_rest1"][j] = float(np.mean(P_E.x_spike[:]) / tau_hsp_s)   # r_est = x_spike / tau_HSP
                out[f"{kind}_nE"][j] = len(ti)
                out[f"{kind}_nI"][j] = spI.num_spikes - i0I
                sec = ti // int(round(1.0 / dt_s))
                out[f"{kind}_maxE1s"][j] = (np.bincount(sec).max() / nE) if len(sec) else 0.0
                if kind == "sham":
                    out["sham_counts"][j] = np.bincount(ii, minlength=nE)
                elif kind in ("enc", "noise"):
                    P = tpb["pres"]
                    trial, within = ti // P, ti % P
                    sel = within >= P - int(round(g["enc_count_ms"] / g["dt_ms"]))
                    C = np.zeros((out[f"{kind}_counts"].shape[1], nE), np.int64)
                    np.add.at(C, (trial[sel], ii[sel]), 1)
                    out[f"{kind}_counts"][j] = np.minimum(C, 255)
                else:
                    B = tpb["mbin"]
                    bn, within = ti // B, ti % B
                    for key, obs in (("mem_counts", g["mem_obs_ms"]), ("mem_counts_alt", g["mem_obs_ms_alt"])):
                        sel = within >= B - int(round(obs / g["dt_ms"]))
                        C = np.zeros((g["mem_bins"], nE), np.int64)
                        np.add.at(C, (bn[sel], ii[sel]), 1)
                        out[key][j] = np.minimum(C, 255)
            net.restore("cp")                              # native continues as if never probed,
            b2seed(_native_seed(seed, j))                  # from a documented seed (replayed by the sham)
            drive.A = 0 * Hz
            if (j + 1) % max(1, n_cp // 10) == 0:
                el = time.time() - t_wall
                print(f"[{tag}] {j + 1}/{n_cp} checkpoints, {el:6.0f}s wall, "
                      f"ETA {el * (n_cp - j - 1) / (j + 1):6.0f}s", flush=True)
        rest = g["run_s"] - (float(net.t / second) - t_start)
        if rest > 1e-9:
            net.run(rest * second, **rk)

        # native summaries
        stE = np.asarray(spE.t / second) - t_start
        stI = np.asarray(spI.t / second) - t_start
        n_s = int(round(g["run_s"]))
        rE = np.bincount(np.floor(stE[(stE >= 0) & (stE < n_s)]).astype(int), minlength=n_s) / nE
        rI = np.bincount(np.floor(stI[(stI >= 0) & (stI < n_s)]).astype(int), minlength=n_s) / int(net["P_I"].N)
        above = np.r_[False, rE > g["seizure_hz"], False]
        d = np.diff(above.astype(int))
        seizures = np.array([(float(a), float(b)) for a, b in zip(np.flatnonzero(d == 1),
                                                                  np.flatnonzero(d == -1))]).reshape(-1, 2)
        # native spikes per cell in each sham window (replay check)
        ticks = np.round(stE / dt_s).astype(np.int64)
        iE = np.asarray(spE.i)
        native_sham = np.zeros((n_cp, nE), np.uint16)
        n_sham = int(round(g["sham_s"] / dt_s))
        for j, tcp in enumerate(cps):
            a = int(round(tcp / dt_s))
            s = (ticks >= a) & (ticks < a + n_sham)
            native_sham[j] = np.bincount(iE[s], minlength=nE)
        kt = np.asarray(kmon.t / second) - t_start
        meta = dict(seed=int(seed), condition=name, freq_hz=f, A_hz=float(A_hz), drug=condition.get("drug"),
                    weights=condition.get("weights"), drive_weight=w_drive, dt_ms=g["dt_ms"], t_start=t_start,
                    n_E=nE, n_checkpoints=n_cp, code_version=_code_version(),
                    software=dict(python=platform.python_version(), brian2=brian2.__version__,
                                  numpy=np.__version__, target=str(brian2.prefs.codegen.target)),
                    baseline=str(baseline_path), sim_seed=5000 + int(seed),
                    grid={k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()},
                    completed=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, cps=cps, rE=rE.astype(np.float32), rI=rI.astype(np.float32),
                            kt=kt, kA=np.asarray(kmon.k_AMPA_E_hsp).mean(0), kG=np.asarray(kmon.k_GABA_E_hsp).mean(0),
                            seizures=seizures, native_sham_counts=native_sham, rows=sch["rows"],
                            offsets=sch["offsets"], enc_q=sch["enc_q"], noise_q=sch["noise_q"],
                            noise_eta=sch["noise_eta"], mem_q=sch["mem_q"], meta=json.dumps(meta), **out)
        replay = float(np.mean(np.all(native_sham == out["sham_counts"], axis=1)))
        print(f"[{tag}] done in {time.time() - t_wall:.0f}s wall: {len(seizures)} seizures, "
              f"sham replay identical at {replay:.0%} of checkpoints -> {out_path}", flush=True)
        return dict(seed=seed, condition=name, ok=True, out_path=str(out_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] FAILED: {exc}", flush=True)
        return dict(seed=seed, condition=name, ok=False, error=str(exc), traceback=traceback.format_exc())


if __name__ == "__main__":
    # one job per process (run_assays.py): python assay_sim.py job.json
    import sys
    spec = json.loads(Path(sys.argv[1]).read_text())
    spec["grid"] = {k: (tuple(v) if isinstance(v, list) else v) for k, v in spec["grid"].items()}
    r = run_assay_job(**spec)
    if not r["ok"]:
        print(r["traceback"], flush=True)
    sys.exit(0 if r["ok"] else 1)
