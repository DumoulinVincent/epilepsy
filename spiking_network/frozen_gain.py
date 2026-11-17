"""Frozen-HSP (fast-subsystem) dynamics: the spiking network with its homeostatic gains held fixed.

The slow state is one-dimensional along the cycle (k_AMPA + k_GABA = 1.00, minimum 0.975, in all seeds), so a gain
value kA sets k_AMPA = kA and k_GABA = 1 - kA in every E cell (override with kG). HSP is frozen by setting its four
time constants to 1e99 s; membrane, synaptic and Poisson-input dynamics run normally.

Two kinds of jobs:
  trial   fixed (f_in, kA): start from rest (init "low") or from a kicked high-rate state (init "high": 2 s of
          100 Hz input, then f_in), run T s, record 1 s E/I rates. Tells whether the low state survives
          (time to first escape, right-censored at T) and whether the high state persists or returns.
  sweep   f_in fixed, kA stepped up then down, each value held hold_s, the network state carried over between
          steps: hysteresis in the low/high state.

    python frozen_gain.py pilot        # small grid at 36 Hz, seed 1 (results/frozen_gain/pilot/)
    python frozen_gain.py full         # regime map, escape statistics, networks 2-5 (results/frozen_gain/full/)

Outputs: one .npz per job with rate_E, rate_I (1 s bins), the gains read back at the end, and the settings.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import worker  # noqa: E402  (puts ../model on sys.path)
from long_tau_cycle import find_condition  # noqa: E402

RESULTS = THIS_DIR / "results"
HIGH_HZ = 10.0          # E rate above this in a 1 s bin = high-rate state (the manuscript's episode criterion)
KICK_HZ, KICK_S = 100.0, 2.0


def _build(seed, f_in, noise_scale=1.0, dt_ms=None):
    """Restored, HSP-frozen network with light monitors: 1 s E/I spike counts.

    noise_scale c: input rate f_in*c with input weight w/c, so the mean afferent drive (rate x weight) is unchanged and
    its variance scales as 1/c (c > 1: less noise). dt_ms: integration step after the restore (default 1 ms)."""
    from brian2 import second, Hz, NeuronGroup, Synapses, StateMonitor
    from seeded_network import build_network_seeded
    from HSP_Phase import _freeze_hsp
    net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second, input_freq=f_in,
                               network_seed=seed)
    net.restore("baseline", filename=str(RESULTS / "baselines" / f"seed{seed}.bri"))
    for mon in [o for o in net.objects if isinstance(o, StateMonitor)] + [net["sp_exc"], net["sp_inh"]]:
        net.remove(mon)
    _freeze_hsp(net)
    P_E, P_I = net["P_E"], net["P_I"]
    cnt = NeuronGroup(2, "n : 1", name="cnt")
    cE = Synapses(P_E, cnt, on_pre="n_post += 1", name="cnt_E"); cE.connect(j="0")
    cI = Synapses(P_I, cnt, on_pre="n_post += 1", name="cnt_I"); cI.connect(j="1")
    m = StateMonitor(cnt, "n", record=True, dt=1 * second, when="end", order=0, name="m_cnt")
    cnt.run_regularly("n = 0", dt=1 * second, when="end", order=1)
    net.add(cnt, cE, cI, m)
    net.f_eff = f_in * noise_scale * Hz
    net["PG_E"].w[:] = np.asarray(net["PG_E"].w[:]) / noise_scale
    net["inp"].rates = net.f_eff
    if dt_ms:
        from brian2 import defaultclock, ms
        defaultclock.dt = dt_ms * ms
    return net


def _set_gains(net, kA, kG):
    P_E = net["P_E"]
    P_E.k_AMPA_E_hsp = float(kA)
    P_E.k_GABA_E_hsp = float(kG)


def _rates(net, i0=0):
    n = np.asarray(net["m_cnt"].n)
    return n[0, i0:] / net["P_E"].N, n[1, i0:] / net["P_I"].N


def run_trial(seed, f_in, kA, kG, init, noise_seed, T, out_path, work_dir, cache_dir, stop_after_escape=None,
              noise_scale=1.0, dt_ms=None):
    try:
        worker._isolate(work_dir, cache_dir, template_cache_dir=RESULTS / "cython_cache" / "_template")
        from brian2 import second, Hz, seed as b2seed
        from HSP_Phase import _runkw
        net = _build(seed, f_in, noise_scale, dt_ms)
        _set_gains(net, kA, kG)
        b2seed(int(noise_seed))
        kw = _runkw(net)
        n_kick = 0
        if init == "high":
            net["inp"].rates = KICK_HZ * Hz
            net.run(KICK_S * second, **kw)
            net["inp"].rates = net.f_eff
            n_kick = int(KICK_S)
        if stop_after_escape is None:
            net.run(T * second, **kw)
        else:                       # chunks of 50 s; stop stop_after_escape s after the first high-rate bin
            done, esc = 0.0, None
            while done < T:
                step = min(50.0, T - done)
                net.run(step * second, **kw)
                done += step
                rE, _ = _rates(net, n_kick)
                if esc is None and (rE > HIGH_HZ).any():
                    esc = int(np.flatnonzero(rE > HIGH_HZ)[0])
                if esc is not None and len(rE) - esc >= stop_after_escape:
                    break
        rE, rI = _rates(net)
        P_E = net["P_E"]
        kA_end, kG_end = np.asarray(P_E.k_AMPA_E_hsp[:]), np.asarray(P_E.k_GABA_E_hsp[:])
        meta = dict(seed=seed, f_in=f_in, kA=kA, kG=kG, init=init, noise_seed=noise_seed, T=T, kick_bins=n_kick,
                    stopped_early=bool(len(rE) - n_kick < int(T)), noise_scale=noise_scale, dt_ms=dt_ms,
                    kA_drift=float(np.abs(kA_end - kA).max()), kG_drift=float(np.abs(kG_end - kG).max()))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, rate_E=rE, rate_I=rI, meta=json.dumps(meta))
        return dict(ok=True, **meta)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, out=str(out_path), error=f"{e}\n{traceback.format_exc()}")


def run_sweep(seed, f_in, kAs, hold_s, noise_seed, out_path, work_dir, cache_dir, kG_of=lambda k: 1.0 - k,
              kick=False, noise_scale=1.0, dt_ms=None):
    try:
        worker._isolate(work_dir, cache_dir, template_cache_dir=RESULTS / "cython_cache" / "_template")
        from brian2 import second, seed as b2seed
        from HSP_Phase import _runkw
        net = _build(seed, f_in, noise_scale, dt_ms)
        b2seed(int(noise_seed))
        kw = _runkw(net)
        n_kick = 0
        if kick:
            from brian2 import Hz
            _set_gains(net, kAs[0], kG_of(kAs[0]))
            net["inp"].rates = KICK_HZ * Hz
            net.run(KICK_S * second, **kw)
            net["inp"].rates = net.f_eff
            n_kick = int(KICK_S)
        for kA in kAs:
            _set_gains(net, kA, kG_of(kA))
            net.run(hold_s * second, **kw)
        rE, rI = _rates(net)
        meta = dict(seed=seed, f_in=f_in, kAs=list(map(float, kAs)), hold_s=hold_s, noise_seed=noise_seed,
                    kick_bins=n_kick, noise_scale=noise_scale, dt_ms=dt_ms)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, rate_E=rE, rate_I=rI, meta=json.dumps(meta))
        return dict(ok=True, **{k: v for k, v in meta.items() if k != "kAs"})
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, out=str(out_path), error=f"{e}\n{traceback.format_exc()}")


def full_jobs():
    """The full batch (~1.5-2 h on 14 cores). Long jobs first for load balancing."""
    out = RESULTS / "frozen_gain" / "full"
    J = []

    def trial(seed, f, kA, ns, T, init="low"):
        p = out / "trials" / f"s{seed}_f{f:g}_kA{kA:.3f}_{init}_n{ns}.npz"
        if not p.exists():
            J.append(("trial", dict(seed=seed, f_in=float(f), kA=float(kA), kG=1 - float(kA), init=init, noise_seed=ns,
                                    T=T, out_path=str(p), stop_after_escape=20)))

    def sweep(seed, f, kAs, hold, ns, direction):
        p = out / "sweeps" / f"s{seed}_f{f:g}_{direction}_n{ns}.npz"
        if not p.exists():
            J.append(("sweep", dict(seed=seed, f_in=float(f), kAs=kAs, hold_s=hold, noise_seed=ns,
                                    kick=(direction == "down"), out_path=str(p))))

    # 3. escape statistics (risk R_Delta, first-passage times), network 1, 1000 s, 20 noise seeds at 36 Hz
    for kA in np.round(np.arange(0.93, 1.0001, 0.01), 3):
        for ns in range(100, 120):
            trial(1, 36, kA, ns, 1000.0)
    for kA in (0.95, 0.97, 0.99, 1.00):           # healthy drive, same gains
        for ns in range(100, 110):
            trial(1, 40, kA, ns, 1000.0)
    # 4. networks 2-5 at 36 Hz: escape (10 noise seeds, 500 s) and sweeps
    for seed in (2, 3, 4, 5):
        for kA in (0.95, 0.97, 0.99):
            for ns in range(100, 110):
                trial(seed, 36, kA, ns, 500.0)
    # 1-2. regime map: up (from rest) and down (from a kicked high state) sweeps, kA 0.50-1.00, 8 input rates
    up = np.round(np.arange(0.50, 1.0001, 0.02), 3)
    for f in (32, 34, 35, 36, 37, 38, 40, 42):
        for ns in (31, 32):
            sweep(1, f, up, 50.0, ns, "up")
            sweep(1, f, up[::-1], 50.0, ns, "down")
    fine = np.round(np.arange(0.60, 1.0001, 0.01), 3)
    for seed in (1, 2, 3, 4, 5):                   # window edges at 36 Hz, finer step
        sweep(seed, 36, fine, 50.0, 41, "upfine")
        sweep(seed, 36, fine[::-1], 50.0, 41, "downfine")
    return J


def checks_jobs():
    """Follow-up batch: risk below k_E 0.93 (36 Hz), risk near the 40 Hz fixed point, input noise at fixed mean drive,
    and a 0.5 ms time step."""
    out = RESULTS / "frozen_gain" / "checks"
    J = []

    def trial(tag, f, kA, ns, T, **kw):
        p = out / tag / f"s1_f{f:g}_kA{kA:.3f}_n{ns}.npz"
        if not p.exists():
            J.append(("trial", dict(seed=1, f_in=float(f), kA=float(kA), kG=1 - float(kA), init="low", noise_seed=ns,
                                    T=T, out_path=str(p), stop_after_escape=20, **kw)))

    def sweep(tag, f, kAs, ns, direction, **kw):
        p = out / tag / f"s1_f{f:g}_{direction}_n{ns}.npz"
        if not p.exists():
            J.append(("sweep", dict(seed=1, f_in=float(f), kAs=kAs, hold_s=50.0, noise_seed=ns,
                                    kick=direction.startswith("down"), out_path=str(p), **kw)))

    for kA in (0.80, 0.85, 0.88, 0.90, 0.92):                          # 1. risk below 0.93 at 36 Hz
        for ns in range(100, 120):
            trial("risk36_low", 36, kA, ns, 1000.0)
    for kA in (0.77, 0.80, 0.83, 0.85):                                # 2. risk near the healthy fixed point
        for ns in range(100, 110):
            trial("risk40", 40, kA, ns, 1000.0)
    fine = np.round(np.arange(0.60, 1.0001, 0.01), 3)
    # 4. input noise at fixed mean drive: NOT run. Rescaling rate x weight at constant product moves the operating
    #    point itself (rest at 0.1 Hz for c = 2, immediate escape for c = 0.5 at k_E = 0.90), because rest firing is
    #    fluctuation-driven; a fair test needs the mean adjusted to keep the rest rate fixed.
    for kA in (0.95, 0.97, 0.99):                                      # 5. time step 0.5 ms
        for ns in range(100, 110):
            trial("dt05", 36, kA, ns, 1000.0, dt_ms=0.5)
    sweep("dt05", 36, fine, 41, "upfine", dt_ms=0.5)
    sweep("dt05", 36, fine[::-1], 41, "downfine", dt_ms=0.5)
    return J


def run_jobs(J, n_jobs):
    print(f"[frozen] {len(J)} job(s)", flush=True)
    from joblib import Parallel, delayed
    fn = dict(trial=run_trial, sweep=run_sweep)
    res = Parallel(n_jobs=min(n_jobs, max(len(J), 1)), backend="loky", verbose=5)(
        delayed(fn[k])(**a, work_dir=str(RESULTS / "work" / f"frozen_{Path(a['out_path']).stem}"),
                       cache_dir=str(RESULTS / "cython_cache" / f"frozen_{Path(a['out_path']).stem}"))
        for k, a in J)
    bad = [r for r in res if not r["ok"]]
    for r in bad:
        print("FAILED", r["out"], r["error"][-800:])
    print(f"[frozen] done, {len(bad)} failed", flush=True)


def pilot(n_jobs):
    out = RESULTS / "frozen_gain" / "pilot"
    jobs = []
    for kA in (0.80, 0.90, 0.95, 0.97, 0.99):
        for init in ("low", "high"):
            for ns in (11, 12, 13):
                p = out / f"trial_s1_f36_kA{kA:.2f}_{init}_n{ns}.npz"
                if not p.exists():
                    jobs.append(("trial", dict(seed=1, f_in=36.0, kA=kA, kG=1 - kA, init=init, noise_seed=ns, T=200.0,
                                               out_path=str(p))))
    up = np.round(np.arange(0.80, 1.0001, 0.01), 3)
    for ns in (21, 22):
        p = out / f"sweep_s1_f36_n{ns}.npz"
        if not p.exists():
            jobs.append(("sweep", dict(seed=1, f_in=36.0, kAs=np.r_[up, up[::-1]], hold_s=100.0, noise_seed=ns,
                                       out_path=str(p))))
    print(f"[pilot] {len(jobs)} job(s)", flush=True)
    from joblib import Parallel, delayed
    fn = dict(trial=run_trial, sweep=run_sweep)
    res = Parallel(n_jobs=min(n_jobs, max(len(jobs), 1)), backend="loky", verbose=5)(
        delayed(fn[k])(**a, work_dir=str(RESULTS / "work" / f"frozen_{Path(a['out_path']).stem}"),
                       cache_dir=str(RESULTS / "cython_cache" / f"frozen_{Path(a['out_path']).stem}"))
        for k, a in jobs)
    for r in res:
        if not r["ok"]:
            print("FAILED", r["out"], r["error"][-800:])


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pilot", "full", "checks"])
    ap.add_argument("--n-jobs", type=int, default=14)
    a = ap.parse_args()
    pilot(a.n_jobs) if a.what == "pilot" else run_jobs(full_jobs() if a.what == "full" else checks_jobs(), a.n_jobs)
