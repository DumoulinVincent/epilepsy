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


def _build(seed, f_in):
    """Restored, HSP-frozen network with light monitors: 1 s E/I spike counts."""
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
    net["inp"].rates = f_in * Hz
    return net


def _set_gains(net, kA, kG):
    P_E = net["P_E"]
    P_E.k_AMPA_E_hsp = float(kA)
    P_E.k_GABA_E_hsp = float(kG)


def _rates(net, i0=0):
    n = np.asarray(net["m_cnt"].n)
    return n[0, i0:] / net["P_E"].N, n[1, i0:] / net["P_I"].N


def run_trial(seed, f_in, kA, kG, init, noise_seed, T, out_path, work_dir, cache_dir):
    try:
        worker._isolate(work_dir, cache_dir, template_cache_dir=RESULTS / "cython_cache" / "_template")
        from brian2 import second, Hz, seed as b2seed
        from HSP_Phase import _runkw
        net = _build(seed, f_in)
        _set_gains(net, kA, kG)
        b2seed(int(noise_seed))
        kw = _runkw(net)
        n_kick = 0
        if init == "high":
            net["inp"].rates = KICK_HZ * Hz
            net.run(KICK_S * second, **kw)
            net["inp"].rates = f_in * Hz
            n_kick = int(KICK_S)
        net.run(T * second, **kw)
        rE, rI = _rates(net)
        P_E = net["P_E"]
        kA_end, kG_end = np.asarray(P_E.k_AMPA_E_hsp[:]), np.asarray(P_E.k_GABA_E_hsp[:])
        meta = dict(seed=seed, f_in=f_in, kA=kA, kG=kG, init=init, noise_seed=noise_seed, T=T, kick_bins=n_kick,
                    kA_drift=float(np.abs(kA_end - kA).max()), kG_drift=float(np.abs(kG_end - kG).max()))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, rate_E=rE, rate_I=rI, meta=json.dumps(meta))
        return dict(ok=True, **meta)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, out=str(out_path), error=f"{e}\n{traceback.format_exc()}")


def run_sweep(seed, f_in, kAs, hold_s, noise_seed, out_path, work_dir, cache_dir, kG_of=lambda k: 1.0 - k):
    try:
        worker._isolate(work_dir, cache_dir, template_cache_dir=RESULTS / "cython_cache" / "_template")
        from brian2 import second, seed as b2seed
        from HSP_Phase import _runkw
        net = _build(seed, f_in)
        b2seed(int(noise_seed))
        kw = _runkw(net)
        for kA in kAs:
            _set_gains(net, kA, kG_of(kA))
            net.run(hold_s * second, **kw)
        rE, rI = _rates(net)
        meta = dict(seed=seed, f_in=f_in, kAs=list(map(float, kAs)), hold_s=hold_s, noise_seed=noise_seed)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, rate_E=rE, rate_I=rI, meta=json.dumps(meta))
        return dict(ok=True, **{k: v for k, v in meta.items() if k != "kAs"})
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, out=str(out_path), error=f"{e}\n{traceback.format_exc()}")


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
    ap.add_argument("what", choices=["pilot"])
    ap.add_argument("--n-jobs", type=int, default=14)
    a = ap.parse_args()
    pilot(a.n_jobs)
