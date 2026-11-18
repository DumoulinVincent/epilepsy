"""Seizure clamp: HSP running, every escape from rest ended from outside after D seconds.

Stand-in for a fast seizure-termination mechanism (the Epileptor's z) that the network itself lacks: with frozen gains
the high-rate state never ends by itself (frozen_gain.py), and with HSP running it ends only because HSP crushes the
gains. Here, once the E rate has been above 10 Hz for D consecutive 1 s bins, every membrane potential is set to V_L
and the recurrent synaptic gates to 0; the rate sensors (x_spike) are left alone, so HSP sees the whole seizure.
While rest exists (k_E below the upper edge of the bistable window), the reset returns the network to rest.

Start: baseline network, per-cell gains snapshotted from the running 36 Hz cycle when the E-mean k_AMPA first reached
k_start (frozen_gain.py cycle_gains), 300 s with HSP frozen so the rate sensors settle, then HSP on with all four time
constants tau (rate sensor 100 s).

Question: with HSP much slower than a seizure, does one seizure end the high-risk period, or do seizures cluster
(several per excursion), or does k_E settle where seizure-driven downscaling balances the homeostatic rise?

    python -u seizure_clamp.py first --n-jobs 15     # start below the risk zone (results/seizure_clamp/raw/)
    python -u seizure_clamp.py inzone --n-jobs 15    # start inside it, 100000 s, saved every 5000 s
    python -u seizure_clamp.py deaff --n-jobs 10     # from deafferentation, resumable (rerun the same command)

Each output: rate_E (1 s bins), t_k / k_AMPA / k_GABA (E means, every 10 s), clamp times (s, after the warm-up).
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
import worker  # noqa: E402
from frozen_gain import _set_gains  # noqa: E402

RESULTS = THIS_DIR / "results"
SNAP = RESULTS / "frozen_gain" / "cycle_gains" / "snapshots"
HIGH_HZ = 10.0
WARM_S = 300.0


def run_clamp(seed, f_in, tau, D, k_start, run_s, noise_seed, out_path, work_dir, cache_dir, gains=None,
              save_every=5000.0, tau_hsp=100.0):
    try:
        worker._isolate(work_dir, cache_dir, template_cache_dir=RESULTS / "cython_cache" / "_template")
        from brian2 import (second, mV, Hz, NeuronGroup, Synapses, StateMonitor, network_operation,
                            seed as b2seed)
        from seeded_network import build_network_seeded
        from HSP_Phase import _runkw, _freeze_hsp
        net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second, input_freq=f_in,
                                   network_seed=seed)
        net.restore("baseline", filename=str(RESULTS / "baselines" / f"seed{seed}.bri"))
        for mon in [o for o in net.objects if isinstance(o, StateMonitor)] + [net["sp_exc"], net["sp_inh"]]:
            net.remove(mon)
        P_E, P_I = net["P_E"], net["P_I"]
        net["inp"].rates = f_in * Hz
        if gains is None:
            g = np.load(SNAP / f"s{seed}_f36_tau250.npz")
            i = int(np.flatnonzero(np.isclose(g["targets"], k_start))[0])
            gains = (g["k_AMPA"][i], g["k_GABA"][i])
        _set_gains(net, *gains)

        cnt = NeuronGroup(1, "n : 1", name="cnt")
        cE = Synapses(P_E, cnt, on_pre="n_post += 1", name="cnt_E"); cE.connect(j="0")
        m_cnt = StateMonitor(cnt, "n", record=True, dt=1 * second, when="end", order=0, name="m_cnt")
        cnt.run_regularly("n = 0", dt=1 * second, when="end", order=2)
        m_k = StateMonitor(P_E, ["k_AMPA_E_hsp", "k_GABA_E_hsp"], record=True, dt=10 * second, name="m_k")
        st = dict(high=0, clamps=[], on=False)
        nE = int(P_E.N)

        @network_operation(dt=1 * second, when="end", order=1)
        def clamp(t):
            if not st["on"]:
                return
            if cnt.n[0] / nE > HIGH_HZ:
                st["high"] += 1
            else:
                st["high"] = 0
            if st["high"] >= D:
                for P in (P_E, P_I):
                    P.v = -70 * mV
                    P.s_AMPA = 0
                    P.s_GABA = 0
                st["clamps"].append(float(t / second))
                st["high"] = 0

        net.add(cnt, cE, m_cnt, m_k, clamp)
        b2seed(int(noise_seed))
        kw = _runkw(net)
        _freeze_hsp(net)
        st["on"] = True                                      # clamp active in the warm-up too (negative times)
        net.run(WARM_S * second, **kw)                       # rate sensors settle on the snapshot gains
        for nm in worker.HSP_TAUS:
            setattr(P_E, nm, tau * second)
        if tau_hsp != 100.0:
            # slower rate sensor (seeded_network.tau_HSP, 100 s, read from the namespace); x_spike rescaled so that
            # r_est = x_spike / tau_HSP is continuous at the switch (as in long_tau_cycle.py)
            kw["namespace"] = dict(kw["namespace"], tau_HSP=tau_hsp * second)
            P_E.x_spike = P_E.x_spike[:] * (tau_hsp / 100.0)
            P_I.x_spike = P_I.x_spike[:] * (tau_hsp / 100.0)
        st["on"] = True
        t0 = float(net.t / second)
        meta = dict(seed=seed, f_in=f_in, tau=tau, D=D, k_start=k_start, run_s=run_s, noise_seed=noise_seed,
                    warm_s=WARM_S, t0=t0, tau_hsp=tau_hsp)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        def save(done):                 # rewritten after every segment, so a crash keeps what was simulated
            n = np.asarray(m_cnt.n)[0] / nE
            np.savez_compressed(out_path, rate_E=n, rate_t=np.arange(len(n)) - WARM_S + 0.5,
                                t_k=np.asarray(m_k.t / second) - t0, k_AMPA=np.asarray(m_k.k_AMPA_E_hsp).mean(0),
                                k_GABA=np.asarray(m_k.k_GABA_E_hsp).mean(0), clamps=np.array(st["clamps"]) - t0,
                                meta=json.dumps(dict(meta, done_s=done, complete=done >= run_s)))

        done = 0.0
        while done < run_s:
            step = min(save_every, run_s - done)
            net.run(step * second, **kw)
            done += step
            save(done)
        return dict(ok=True, out=str(out_path), n_clamps=len(st["clamps"]))
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, out=str(out_path), error=f"{e}\n{traceback.format_exc()}")


def run_deaff(seed, D, tau_up, tau_down, tau_hsp, run_s, noise_seed, out_path, work_dir, cache_dir, f_in=36.0,
              ckpt_every=10000.0, rec_k_dt=50.0, stop_after_segments=None):
    """Deafferentation with the seizure clamp, resumable.

    Start from the healthy 40 Hz state (baseline gains, k_AMPA ~0.75), drive f_in, 300 s with HSP frozen (rate sensors
    settle; clamp active), then HSP on: k_E rises with tau_up (AMPA insertion, GABA internalization) and falls with
    tau_down (AMPA internalization, GABA insertion); rate sensor tau_hsp. Every seizure ended after D s.

    Every ckpt_every s the full network state (including monitors and the random-number state) is written to
    <out>.ckpt with the clamp's own state in <out>.ckpt.json, and the results so far to <out>.npz. Calling the
    function again with the same arguments continues from the last checkpoint; the checkpoint is removed when the run
    is complete. stop_after_segments: stop early (tests)."""
    out_path = Path(out_path)
    ck, ckj = out_path.with_suffix(".ckpt"), out_path.with_suffix(".ckpt.json")
    try:
        worker._isolate(work_dir, cache_dir, template_cache_dir=RESULTS / "cython_cache" / "_template")
        from brian2 import (second, mV, Hz, NeuronGroup, Synapses, StateMonitor, network_operation,
                            seed as b2seed)
        from seeded_network import build_network_seeded
        from HSP_Phase import _runkw, _freeze_hsp
        net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second, input_freq=f_in,
                                   network_seed=seed)
        net.restore("baseline", filename=str(RESULTS / "baselines" / f"seed{seed}.bri"))
        for mon in [o for o in net.objects if isinstance(o, StateMonitor)] + [net["sp_exc"], net["sp_inh"]]:
            net.remove(mon)
        P_E, P_I = net["P_E"], net["P_I"]
        net["inp"].rates = f_in * Hz
        cnt = NeuronGroup(1, "n : 1", name="cnt")
        cE = Synapses(P_E, cnt, on_pre="n_post += 1", name="cnt_E"); cE.connect(j="0")
        m_cnt = StateMonitor(cnt, "n", record=True, dt=1 * second, when="end", order=0, name="m_cnt")
        cnt.run_regularly("n = 0", dt=1 * second, when="end", order=2, name="cnt_reset")
        m_k = StateMonitor(P_E, ["k_AMPA_E_hsp", "k_GABA_E_hsp"], record=True, dt=rec_k_dt * second, name="m_k")
        st = dict(high=0, clamps=[], on=True, done=0.0, t0=None)
        nE = int(P_E.N)

        @network_operation(dt=1 * second, when="end", order=1, name="clamp_op")
        def clamp(t):
            if cnt.n[0] / nE > HIGH_HZ:
                st["high"] += 1
            else:
                st["high"] = 0
            if st["high"] >= D:
                for P in (P_E, P_I):
                    P.v = -70 * mV
                    P.s_AMPA = 0
                    P.s_GABA = 0
                st["clamps"].append(float(t / second))
                st["high"] = 0

        net.add(cnt, cE, m_cnt, m_k, clamp)
        kw = _runkw(net)
        kw["namespace"] = dict(kw["namespace"], tau_HSP=tau_hsp * second)
        if ck.exists() and ckj.exists():                        # continue an interrupted run
            net.restore("ck", filename=str(ck), restore_random_state=True)
            st.update(json.loads(ckj.read_text()))
            print(f"[deaff {out_path.stem}] resumed at {st['done']:g} s", flush=True)
        else:
            b2seed(int(noise_seed))
            _freeze_hsp(net)
            net.run(WARM_S * second, **dict(kw, namespace=dict(kw["namespace"], tau_HSP=100 * second)))
            for nm, v in zip(worker.HSP_TAUS, (tau_up, tau_down, tau_down, tau_up)):  # ins_A, int_A, ins_G, int_G
                setattr(P_E, nm, v * second)
            P_E.x_spike = P_E.x_spike[:] * (tau_hsp / 100.0)    # r_est = x_spike / tau_HSP continuous at the switch
            P_I.x_spike = P_I.x_spike[:] * (tau_hsp / 100.0)
            st["t0"] = float(net.t / second)
        meta = dict(seed=seed, f_in=f_in, D=D, tau_up=tau_up, tau_down=tau_down, tau_hsp=tau_hsp, run_s=run_s,
                    noise_seed=noise_seed, warm_s=WARM_S, t0=st["t0"], start="deafferentation from 40 Hz baseline")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def save():
            n = np.asarray(m_cnt.n)[0] / nE
            np.savez_compressed(out_path, rate_E=n, rate_t=np.arange(len(n)) - WARM_S + 0.5,
                                t_k=np.asarray(m_k.t / second) - st["t0"],
                                k_AMPA=np.asarray(m_k.k_AMPA_E_hsp).mean(0),
                                k_GABA=np.asarray(m_k.k_GABA_E_hsp).mean(0),
                                k_AMPA_cells=np.asarray(m_k.k_AMPA_E_hsp, dtype=np.float32),
                                clamps=np.array(st["clamps"]) - st["t0"],
                                meta=json.dumps(dict(meta, done_s=st["done"], complete=st["done"] >= run_s)))

        segs = 0
        while st["done"] < run_s:
            step = min(ckpt_every, run_s - st["done"])
            net.run(step * second, **kw)
            st["done"] += step
            tmp = ck.with_suffix(".ckpt.tmp")
            if tmp.exists():
                tmp.unlink()
            net.store("ck", filename=str(tmp))
            os.replace(tmp, ck)
            ckj.write_text(json.dumps(st))
            save()
            segs += 1
            print(f"[deaff {out_path.stem}] {st['done']:g}/{run_s:g} s, {len(st['clamps'])} seizures", flush=True)
            if stop_after_segments and segs >= stop_after_segments and st["done"] < run_s:
                return dict(ok=True, out=str(out_path), stopped=True, done=st["done"])
        for p in (ck, ckj):
            if p.exists():
                p.unlink()
        return dict(ok=True, out=str(out_path), n_clamps=len(st["clamps"]))
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, out=str(out_path), error=f"{e}\n{traceback.format_exc()}")


def deaff_jobs():
    """Clusters after deafferentation, 5 networks, seizures ended after 10 s (settings chosen with event_model.py):
    tau_up 10000 s, tau_down 5000 s, sensor 2000 s, 500000 s; tau_up = tau_down = 5000 s, sensor 2500 s, 250000 s."""
    out = RESULTS / "seizure_clamp" / "deaff"
    J = []
    for tu, td, th, T in ((10000.0, 5000.0, 2000.0, 500000.0), (5000.0, 5000.0, 2500.0, 250000.0)):
        for seed in (1, 2, 3, 4, 5):
            p = out / f"s{seed}_up{tu:g}_down{td:g}_sensor{th:g}_D10_{T:g}s.npz"
            if not _complete(p):
                J.append(dict(seed=seed, D=10, tau_up=tu, tau_down=td, tau_hsp=th, run_s=T, noise_seed=900 + seed,
                              out_path=str(p)))
    return J


def _complete(p):
    return p.exists() and json.loads(str(np.load(p)["meta"])).get("complete", True)


def jobs(stage, run_s):
    """first: start at k_E 0.92 (below the risk zone), tau 2500 s (D 10 s) and 25000 s (D 10, 30 s), 50000 s.
    inzone: start inside the risk zone (0.95), D 10 s, 100000 s, rate sensor slowed with the gains (tau_hsp = tau/10):
        tau 25000 s / sensor 2500 s, tau 10000 s / sensor 1000 s, and tau 25000 s with the default 100 s sensor."""
    out = RESULTS / "seizure_clamp" / "raw"
    if stage == "first":
        grid = [(2500.0, 10, 0.92, 100.0), (25000.0, 10, 0.92, 100.0), (25000.0, 30, 0.92, 100.0)]
    else:
        grid = [(25000.0, 10, 0.95, 2500.0), (10000.0, 10, 0.95, 1000.0), (25000.0, 10, 0.95, 100.0)]
    J = []
    for tau, D, k0, th in grid:
        for seed in (1, 2, 3, 4, 5):
            sens = "" if th == 100.0 else f"_sensor{th:g}"
            p = out / f"s{seed}_f36_tau{tau:g}{sens}_D{D}_k{k0:.2f}_{run_s:g}s.npz"
            if not _complete(p):
                J.append(dict(seed=seed, f_in=36.0, tau=tau, D=D, k_start=k0, run_s=run_s, noise_seed=700 + seed,
                              out_path=str(p), tau_hsp=th))
    return J


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="first", choices=["first", "inzone", "deaff"])
    ap.add_argument("--run-s", type=float, default=None, help="default 50000 (first) or 100000 (inzone)")
    ap.add_argument("--n-jobs", type=int, default=15)
    a = ap.parse_args()
    if a.stage == "deaff":
        J, fn = deaff_jobs(), run_deaff
    else:
        J, fn = jobs(a.stage, a.run_s or (50000.0 if a.stage == "first" else 100000.0)), run_clamp
    print(f"[clamp] {len(J)} job(s)", flush=True)
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=min(a.n_jobs, max(len(J), 1)), backend="loky", verbose=5)(
        delayed(fn)(**j, work_dir=str(RESULTS / "work" / f"clamp_{Path(j['out_path']).stem}"),
                           cache_dir=str(RESULTS / "cython_cache" / f"clamp_{Path(j['out_path']).stem}"))
        for j in J)
    bad = [r for r in res if not r["ok"]]
    for r in bad:
        print("FAILED", r["out"], r["error"][-800:])
    print(f"[clamp] done, {len(bad)} failed", flush=True)
