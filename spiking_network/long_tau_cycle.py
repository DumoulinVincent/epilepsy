"""Long runs with slow HSP: the untreated 36 Hz network (and the 40 Hz
reference) over many high-/low-risk cycles, for the poster's cycle figure.

Same network and baseline as lfp_cycle.py, but the four HSP insertion /
internalization time constants are set per job (tau_ins_AMPA, tau_int_AMPA,
tau_ins_GABA, tau_int_GABA, s) and nothing heavy is recorded: the network's
own monitors are removed and replaced by 1 s population spike counts (E, I)
and the E-mean HSP gains every 10 s, so 10^5 s runs stay small.

    python -u long_tau_cycle.py --tau 2500 --run-s 150000 --seeds 1,2,3
    python -u long_tau_cycle.py --tau 2500 --tau-int-ampa 3750 --tau-ins-gaba 3750 ...
    python -u long_tau_cycle.py --tau 1000 --tau-hsp 1000 ...     # slower rate sensor (default 100 s)

Saves results/long_tau/raw/<tag>_seed<s>_<cond>.npz:
  rate_t, rate_E, rate_I  -- 1 s bins (Hz per cell)
  t_k, k_AMPA, k_GABA     -- E-mean gains every 10 s
"""
import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import worker  # noqa: E402
import conditions as cond_mod  # noqa: E402
from lfp_cycle import _reporter  # noqa: E402

RESULTS = THIS_DIR / "results"


def find_condition(name):
    """A conditions.CYCLE_CONDITIONS entry, or any "DBS XpA" built like them (36 Hz drive)."""
    for c in cond_mod.CYCLE_CONDITIONS:
        if c["name"] == name:
            return c
    u = re.fullmatch(r"untreated (\d+(?:\.\d+)?)Hz", name)       # untreated at another input rate
    if u:
        return dict(cond_mod.make_condition(name), freq_hz=float(u.group(1)))
    m = re.fullmatch(r"DBS (\d+(?:\.\d+)?)pA", name)
    if not m:
        raise ValueError(f"unknown condition {name!r}")
    return dict(cond_mod.make_condition(name, drug={"dbs": True, "dbs_freq_hz": cond_mod.DBS_FREQ_HZ,
                                                    "dbs_amp_pA": float(m.group(1))}),
                freq_hz=cond_mod.BURST_FREQ_HZ)


def run_long_job(seed, cond_name, taus, run_s, out_path, work_dir, cache_dir, tau_hsp=None):
    tag = f"long {Path(out_path).stem}"
    try:
        worker._isolate(work_dir, cache_dir,
                        template_cache_dir=str(RESULTS / "cython_cache" / "_template"))
        from brian2 import (second, ms, Hz, NeuronGroup, Synapses, StateMonitor,
                            seed as b2seed)
        from seeded_network import build_network_seeded
        from HSP_Phase import _apply_drugs, _runkw

        cond = find_condition(cond_name)
        freq = float(cond["freq_hz"])
        net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second,
                                   input_freq=freq, network_seed=seed)
        net.restore("baseline", filename=str(RESULTS / "baselines" / f"seed{seed}.bri"))
        P_E, P_I = net["P_E"], net["P_I"]
        for nm, v in zip(worker.HSP_TAUS, taus):
            setattr(P_E, nm, v * second)
        _apply_drugs(net, cond.get("drug") or {})
        net["inp"].rates = freq * Hz
        for mon in [o for o in net.objects if isinstance(o, StateMonitor)] + \
                [net["sp_exc"], net["sp_inh"]]:
            net.remove(mon)

        # 1 s population spike counts, reset after each sample
        cnt = NeuronGroup(2, "n : 1", name="cnt")
        cE = Synapses(P_E, cnt, on_pre="n_post += 1", name="cnt_E")
        cE.connect(j="0")
        cI = Synapses(P_I, cnt, on_pre="n_post += 1", name="cnt_I")
        cI.connect(j="1")
        m_cnt = StateMonitor(cnt, "n", record=True, dt=1 * second, when="end", order=0,
                             name="m_cnt")
        cnt.run_regularly("n = 0", dt=1 * second, when="end", order=1)
        m_k = StateMonitor(P_E, ["k_AMPA_E_hsp", "k_GABA_E_hsp"], record=True,
                           dt=10 * second, name="m_k")
        net.add(cnt, cE, cI, m_cnt, m_k)

        b2seed(3000 + int(seed))
        t0 = float(net.t / second)
        print(f"[{tag}] start: {run_s:g} s at {freq:g} Hz, taus {taus}", flush=True)
        kw = _runkw(net)
        if tau_hsp:
            # rate-sensor time constant (seeded_network.tau_HSP, 100 s, taken from the namespace);
            # x_spike is rescaled so that r_est = x_spike / tau_HSP is continuous at the switch
            kw["namespace"] = dict(kw["namespace"], tau_HSP=tau_hsp * second)
            P_E.x_spike = P_E.x_spike[:] * (tau_hsp / 100.0)
            P_I.x_spike = P_I.x_spike[:] * (tau_hsp / 100.0)
        net.run(run_s * second, report=_reporter(tag, every=0.02), report_period=60 * second, **kw)

        n = np.asarray(m_cnt.n)
        tk = np.asarray(m_k.t / second) - t0
        meta = dict(seed=int(seed), condition=cond_name, freq_hz=freq, run_s=run_s, t0=t0,
                    taus=dict(zip(worker.HSP_TAUS, taus)), tau_hsp=tau_hsp or 100.0)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, rate_t=np.asarray(m_cnt.t / second) - t0 - 0.5,
                            rate_E=n[0] / P_E.N, rate_I=n[1] / P_I.N,
                            t_k=tk, k_AMPA=np.asarray(m_k.k_AMPA_E_hsp).mean(0),
                            k_GABA=np.asarray(m_k.k_GABA_E_hsp).mean(0), meta=json.dumps(meta))
        print(f"[{tag}] done -> {out_path}", flush=True)
        return dict(ok=True, out=str(out_path))
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] FAILED: {e}\n{traceback.format_exc()}", flush=True)
        return dict(ok=False, out=str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True, help="all four HSP taus (s)")
    ap.add_argument("--tau-int-ampa", type=float, default=None)
    ap.add_argument("--tau-ins-gaba", type=float, default=None)
    ap.add_argument("--tau-hsp", type=float, default=None, help="rate-sensor time constant (s; default 100)")
    ap.add_argument("--run-s", type=float, required=True)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--conditions", default="untreated")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS / "long_tau"))
    args = ap.parse_args()
    # order of worker.HSP_TAUS: ins_AMPA, int_AMPA, ins_GABA, int_GABA
    taus = (args.tau, args.tau_int_ampa or args.tau, args.tau_ins_gaba or args.tau, args.tau)
    tag = f"tau{args.tau:g}" + ("" if taus[1] == taus[2] == args.tau
                                else f"_intA{taus[1]:g}_insG{taus[2]:g}")
    tag += f"_tauhsp{args.tau_hsp:g}" if args.tau_hsp else ""
    raw = Path(args.out).resolve() / "raw"
    jobs = []
    for s in [int(x) for x in args.seeds.split(",")]:
        for c in [c.strip() for c in args.conditions.split(",")]:
            out = raw / f"{tag}_{worker._slug(f'seed{s}_{c}')}_{args.run_s:g}s.npz"
            if out.exists():
                print(f"[run] cached {out}")
                continue
            slug = out.stem
            jobs.append(dict(seed=s, cond_name=c, taus=taus, run_s=args.run_s, out_path=str(out), tau_hsp=args.tau_hsp,
                             work_dir=str(RESULTS / "work" / f"long_{slug}"),
                             cache_dir=str(RESULTS / "cython_cache" / f"long_{slug}")))
    print(f"[run] {len(jobs)} job(s), taus {taus}", flush=True)
    if jobs:
        from joblib import Parallel, delayed
        Parallel(n_jobs=min(args.n_jobs, len(jobs)), backend="loky", verbose=10)(
            delayed(run_long_job)(**j) for j in jobs)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
