"""Does pairing by probe seed cancel the IPC probe noise? And is DBS on during
the probes?

    python paired_probe_check.py --seed 3 --conditions "untreated,DBS 10pA,DBS 30pA"

For one network seed, every condition goes through the same steps as
worker.run_condition_job (local baseline, plateau snapshot, HSP frozen,
COARSE_GRID probes with the same probe seeds), minus characterize_cycle, which
only feeds the cycle stats. Prints, per condition, the per-probe IPC, then the
probe-to-probe SD of each condition minus untreated: unpaired (two independent
sets of probes) vs paired (same probe seed subtracted).

DBS check, for every DBS condition: after each probe, the dbs Poisson rate and
DBS_E.w_dbs as restored from the snapshot, and the mean DBS current into E
cells over the probe (I_dbs recorded at every time step). One extra probe is
then rerun with w_dbs = 0 on the same snapshot and probe seed -- if DBS was on,
the E rate changes.

Writes results/ipc_probe_inspection/paired_seed{N}.json.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import worker  # noqa: E402  (puts the epilepsy dir on sys.path)
import conditions as cond_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--conditions", default="untreated,DBS 10pA,DBS 30pA")
    args = ap.parse_args()
    names = [c.strip() for c in args.conditions.split(",")]
    if "untreated" not in names:
        names.insert(0, "untreated")
    g = cond_mod.COARSE_GRID
    results = THIS_DIR / "results"
    out = results / "ipc_probe_inspection"
    out.mkdir(parents=True, exist_ok=True)
    work = results / "work" / f"paired_seed{args.seed}"
    worker._isolate(work, results / "cython_cache" / f"paired_seed{args.seed}",
                    template_cache_dir=results / "cython_cache" / "_template")

    from brian2 import second, Hz, pA, StateMonitor
    from seeded_network import build_network_seeded
    from HSP_Phase import _ipc_probe
    from hsp_capacity import make_input_signal
    import coarse_phase as cp

    print(f"seed {args.seed}; probe {g['probe_r0_hz']:g} +- {g['amp_hz']:g} Hz, "
          f"{g['n_steps']} x {g['step_s']:g} s, probe seeds {g['probe_seeds']}", flush=True)
    per = {}
    for name in names:
        cond = next(c for c in cond_mod.CONDITIONS if c["name"] == name)
        net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second,
                                   input_freq=cond_mod.BURST_FREQ_HZ, network_seed=args.seed)
        is_dbs = bool(cond["drug"].get("dbs"))
        worker.prepare_local_baseline(net, results / "baselines" / f"seed{args.seed}.bri",
                                      work, g["hsp_tau_s"], cond["weights"])
        print(f"[{name}] plateau snapshot ...", flush=True)
        snap = cp._plateau_snapshot(net, cond_mod.BURST_FREQ_HZ, cond["drug"],
                                    g["plateau_window_s"], g["plateau_tol"],
                                    g["plateau_chunk_s"], g["plateau_max_s"], "snap")
        if not snap["found"]:
            print(f"[{name}] no snapshot: {snap['status']}", flush=True)
            continue
        t_snap = float(net.t / second)
        if is_dbs:
            # every-step I_dbs of 10 E cells, added after the snapshot (the
            # baseline files don't contain it) and stored with it again, so
            # each probe's restore empties it
            mon = StateMonitor(net["P_E"], "I_dbs", record=list(range(10)), name="I_dbs_mon")
            net.add(mon)
            net.store("snap")
        rec = dict(settle_s=snap["settle_s"], k_AMPA=snap["k_AMPA"], k_GABA=snap["k_GABA"],
                   ipc=[], E=[])
        for s in g["probe_seeds"]:
            u = make_input_signal(g["n_steps"], seed=s)
            r = _ipc_probe(net, "snap", u, r0_hz=g["probe_r0_hz"], amp_hz=g["amp_hz"],
                           step_s=g["step_s"], max_delay=g["max_delay"], rng_seed=s)
            rec["ipc"].append(r["ipc_total"])
            rec["E"].append(r["E_rate_probe"])
            line = f"[{name}] probe {s}: IPC {r['ipc_total']:.3f}, E {r['E_rate_probe']:.2f} Hz"
            if is_dbs:
                t = np.asarray(mon.t / second)
                i_dbs = np.asarray(mon.I_dbs / pA)[:, t >= t_snap]
                line += (f"; dbs rate {float(net['dbs'].rates[0] / Hz):g} Hz, w_dbs "
                         f"{float(net['DBS_E'].w_dbs[0] / pA):g} pA, mean I_dbs over probe "
                         f"{i_dbs.mean():.2f} pA")
                rec.setdefault("I_dbs_pA", []).append(float(i_dbs.mean()))
            print(line, flush=True)
        if is_dbs:
            s = g["probe_seeds"][0]
            net.restore("snap")
            net["DBS_E"].w_dbs = 0 * pA
            net.store("snap_nodbs")
            r = _ipc_probe(net, "snap_nodbs", make_input_signal(g["n_steps"], seed=s),
                           r0_hz=g["probe_r0_hz"], amp_hz=g["amp_hz"], step_s=g["step_s"],
                           max_delay=g["max_delay"], rng_seed=s)
            rec["control_w0"] = dict(probe_seed=s, ipc=r["ipc_total"], E=r["E_rate_probe"])
            print(f"[{name}] control, same snapshot + probe {s} with w_dbs = 0: "
                  f"E {r['E_rate_probe']:.2f} Hz (vs {rec['E'][0]:.2f} with DBS), "
                  f"IPC {r['ipc_total']:.3f} (vs {rec['ipc'][0]:.3f})", flush=True)
        per[name] = rec
        del net
        gc.collect()   # drop the old objects so the next build gets the same auto-names

    print("\nper-probe IPC (probe seeds " + "/".join(map(str, g["probe_seeds"])) + ")")
    for name, rec in per.items():
        v = np.array(rec["ipc"])
        print(f"  {name:12s} {np.round(v, 3)}  mean {v.mean():.3f}  SD {v.std(ddof=1):.3f}")
    if "untreated" in per:
        u = np.array(per["untreated"]["ipc"])
        print("\ncondition - untreated, probe-to-probe SD of the difference:")
        for name, rec in per.items():
            if name == "untreated":
                continue
            v = np.array(rec["ipc"])
            unp = np.sqrt(v.var(ddof=1) + u.var(ddof=1))
            d = v - u
            print(f"  {name:12s} diff mean {d.mean():+.3f}; SD unpaired {unp:.3f}, "
                  f"paired {d.std(ddof=1):.3f}; per probe {np.round(d, 3)}")
    path = out / f"paired_seed{args.seed}.json"
    path.write_text(json.dumps(per, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
