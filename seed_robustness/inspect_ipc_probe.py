"""Look inside one IPC computation: where the snapshot lands, what the network
does during each probe (the signal-reconstruction run), and what a runaway
probe is.

    python inspect_ipc_probe.py --seed 3 --condition "DBS 10pA"
    python inspect_ipc_probe.py --seed 3 --condition untreated --amp 2

--r0 / --amp override the probe input (default: conditions.COARSE_GRID; r0
defaults to the network's own drive). Output names carry r0 and amp so test
settings don't overwrite each other.

Runs exactly what worker.run_condition_job does for one (seed, condition) --
local baseline (HSP time constants, weight intervention), characterize_cycle,
coarse_phase._plateau_snapshot, then HSP_Phase._ipc_probe for every probe
seed -- and plots, into results/ipc_probe_inspection/:

  *_context.png, *_probes.png -- drawn by probe_plots.py (the same figures
                    every run_seed_check.py job saves), data in *.npz:
  *_context.png  -- HSP ON (normal simulation, NOT the IPC): E and I rate, mean
                    k_AMPA / k_GABA from restore to the snapshot (and inspect_extend_s
                    beyond it, if set). The snapshot is the dashed line; the window the
                    plateau rule checked is shaded.
  *_probes.png   -- HSP FROZEN (the IPC computation): per probe, input rate,
                    E and I rate (50 ms bins), and k_AMPA / k_GABA (mean and
                    min-max over E cells) to show they do not move.
  *_runaway.png  -- only if a probe ran away: (a) input cut to 0 Hz with HSP
                    still frozen, then (b) HSP switched back on with the input
                    restored.
"""
import argparse
import os
import sys

# One BLAS thread: the IPC readout is thousands of small ridge solves, and
# several of these scripts in parallel with 16 OpenBLAS threads each thrash
# (joblib already does this for run_seed_check.py's workers).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import worker  # noqa: E402  (puts the epilepsy dir on sys.path)
import conditions as cond_mod  # noqa: E402
import probe_plots  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_E, N_I = 100, 25


def binned(spike_t, n, t0, t1, bin_s):
    edges = np.arange(t0, t1 + 1e-9, bin_s)
    c, _ = np.histogram(spike_t, bins=edges)
    return edges[:-1], c / (n * bin_s)


def spikes(net):
    from brian2 import second
    return (np.asarray(net["sp_exc"].t / second), np.asarray(net["sp_inh"].t / second))


def k_traces(net, t0, t1):
    """(t, kA[neuron, t], kG[neuron, t]) from the 200 ms k monitors."""
    from brian2 import second
    m = net["k_AMPA_E_hsp"]
    t = np.asarray(m.t / second)
    sel = (t >= t0) & (t <= t1)
    return (t[sel], np.asarray(m.k_AMPA_E_hsp)[:, sel],
            np.asarray(net["k_GABA_E_hsp"].k_GABA_E_hsp)[:, sel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--condition", default="DBS 10pA",
                    help='a conditions.CONDITIONS name at 36 Hz, or "reference" for '
                         'the untreated 40 Hz reference')
    ap.add_argument("--r0", type=float, default=None,
                    help="probe mean input rate in Hz (default: the network's drive)")
    ap.add_argument("--amp", type=float, default=None,
                    help="probe modulation amplitude in Hz (default: COARSE_GRID amp_hz)")
    ap.add_argument("--skip-cycle", action="store_true",
                    help="skip characterize_cycle (cycle stats only; the snapshot and "
                         "probes don't use it)")
    args = ap.parse_args()

    if args.condition == "reference":
        cond = cond_mod.make_condition(f"reference {cond_mod.STEADY_FREQ_HZ:g}Hz")
        freq = cond_mod.STEADY_FREQ_HZ
    else:
        cond = next(c for c in cond_mod.CONDITIONS if c["name"] == args.condition)
        freq = cond_mod.BURST_FREQ_HZ
    drug = cond["drug"]
    g = dict(cond_mod.COARSE_GRID)
    if args.amp is not None:
        g["amp_hz"] = args.amp
    g["probe_r0_hz"] = (args.r0 if args.r0 is not None
                        else g["probe_r0_hz"] if g["probe_r0_hz"] is not None
                        else freq)
    results = THIS_DIR / "results"
    slug = worker._slug(f"seed{args.seed}_{cond['name']}_probe{g['probe_r0_hz']:g}"
                        f"pm{g['amp_hz']:g}")
    out = results / "ipc_probe_inspection"
    out.mkdir(parents=True, exist_ok=True)
    work = results / "work" / f"inspect_{slug}"
    worker._isolate(work, results / "cython_cache" / f"inspect_{slug}",
                    template_cache_dir=results / "cython_cache" / "_template")

    from brian2 import second, Hz
    from seeded_network import build_network_seeded
    from HSP_Phase import characterize_cycle, _runkw
    from ipc_probe import ipc_probe
    from hsp_capacity import make_input_signal
    import coarse_phase as cp

    net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second,
                               input_freq=freq, network_seed=args.seed)
    worker.prepare_local_baseline(net, results / "baselines" / f"seed{args.seed}.bri",
                                  work, g["hsp_tau_s"], cond["weights"])
    print(f"HSP tau_ins/int = {g['hsp_tau_s']:g} s; probe {g['probe_r0_hz']:g} +- {g['amp_hz']:g} Hz; plateau "
          f"rule: k_AMPA range < {g['plateau_tol']:g} over {g['plateau_window_s']:g} s",
          flush=True)
    if not args.skip_cycle:
        print("characterizing cycle ...", flush=True)
        cyc = characterize_cycle(net, freq, drug=drug, **g["char_kw"])
        print(f"bursting: {cyc['is_bursting']}, period {cyc['period_s']:.0f} s", flush=True)
    snap = cp._plateau_snapshot(net, freq, drug, g["plateau_window_s"], g["plateau_tol"],
                                g["plateau_chunk_s"], g["plateau_max_s"], "snap")
    if not snap["found"]:
        raise SystemExit(f"no snapshot: {snap['status']}")
    P = net["P_E"]
    t_snap = float(net.t / second)
    t_rest = t_snap - snap["settle_s"]
    taus = {nm: float(np.asarray(getattr(P, nm)[:])) * second for nm in worker.HSP_TAUS}
    print(f"snapshot at {snap['settle_s']:.0f} s after restore: k_AMPA {snap['k_AMPA']:.3f}, "
          f"k_GABA {snap['k_GABA']:.3f}, k_AMPA range over the window {snap['k_range']:.4f}",
          flush=True)

    # ---- context (HSP ON) and probes (HSP FROZEN): probe_plots.py, the same
    # figures every pipeline job saves ------------------------------------------
    if g["inspect_extend_s"]:
        net.run(g["inspect_extend_s"] * second, **_runkw(net))
    d = probe_plots.context_data(net, t_rest, t_snap, g["plateau_window_s"], snap)
    d["meta"].update(plateau_tol=g["plateau_tol"], probe_r0_hz=g["probe_r0_hz"],
                     amp_hz=g["amp_hz"], step_s=g["step_s"], n_steps=g["n_steps"],
                     input_freq_hz=freq)
    runaway_seed = None
    for s in g["probe_seeds"]:
        u = make_input_signal(g["n_steps"], seed=s)
        r = ipc_probe(net, "snap", u, r0_hz=g["probe_r0_hz"], amp_hz=g["amp_hz"],
                      step_s=g["step_s"], max_delay=g["max_delay"], rng_seed=s)
        rates = np.clip(g["probe_r0_hz"] + g["amp_hz"] * u, 0, None)
        pE = probe_plots.add_probe(d, net, t_snap, g["n_steps"], g["step_s"], rates, s, r)
        ran_away, note = probe_plots.runaway_note(pE, g["step_s"])
        if ran_away and runaway_seed is None:
            runaway_seed = s
            net.store("runaway_end")
        print(f"probe {s}: IPC {r['ipc_total']:.3f} (encoding {r['ipc_encoding']:.3f}, "
              f"memory {r['ipc_memory']:.3f}), mean E {r['E_rate_probe']:.2f} Hz, {note}; "
              f"max |dk_AMPA| {d['probe_dkA'][-1]:.2e}, max |dk_GABA| {d['probe_dkG'][-1]:.2e}",
              flush=True)
    print(f"tau_ins_AMPA during the probes: {P.tau_ins_AMPA_E}", flush=True)
    npz = probe_plots.save_npz(d, out / f"{slug}.npz", label=f"seed {args.seed}, {cond['name']}",
                               hsp_tau_s=g["hsp_tau_s"])
    probe_plots.plot_npz(npz)

    # ---- runaway: is it self-sustained, and does HSP end it? -----------------
    if runaway_seed is not None:
        print(f"runaway test on probe {runaway_seed}: (a) input 0 Hz, HSP frozen, 10 s; "
              f"(b) HSP back on, input {freq:g} Hz, 300 s ...", flush=True)
        net.restore("runaway_end")
        t0 = float(net.t / second)
        net["inp"].rates = 0 * Hz
        net.run(10 * second, **_runkw(net))
        t1 = float(net.t / second)
        for nm, v in taus.items():
            setattr(P, nm, v)
        net["inp"].rates = freq * Hz
        net.run(300 * second, **_runkw(net))
        t2 = float(net.t / second)
        sE, sI = spikes(net)
        ta, eA = binned(sE, N_E, t0, t1, 0.05)
        tb2, eB = binned(sE, N_E, t1, t2, 1.0)
        _, iB = binned(sI, N_I, t1, t2, 1.0)
        tkb, kAb, kGb = k_traces(net, t1, t2)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(ta - t0, eA, color="tab:red", lw=0.8)
        ax[0].set_title("(a) input cut to 0 Hz, HSP still frozen", fontsize=10)
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("E rate (Hz, 50 ms)")
        ax[1].plot(tb2 - t1, eB, color="tab:red", lw=0.8, label="E")
        ax[1].plot(tb2 - t1, iB, color="tab:blue", lw=0.8, label="I")
        ax[1].set_yscale("symlog", linthresh=1)
        ax[1].set_ylim(bottom=0)
        ax[1].legend(fontsize=8)
        ax[1].set_title(f"(b) HSP switched back on, input {freq:g} Hz", fontsize=10)
        ax[1].set_xlabel("time (s)")
        ax[1].set_ylabel("rate (Hz, 1 s)")
        ax[2].plot(tkb - t1, kAb.mean(0), color="tab:orange", label="k_AMPA")
        ax[2].plot(tkb - t1, kGb.mean(0), color="tab:green", label="k_GABA")
        ax[2].legend(fontsize=8)
        ax[2].set_title("(b) HSP gains", fontsize=10)
        ax[2].set_xlabel("time (s)")
        fig.suptitle(f"What a runaway probe is -- seed {args.seed}, {cond['name']}, "
                     f"probe seed {runaway_seed}")
        fig.tight_layout()
        fig.savefig(out / f"{slug}_runaway.png", dpi=120)
        plt.close(fig)
        print(f"(a) E with input at 0 Hz, last 5 s: {eA[-100:].mean():.1f} Hz; "
              f"(b) E after HSP back on, last 30 s: {eB[-30:].mean():.2f} Hz", flush=True)
    else:
        print("no probe ran away", flush=True)
    print(f"wrote {out}/{slug}_*.png")


if __name__ == "__main__":
    main()
