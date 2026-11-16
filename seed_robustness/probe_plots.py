"""The IPC probe inspection figures, shared by every pipeline job
(coarse_phase.phase_capacity_seeded_coarse with inspect_extend_s set) and
inspect_ipc_probe.py, so both draw the same plots.

  <slug>_context.png -- HSP ON (normal simulation, NOT the IPC): E and I rate,
                        mean k_AMPA / k_GABA from restore to the snapshot and
                        some time beyond it. Snapshot = dashed line; the window
                        the plateau rule checked is shaded.
  <slug>_probes.png  -- HSP FROZEN (the IPC computation): per probe, input
                        rate, E and I rate (one bin per probe step), and
                        k_AMPA / k_GABA (mean and min-max over E cells).

Collection functions read the live network's monitors and return small binned
arrays; save_npz/load_npz keep them, so the figures can be redrawn without
rerunning (python probe_plots.py results/ipc_probe_inspection/runs/*.npz).
"""
import json
import sys
from pathlib import Path

import numpy as np

RUNAWAY_HZ = 10.0


def _binned(spike_t, n, t0, t1, bin_s):
    edges = t0 + np.arange(int(round((t1 - t0) / bin_s)) + 1) * bin_s
    c, _ = np.histogram(spike_t, bins=edges)
    return edges[:-1], c / (n * bin_s)


def _spikes(net):
    from brian2 import second
    sE, sI = net["sp_exc"], net["sp_inh"]
    return (np.asarray(sE.t / second), int(sE.source.N),
            np.asarray(sI.t / second), int(sI.source.N))


def _k(net, t0, t1):
    """(t, kA[neuron, t], kG[neuron, t]) from the 200 ms k monitors."""
    from brian2 import second
    m = net["k_AMPA_E_hsp"]
    t = np.asarray(m.t / second)
    sel = (t >= t0) & (t <= t1)
    return (t[sel], np.asarray(m.k_AMPA_E_hsp)[:, sel],
            np.asarray(net["k_GABA_E_hsp"].k_GABA_E_hsp)[:, sel])


def context_data(net, t_rest, t_snap, window_s, snap):
    """HSP-on run from restore (t_rest) to now; call right after the
    extension past the snapshot (t_snap)."""
    from brian2 import second
    t_end = float(net.t / second)
    sE, nE, sI, nI = _spikes(net)
    tb, rE = _binned(sE, nE, t_rest, t_end, 1.0)
    _, rI = _binned(sI, nI, t_rest, t_end, 1.0)
    tk, kA, kG = _k(net, t_rest, t_end)
    return dict(ctx_t=tb - t_rest, ctx_E=rE, ctx_I=rI, ctx_tk=tk - t_rest,
                ctx_kA=kA.mean(0), ctx_kG=kG.mean(0),
                meta=dict(snap_s=t_snap - t_rest, window_s=window_s,
                          settle_s=snap["settle_s"], k_AMPA=snap["k_AMPA"],
                          k_GABA=snap["k_GABA"], k_range=snap["k_range"],
                          probe_seeds=[], ipc=[], ipc_encoding=[], ipc_memory=[],
                          E_rate=[]))


def add_probe(d, net, t_snap, n_steps, step_s, rates, probe_seed, r):
    """Append one probe (just run by _ipc_probe from the snapshot at t_snap)
    to d, a dict from context_data()."""
    dur = n_steps * step_s
    sE, nE, sI, nI = _spikes(net)
    _, pE = _binned(sE, nE, t_snap, t_snap + dur, step_s)
    _, pI = _binned(sI, nI, t_snap, t_snap + dur, step_s)
    tk, kA, kG = _k(net, t_snap, t_snap + dur)
    row = dict(probe_input=np.asarray(rates, float), probe_E=pE, probe_I=pI,
               probe_tk=tk - t_snap,
               probe_kA_mean=kA.mean(0), probe_kA_min=kA.min(0), probe_kA_max=kA.max(0),
               probe_kG_mean=kG.mean(0), probe_kG_min=kG.min(0), probe_kG_max=kG.max(0),
               probe_dkA=np.abs(kA - kA[:, :1]).max(), probe_dkG=np.abs(kG - kG[:, :1]).max())
    for k, v in row.items():
        d.setdefault(k, []).append(v)
    d["meta"]["probe_seeds"].append(int(probe_seed))
    d["meta"]["ipc"].append(float(r["ipc_total"]))
    for k in ("ipc_encoding", "ipc_memory"):
        if k in r:
            d["meta"].setdefault(k, []).append(float(r[k]))
    d["meta"]["E_rate"].append(float(r["E_rate_probe"]))
    return pE


def save_npz(d, path, **meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = {**d["meta"], **meta}
    arrays = {k: np.asarray(v) for k, v in d.items() if k != "meta"}
    np.savez_compressed(path, meta=json.dumps(m), **arrays)
    return path


def load_npz(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def runaway_note(pE, step_s):
    high = np.flatnonzero(pE > RUNAWAY_HZ)
    if not high.size:
        return False, f"E never > {RUNAWAY_HZ:g} Hz (max {pE.max():.1f})"
    return True, (f"E > {RUNAWAY_HZ:g} Hz from {high[0] * step_s:.1f} s, then "
                  f"{100 * np.mean(pE[high[0]:] > RUNAWAY_HZ):.0f}% of steps")


def plot_context(d, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = d["meta"]
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(d["ctx_t"], d["ctx_E"], color="tab:red", lw=0.8)
    ax[0].set_ylabel("E rate (Hz, 1 s)")
    ax[1].plot(d["ctx_t"], d["ctx_I"], color="tab:blue", lw=0.8)
    ax[1].set_ylabel("I rate (Hz, 1 s)")
    for a in ax[:2]:
        a.set_yscale("symlog", linthresh=1)
        a.set_ylim(bottom=0)
    ax[2].plot(d["ctx_tk"], d["ctx_kA"], color="tab:orange")
    ax[2].set_ylabel("mean k_AMPA")
    ax[3].plot(d["ctx_tk"], d["ctx_kG"], color="tab:green")
    ax[3].set_ylabel("mean k_GABA")
    ax[3].set_xlabel("time since restore (s)  -- HSP ON throughout (not the IPC)")
    for a in ax:
        a.axvspan(m["snap_s"] - m["window_s"], m["snap_s"], color="tab:purple", alpha=0.12,
                  label="window checked" if a is ax[0] else None)
        a.axvline(m["snap_s"], color="k", ls="--", lw=1,
                  label=f"snapshot ({m['settle_s']:.0f} s)" if a is ax[0] else None)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title(f"{label}: normal run, HSP on (tau {m['hsp_tau_s']:g} s); snapshot once "
                    f"k_AMPA range < {m['plateau_tol']:g} over {m['window_s']:g} s",
                    fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_probes(d, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = d["meta"]
    seeds = m["probe_seeds"]
    step_s = m["step_s"]
    fig, ax = plt.subplots(5, len(seeds), figsize=(4.2 * len(seeds), 12), sharex=True,
                           sharey="row", squeeze=False)
    for j, s in enumerate(seeds):
        pE, pI = d["probe_E"][j], d["probe_I"][j]
        xp = np.arange(len(pE)) * step_s
        tk = d["probe_tk"][j]
        ax[0, j].step(xp, d["probe_input"][j][:len(xp)], where="post", color="0.3", lw=0.6)
        ax[1, j].plot(xp, pE, color="tab:red", lw=0.7)
        ax[2, j].plot(xp, pI, color="tab:blue", lw=0.7)
        for row, key, c in ((3, "kA", "tab:orange"), (4, "kG", "tab:green")):
            ax[row, j].fill_between(tk, d[f"probe_{key}_min"][j], d[f"probe_{key}_max"][j],
                                    color=c, alpha=0.25)
            ax[row, j].plot(tk, d[f"probe_{key}_mean"][j], color=c, lw=1.2)
        _, note = runaway_note(pE, step_s)
        split = (f" (encoding {m['ipc_encoding'][j]:.3f}, memory {m['ipc_memory'][j]:.3f})"
                 if len(m.get("ipc_encoding", [])) > j else "")
        ax[0, j].set_title(f"probe seed {s}: IPC {m['ipc'][j]:.3f}{split}\n{note}\n"
                           f"max |dk| over probe: AMPA {d['probe_dkA'][j]:.1e}, "
                           f"GABA {d['probe_dkG'][j]:.1e}", fontsize=9)
    ax[0, 0].set_ylabel("input rate (Hz)")
    ax[1, 0].set_ylabel(f"E rate (Hz, {step_s * 1000:g} ms)")
    ax[2, 0].set_ylabel(f"I rate (Hz, {step_s * 1000:g} ms)")
    ax[3, 0].set_ylabel("k_AMPA (mean, min-max)")
    ax[4, 0].set_ylabel("k_GABA (mean, min-max)")
    for a in ax[1:3, :].ravel():
        a.set_yscale("symlog", linthresh=1)
        a.set_ylim(bottom=0)
    for a in ax[-1, :]:
        a.set_xlabel("time in probe (s)")
    fig.suptitle(f"IPC probes, HSP FROZEN -- {label}, input {m['probe_r0_hz']:g} +- "
                 f"{m['amp_hz']:g} Hz (snapshot adapted at {m['input_freq_hz']:g} Hz), "
                 f"k_AMPA {m['k_AMPA']:.3f}, k_GABA {m['k_GABA']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_npz(npz_path):
    """Both figures next to the .npz, named <slug>_context.png / _probes.png."""
    npz_path = Path(npz_path)
    d = load_npz(npz_path)
    label = d["meta"]["label"]
    stem = npz_path.with_suffix("")
    return (plot_context(d, f"{stem}_context.png", label),
            plot_probes(d, f"{stem}_probes.png", label))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        for out in plot_npz(p):
            print(f"wrote {out}")
