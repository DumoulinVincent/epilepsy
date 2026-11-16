"""LFP proxy, untreated 36 Hz vs the 40 Hz reference (lfp_cycle.py runs).

    python lfp_tau_plot.py results/lfp_tau500 [--seed 1]

  <dir>/plots/lfp_timeseries.png -- LFP proxy (|I_exc| + |I_inh|, pA; 100 ms
      means), both conditions on the same y axis, limited to the plateau
      (non-seizure) range so details stay visible; seizure peaks are clipped
      and labelled. Below: E rate, k_AMPA, k_GABA. Seizures shaded.
  <dir>/plots/lfp_features.png -- LFP mean, SD, CV, power < 3 Hz and 3-30 Hz,
      E rate in 50 s windows over the whole run (seizure windows left out,
      onsets marked): how the LFP evolves on the way to a seizure.
  <dir>/plots/lfp_psd.png -- Welch PSD (4 s windows, 50% overlap) of both
      conditions on top of each other: (a) non-seizure time only (after the
      settling period, seizures +- 60 s removed), (b) the same normalised to
      total 1-150 Hz power (shape only), (c) the whole run, seizures included.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEIZURE_HZ, PAD_S, SETTLE_S = 10.0, 60.0, 500.0
COL = {"untreated": "#eb6834", "reference 40Hz": "#52514e"}
INK, MUTED, GRID, SZ = "#0b0b0b", "#52514e", "#e4e3de", "#d03b3b"


def load(path):
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    lfp = z["lfp_exc"].astype(float) + z["lfp_inh"].astype(float)
    above = np.r_[False, z["rate_E"] > SEIZURE_HZ, False]
    d = np.diff(above.astype(int))
    t1 = z["rate_t"] - 0.5
    sz = [(float(t1[a]), float(t1[b - 1] + 1)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return dict(meta=m, fs=m["fs"], lfp=lfp, rate_t=z["rate_t"], rE=z["rate_E"], rI=z["rate_I"],
                t_k=z["t_k"], kA=z["k_AMPA"], kG=z["k_GABA"], seizures=sz)


def block_mean(x, n):
    k = len(x) // n
    return x[:k * n].reshape(k, n).mean(1)


def valid_mask(r):
    t = np.arange(len(r["lfp"])) / r["fs"]
    ok = t >= SETTLE_S
    for a, b in r["seizures"]:
        ok &= ~((t >= a - PAD_S) & (t < b + PAD_S))
    return ok


def welch_masked(x, mask, fs, nper):
    """Welch PSD averaged over all windows lying fully inside `mask`."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    acc, n_win, f = None, 0, None
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < nper:
            continue
        f, P = signal.welch(x[a:b], fs=fs, nperseg=nper, noverlap=nper // 2)
        w = 1 + (b - a - nper) // (nper // 2)
        acc = P * w if acc is None else acc + P * w
        n_win += w
    return f, (acc / n_win if n_win else None), n_win


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def window_features(r, win_s=50.0):
    """Per non-overlapping win_s window: LFP mean, SD and CV of the 100 ms
    means, power below 3 Hz and at 3-30 Hz (Welch, 4 s), E rate."""
    fs = int(r["fs"])
    x = r["lfp"]
    n = int(win_s * fs)
    rows = []
    for a in range(0, len(x) - n + 1, n):
        seg = x[a:a + n]
        m100 = block_mean(seg, fs // 10)
        f, P = signal.welch(seg, fs=fs, nperseg=4 * fs)
        df = f[1] - f[0]
        t0 = a / fs
        rows.append(dict(t=t0 + win_s / 2, mean=seg.mean(), sd=m100.std(), cv=m100.std() / m100.mean(),
                         p_lo=P[(f > 0) & (f < 3)].sum() * df, p_mid=P[(f >= 3) & (f < 30)].sum() * df,
                         rE=r["rE"][int(t0):int(t0 + win_s)].mean(),
                         ictal=any(t0 < b and t0 + win_s > a_ for a_, b in r["seizures"])))
    return rows


def plot_features(runs, path, tau):
    keys = (("mean", "LFP mean (pA)"), ("sd", "LFP SD of 100 ms means (pA)"), ("cv", "LFP CV"),
            ("p_lo", "LFP power < 3 Hz (pA²)"), ("p_mid", "LFP power 3-30 Hz (pA²)"), ("rE", "E rate (Hz)"))
    fig, ax = plt.subplots(len(keys), 1, figsize=(15, 2.2 * len(keys) + 1), sharex=True)
    for c, r in runs.items():
        F = window_features(r)
        t = np.array([w["t"] for w in F])
        quiet = ~np.array([w["ictal"] for w in F])
        for a, (k, lab) in zip(ax, keys):
            v = np.array([w[k] for w in F], float)
            a.plot(t[quiet], v[quiet], ".-", color=COL[c], ms=2, lw=0.8, label=c)
            a.set_ylabel(lab, fontsize=9)
    for a in ax:
        style(a)
        a.axvspan(0, SETTLE_S, color=GRID, alpha=0.5, lw=0)
        for c, r in runs.items():
            for s0, s1 in r["seizures"]:
                a.axvspan(s0, s1, color=SZ, alpha=0.3, lw=0)
                a.axvline(s0, color=SZ, lw=0.8)
    ax[0].legend(fontsize=8, frameon=False)
    ax[-1].set_xlabel("time (s)")
    ax[0].set_title(f"LFP features in 50 s windows (windows containing a seizure left out), HSP tau {tau:g} s; "
                    f"seizure onsets red", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(d, seed=1, suffix=""):
    d = Path(d)
    runs = {}
    for cond, slug in (("untreated", "untreated"), ("reference 40Hz", "reference_40Hz")):
        p = d / "raw" / f"seed{seed}_{slug}.npz"
        if p.exists():
            runs[cond] = load(p)
    if not runs:
        raise SystemExit(f"no runs in {d / 'raw'}")
    (d / "plots").mkdir(exist_ok=True)
    tau = next(iter(runs.values()))["meta"].get("hsp_tau_s")
    # ---------------------------------------------------------------- time series
    nb = int(next(iter(runs.values()))["fs"] * 0.1)
    means = {c: block_mean(r["lfp"], nb) for c, r in runs.items()}
    quiet = [means[c][valid_mask(r)[: len(means[c]) * nb].reshape(-1, nb).all(1)] for c, r in runs.items()]
    ylim = 1.3 * max(np.percentile(q, 99.5) for q in quiet if q.size)
    rlim = 1.3 * max(np.percentile(r["rE"][valid_mask(r)[:: int(r["fs"])][: len(r["rE"])]], 99.5) for r in runs.values())
    fig, ax = plt.subplots(len(runs) + 3, 1, figsize=(15, 3 + 2.3 * (len(runs) + 3)), sharex=True)
    for i, (c, r) in enumerate(runs.items()):
        t = (np.arange(len(means[c])) + 0.5) * 0.1
        ax[i].plot(t, means[c], color=COL[c], lw=0.4)
        ax[i].set_ylim(0, ylim)
        ax[i].set_ylabel(f"LFP proxy (pA)\n{c}, 100 ms means")
        for a, b in r["seizures"]:
            pk = means[c][int(a * 10):int(b * 10) + 10].max()
            ax[i].annotate(f"{pk:.0f} pA", (a, ylim), xytext=(3, -12), textcoords="offset points",
                           fontsize=7, color=SZ)
    k = len(runs)
    for c, r in runs.items():
        ax[k].plot(r["rate_t"], r["rE"], color=COL[c], lw=0.5, label=c)
        ax[k + 1].plot(r["t_k"], r["kA"], color=COL[c], lw=1.2, label=c)
        ax[k + 2].plot(r["t_k"], r["kG"], color=COL[c], lw=1.2, label=c)
    ax[k].set_ylim(0, rlim)
    ax[k].set_ylabel("E rate (Hz, 1 s)\n(clipped at plateau range)")
    ax[k + 1].set_ylabel("mean k_AMPA")
    ax[k + 2].set_ylabel("mean k_GABA")
    ax[k].legend(fontsize=8, frameon=False, loc="upper right")
    ax[-1].set_xlabel("time (s)")
    for a in ax:
        style(a)
        a.axvspan(0, SETTLE_S, color=GRID, alpha=0.5, lw=0)
        for c, r in runs.items():
            for s0, s1 in r["seizures"]:
                a.axvspan(s0, s1, color=SZ, alpha=0.3, lw=0)
    ax[0].set_title(f"LFP proxy |I_exc| + |I_inh| onto E cells, seed {seed}, HSP tau {tau:g} s: y axes limited to the "
                    f"plateau range (seizure peaks clipped, value labelled); seizures shaded red", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(d / "plots" / f"lfp_timeseries{suffix}.png", dpi=110)
    plt.close(fig)
    # ---------------------------------------------------------------------- PSD
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for c, r in runs.items():
        fs = r["fs"]
        nper = int(4 * fs)
        f, P, n = welch_masked(r["lfp"], valid_mask(r), fs, nper)
        sel = (f >= 0.25) & (f <= 200)
        if P is not None:
            ax[0].loglog(f[sel], P[sel], color=COL[c], lw=1.2, label=f"{c} ({n} windows)")
            band = (f >= 1) & (f <= 150)
            ax[1].loglog(f[sel], P[sel] / P[band].sum(), color=COL[c], lw=1.2, label=c)
        full = np.arange(len(r["lfp"])) / fs >= SETTLE_S
        f2, P2, n2 = welch_masked(r["lfp"], full, fs, nper)
        ax[2].loglog(f2[sel], P2[sel], color=COL[c], lw=1.2,
                     label=f"{c} ({len(r['seizures'])} seizures included)")
    ax[0].set_title("(a) non-seizure time (seizures +- 60 s removed)", fontsize=10, loc="left")
    ax[1].set_title("(b) same, normalised to 1-150 Hz power (shape)", fontsize=10, loc="left")
    ax[2].set_title("(c) whole run after settling, seizures included", fontsize=10, loc="left")
    ax[0].set_ylabel("PSD (pA²/Hz)")
    ax[1].set_ylabel("relative PSD (1/Hz)")
    for a in ax:
        a.set_xlabel("frequency (Hz)")
        a.legend(fontsize=8, frameon=False)
        style(a)
    fig.suptitle(f"LFP proxy power spectra, seed {seed}, HSP tau {tau:g} s (Welch, 4 s windows, 1 ms sampling)",
                 color=INK)
    fig.tight_layout()
    fig.savefig(d / "plots" / f"lfp_psd{suffix}.png", dpi=120)
    plt.close(fig)
    for c, r in runs.items():
        q = valid_mask(r)
        print(f"{c:15s}: {len(r['seizures'])} seizures {[(round(a), round(b - a)) for a, b in r['seizures']]}, "
              f"LFP non-seizure mean {r['lfp'][q].mean():.2f} pA, E {r['rE'][q[:: int(r['fs'])][: len(r['rE'])]].mean():.2f} Hz")
    plot_features(runs, d / "plots" / f"lfp_features{suffix}.png", tau)
    print(f"wrote {d / 'plots'}/lfp_timeseries{suffix}.png, lfp_psd{suffix}.png and lfp_features{suffix}.png")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = Path(args[0] if args else Path(__file__).resolve().parent / "results" / "lfp_tau500")
    if "--seed" in sys.argv:
        main(d, int(sys.argv[sys.argv.index("--seed") + 1]))
    else:   # every seed found; one set of figures per seed when there are several
        seeds = sorted({int(p.stem.split("_")[0][4:]) for p in (d / "raw").glob("seed*_*.npz")})
        for sd in seeds:
            print(f"--- seed {sd}")
            main(d, sd, suffix=f"_seed{sd}" if len(seeds) > 1 else "")
