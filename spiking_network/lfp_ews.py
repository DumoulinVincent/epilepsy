"""Early-warning signals in the LFP proxy before each seizure (lfp_cycle.py runs).

    python lfp_ews.py results/lfp_tau250_15k

For every untreated seizure (E rate > 10 Hz in 1 s bins) whose preceding
PRE_S seconds are seizure-free and after the settling period: WIN_S windows
over [onset - PRE_S, onset - GAP_S) (the onset transient itself is left out).
Per window: LFP mean, variance and CV of the 100 ms means, lag-1
autocorrelation of the 100 ms means and of the 10 ms means, power < 3 Hz and
at 3-30 Hz (Welch, 2 s segments), E rate.
Control: the same time segments of the same seed's 40 Hz reference (no
seizures), so a trend that also appears there is not a seizure precursor.
Trend: Kendall tau of each feature against time, per seizure / segment.

Outputs (<dir>/ews/): ews_windows.csv, ews_kendall.csv, and
plots/lfp_ews.png -- per feature: every seizure's trajectory (thin) and the
mean +- SEM of the change relative to the first 5 minutes, untreated vs
reference; Kendall tau per seizure on the right.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import signal, stats

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import lfp_tau_plot as lp  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRE_S, WIN_S, GAP_S, BASE_S = 600.0, 10.0, 2.0, 300.0
FEATS = (("mean", "LFP mean (pA)"), ("var", "LFP variance (100 ms means, pA²)"),
         ("cv", "LFP CV (100 ms means)"), ("ac1_100", "lag-1 autocorr (100 ms means)"),
         ("ac1_10", "lag-1 autocorr (10 ms means)"), ("p_lo", "power < 3 Hz (pA²)"),
         ("p_mid", "power 3-30 Hz (pA²)"), ("rE", "E rate (Hz)"))
COL = {"untreated": "#eb6834", "reference 40Hz": "#52514e"}


def ac1(x):
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[:-1], x[1:]) / d) if d > 0 else np.nan


def features(lfp, rE, fs, t0, t1):
    a, b = int(t0 * fs), int(t1 * fs)
    seg = lfp[a:b]
    m100 = lp.block_mean(seg, fs // 10)
    m10 = lp.block_mean(seg, fs // 100)
    f, P = signal.welch(seg, fs=fs, nperseg=2 * fs)
    df = f[1] - f[0]
    return dict(mean=seg.mean(), var=m100.var(), cv=m100.std() / m100.mean(), ac1_100=ac1(m100),
                ac1_10=ac1(m10), p_lo=P[(f > 0) & (f < 3)].sum() * df,
                p_mid=P[(f >= 3) & (f < 30)].sum() * df, rE=float(np.mean(rE[int(t0):int(t1)])))


def segments(d):
    """(seed, onset, untreated run, reference run) for every usable seizure."""
    out = []
    for p in sorted((Path(d) / "raw").glob("seed*_untreated.npz")):
        seed = int(p.stem.split("_")[0][4:])
        u = lp.load(p)
        rp = Path(d) / "raw" / f"seed{seed}_reference_40Hz.npz"
        ref = lp.load(rp) if rp.exists() else None
        prev_end = -np.inf
        for a, b in u["seizures"]:
            if a - PRE_S >= lp.SETTLE_S and prev_end <= a - PRE_S:
                out.append((seed, a, u, ref))
            prev_end = b
    return out


def main(d):
    d = Path(d)
    segs = segments(d)
    if not segs:
        raise SystemExit("no usable seizure (need PRE_S seizure-free seconds before onset)")
    edges = np.arange(-PRE_S, -GAP_S - WIN_S + 1e-9, WIN_S)
    rows = []
    for seed, onset, u, ref in segs:
        for cond, r in (("untreated", u), ("reference 40Hz", ref)):
            if r is None:
                continue
            fs = int(r["fs"])
            for e in edges:
                rows.append(dict(seed=seed, onset=onset, condition=cond, t_rel=e + WIN_S / 2,
                                 **features(r["lfp"], r["rE"], fs, onset + e, onset + e + WIN_S)))
    out = d / "ews"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    with open(out / "ews_windows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # Kendall tau per seizure / reference segment
    ktab = []
    for seed, onset, _, _ in segs:
        for cond in COL:
            R = [r for r in rows if r["seed"] == seed and r["onset"] == onset and r["condition"] == cond]
            if not R:
                continue
            t = np.array([r["t_rel"] for r in R])
            ktab.append(dict(seed=seed, onset=onset, condition=cond,
                             **{k: float(stats.kendalltau(t, [r[k] for r in R]).statistic) for k, _ in FEATS}))
    with open(out / "ews_kendall.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ktab[0]))
        w.writeheader()
        w.writerows(ktab)
    # figure
    fig, ax = plt.subplots(len(FEATS), 3, figsize=(17, 2.4 * len(FEATS) + 1),
                           gridspec_kw=dict(width_ratios=[2.2, 2.2, 1]))
    for i, (k, lab) in enumerate(FEATS):
        for cond, col in COL.items():
            curves = []
            for seed, onset, _, _ in segs:
                R = [r for r in rows if r["seed"] == seed and r["onset"] == onset and r["condition"] == cond]
                if not R:
                    continue
                t = np.array([r["t_rel"] for r in R])
                v = np.array([r[k] for r in R], float)
                if cond == "untreated":
                    ax[i, 0].plot(t, v, color=col, lw=0.7, alpha=0.6)
                base = np.nanmean(v[t < -PRE_S + BASE_S])
                curves.append(v / base if k not in ("ac1_100", "ac1_10") else v - base)
            if curves:
                C = np.array(curves)
                m, se = np.nanmean(C, 0), np.nanstd(C, 0) / np.sqrt(np.sum(np.isfinite(C), 0))
                ax[i, 1].fill_between(t, m - se, m + se, color=col, alpha=0.2, lw=0)
                ax[i, 1].plot(t, m, color=col, lw=1.6, label=f"{cond} (n={len(C)})")
            kt = [r[k] for r in ktab if r["condition"] == cond]
            x = 0 if cond == "untreated" else 1
            ax[i, 2].plot(np.full(len(kt), x) + np.random.default_rng(0).uniform(-0.1, 0.1, len(kt)), kt, "o",
                          color=col, ms=5, alpha=0.8)
            if kt:
                ax[i, 2].plot([x - 0.2, x + 0.2], [np.mean(kt)] * 2, color=col, lw=2)
        ax[i, 0].set_ylabel(lab, fontsize=9)
        rel = "difference from" if k in ("ac1_100", "ac1_10") else "ratio to"
        ax[i, 1].set_ylabel(f"{rel} first 5 min", fontsize=9)
        ax[i, 1].axhline(0 if k in ("ac1_100", "ac1_10") else 1, color="#9a9890", lw=0.8)
        ax[i, 2].axhline(0, color="#9a9890", lw=0.8)
        ax[i, 2].set_xticks([0, 1])
        ax[i, 2].set_xticklabels(["untreated", "reference"], fontsize=8)
        ax[i, 2].set_ylim(-1, 1)
        for a in ax[i]:
            lp.style(a)
    ax[0, 0].set_title("untreated: each seizure (raw)", fontsize=10, loc="left")
    ax[0, 1].set_title("mean ± SEM over seizures, vs same segments of the reference", fontsize=10, loc="left")
    ax[0, 2].set_title("Kendall tau vs time", fontsize=10, loc="left")
    ax[0, 1].legend(fontsize=8, frameon=False)
    for a in ax[-1, :2]:
        a.set_xlabel("time to seizure onset (s)")
    fig.suptitle(f"LFP early-warning signals: {WIN_S:g} s windows over the last {PRE_S:g} s before onset "
                 f"({len(segs)} seizures, {len({s[0] for s in segs})} seeds); onset transient excluded "
                 f"(last {GAP_S:g} s)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out / "plots" / "lfp_ews.png", dpi=110)
    plt.close(fig)
    print(f"{len(segs)} usable seizures: {[(s, round(o)) for s, o, _, _ in segs]}")
    for cond in COL:
        K = [r for r in ktab if r["condition"] == cond]
        if K:
            print(f"{cond:15s} mean Kendall tau over {len(K)} segments: " +
                  ", ".join(f"{k} {np.mean([r[k] for r in K]):+.2f}" for k, _ in FEATS))
    print(f"wrote {out}/ews_windows.csv, ews_kendall.csv, plots/lfp_ews.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else THIS_DIR / "results" / "lfp_tau250_15k")
