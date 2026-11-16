"""LFP right before a seizure vs the plateau (reads results/lfp_cycle/raw/*.npz).

    python lfp_cycle_analysis.py [results_dir]

For every seizure (E rate > SEIZURE_HZ in 1 s bins), windows of WIN_S every
STEP_S between the previous seizure (+ POST_S) and onset are labelled:
  pre-ictal -- the window ends within the last PRE_S s before onset (and at
               least GAP_S before it, so the onset itself is excluded)
  plateau   -- the window ends at least FAR_S before onset AND its mean
               k_AMPA is within K_TOL of k_AMPA just before this onset: the
               same high-risk state, but not about to seize
Per window: mean, SD, CV, lag-1 autocorrelation (10 ms bins), spectral
entropy (1-100 Hz), permutation entropy (order PE_ORDER on PE_BIN_MS bins),
relative band powers, 1/f slope, and the log PSD.

Outputs (results/lfp_cycle/):
  windows.csv                    -- every labelled window
  separability.csv               -- per condition and feature: AUC (pre-ictal
                                    vs plateau windows), paired per-seizure
                                    difference and Wilcoxon p
  plots/psd_preictal_vs_plateau.png
  plots/features_vs_time.png     -- features vs time to onset
  plots/separability.png         -- AUC per feature and condition
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import signal, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import conditions as cond_mod  # noqa: E402

SEIZURE_HZ = 10.0
SETTLE_S = 500.0
WIN_S, STEP_S = 10.0, 5.0
PRE_S, GAP_S = 60.0, 2.0
FAR_S, K_TOL = 180.0, 0.03
POST_S = 60.0
TRACK_S = 900.0                     # features vs time: this long before onset
PE_ORDER, PE_BIN_MS = 5, 4.0
BANDS = dict(delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 80),
             high_gamma=(80, 150))
FEATURES = [("mean", "mean (pA)"), ("sd", "SD (pA)"), ("cv", "CV"),
            ("ac1", "lag-1 autocorr (10 ms)"), ("spec_entropy", "spectral entropy"),
            ("perm_entropy", f"permutation entropy (m={PE_ORDER})"), ("slope", "1/f slope (4-70 Hz)")] + \
           [(f"rel_{k}", f"rel. {k} power") for k in BANDS]
PRE_C, PLAT_C = "#d03b3b", "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"


def find_seizures(rate_t, rate_E):
    above = np.r_[False, rate_E > SEIZURE_HZ, False]
    d = np.diff(above.astype(int))
    t = rate_t - 0.5
    return [(float(t[a]), float(t[b - 1] + 1.0))
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def perm_entropy(x, m=PE_ORDER):
    n = len(x) - m + 1
    if n <= 1:
        return np.nan
    emb = np.lib.stride_tricks.sliding_window_view(x, m)
    ranks = np.argsort(emb, axis=1)
    codes = (ranks * (m ** np.arange(m))).sum(1)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def window_features(w, fs):
    out = dict(mean=float(w.mean()), sd=float(w.std()))
    out["cv"] = out["sd"] / out["mean"] if out["mean"] > 0 else np.nan
    b = max(1, int(round(0.010 * fs)))
    c = w[: len(w) // b * b].reshape(-1, b).mean(1)
    c = c - c.mean()
    out["ac1"] = float(np.dot(c[:-1], c[1:]) / np.dot(c, c)) if np.dot(c, c) > 0 else np.nan
    f, P = signal.welch(signal.detrend(w), fs=fs, nperseg=int(2 * fs))
    sel = (f >= 1) & (f <= 100) & (P > 0)
    p = P[sel] / P[sel].sum()
    out["spec_entropy"] = float(-(p * np.log(p)).sum() / np.log(sel.sum()))
    bp = max(1, int(round(PE_BIN_MS / 1000 * fs)))
    out["perm_entropy"] = perm_entropy(w[: len(w) // bp * bp].reshape(-1, bp).mean(1))
    tot = P[(f >= 1) & (f < 150)].sum()
    for k, (lo, hi) in BANDS.items():
        out[f"rel_{k}"] = float(P[(f >= lo) & (f < hi)].sum() / tot)
    fit = (f >= 4) & (f <= 70) & (P > 0)
    out["slope"] = float(np.polyfit(np.log10(f[fit]), np.log10(P[fit]), 1)[0])
    return out, f, P


def analyse_run(path):
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    fs = m["fs"]
    x = z["lfp_exc"].astype(float) + z["lfp_inh"].astype(float)
    sz = find_seizures(z["rate_t"], z["rate_E"])
    t_k, kA = z["t_k"], z["k_AMPA"]
    nw, ns = int(WIN_S * fs), int(STEP_S * fs)
    rows, psds, f_out = [], [], None
    prev = SETTLE_S - POST_S
    for ci, (on, off) in enumerate(sz):
        lo = prev + POST_S
        prev = off
        if on - lo < TRACK_S / 3:
            continue
        sel = (t_k >= on - PRE_S) & (t_k < on - GAP_S)
        k_pre = float(kA[sel].mean())
        start = max(lo, on - TRACK_S)
        for a in np.arange(start, on - GAP_S - WIN_S + 1e-9, STEP_S):
            end = a + WIN_S
            i0 = int(round(a * fs))
            w = x[i0:i0 + nw]
            if len(w) < nw:
                continue
            ks = (t_k >= a) & (t_k < end)
            k_w = float(kA[ks].mean()) if ks.any() else np.nan
            if end >= on - PRE_S:
                cls = "preictal"
            elif end <= on - FAR_S and abs(k_w - k_pre) <= K_TOL:
                cls = "plateau"
            else:
                cls = "other"
            feats, f, P = window_features(w, fs)
            f_out = f
            rows.append(dict(seed=m["seed"], condition=m["condition"], seizure=ci,
                             onset=on, t_to_onset=end - on, cls=cls, k_AMPA=k_w,
                             E_rate=float(z["rate_E"][(z["rate_t"] >= a) & (z["rate_t"] < end)].mean()),
                             **feats))
            psds.append(np.log10(np.maximum(P, 1e-30)))
    return rows, (f_out, np.array(psds)), sz, m


# ------------------------------------------------------------------ stats
def auc(a, b):
    """P(feature in a > feature in b) -- 0.5 = no separation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        return np.nan
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (a.size * b.size))


def separability(rows, conds):
    out = []
    for cnd in conds:
        R = [r for r in rows if r["condition"] == cnd]
        for k, _ in FEATURES:
            pre = [r[k] for r in R if r["cls"] == "preictal"]
            pla = [r[k] for r in R if r["cls"] == "plateau"]
            diffs = []
            for key in sorted({(r["seed"], r["seizure"]) for r in R}):
                a = [r[k] for r in R if (r["seed"], r["seizure"]) == key and r["cls"] == "preictal"]
                b = [r[k] for r in R if (r["seed"], r["seizure"]) == key and r["cls"] == "plateau"]
                if a and b:
                    diffs.append(np.mean(a) - np.mean(b))
            diffs = np.array(diffs)
            p = (float(stats.wilcoxon(diffs).pvalue) if diffs.size >= 6 and np.any(diffs != 0)
                 else np.nan)
            out.append(dict(condition=cnd, feature=k, auc=auc(pre, pla), n_pre=len(pre),
                            n_plateau=len(pla), n_seizures=int(diffs.size),
                            paired_mean_diff=float(diffs.mean()) if diffs.size else np.nan,
                            plateau_mean=float(np.mean(pla)) if pla else np.nan, wilcoxon_p=p))
    return out


# ------------------------------------------------------------------ plots
def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_psd(rows, psd_by_run, conds, path):
    fig, ax = plt.subplots(2, len(conds), figsize=(5.5 * len(conds), 8), sharex=True,
                           squeeze=False, gridspec_kw=dict(height_ratios=[2, 1]))
    for j, cnd in enumerate(conds):
        per = {"preictal": [], "plateau": []}
        f = None
        for (seed, c), (fr, P, R) in psd_by_run.items():
            if c != cnd or fr is None:
                continue
            f = fr
            for ci in sorted({r["seizure"] for r in R}):
                for cls in per:
                    idx = [i for i, r in enumerate(R) if r["seizure"] == ci and r["cls"] == cls]
                    if idx:
                        per[cls].append(P[idx].mean(0))
        if f is None or not per["preictal"] or not per["plateau"]:
            ax[0, j].set_title(f"{cnd}: not enough seizures", fontsize=10)
            continue
        band = (f >= 1) & (f <= 150)
        for cls, c, lab in (("plateau", PLAT_C, "plateau"), ("preictal", PRE_C, f"last {PRE_S:g} s")):
            A = np.array(per[cls])[:, band]
            mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
            ax[0, j].fill_between(f[band], mu - se, mu + se, color=c, alpha=0.2, lw=0)
            ax[0, j].plot(f[band], mu, color=c, lw=1.4, label=f"{lab} (n={len(A)} seizures)")
        # paired difference per seizure
        n = min(len(per["preictal"]), len(per["plateau"]))
        D = np.array(per["preictal"][:n])[:, band] - np.array(per["plateau"][:n])[:, band]
        mu, se = D.mean(0), D.std(0, ddof=1) / np.sqrt(n)
        ax[1, j].fill_between(f[band], mu - 1.96 * se, mu + 1.96 * se, color=MUTED, alpha=0.25, lw=0)
        ax[1, j].plot(f[band], mu, color=INK, lw=1.2)
        ax[1, j].axhline(0, color=MUTED, lw=0.8)
        ax[0, j].set_xscale("log")
        ax[0, j].set_title(cnd, fontsize=11, color=INK, loc="left")
        ax[0, j].legend(fontsize=8, frameon=False)
        ax[1, j].set_xlabel("frequency (Hz)")
        for a in ax[:, j]:
            _style(a)
    ax[0, 0].set_ylabel("log10 PSD (pA²/Hz)\nmean ± SEM over seizures")
    ax[1, 0].set_ylabel("pre-ictal − plateau\n(log10, mean ± 95% CI)")
    fig.suptitle(f"LFP proxy PSD: last {PRE_S:g} s before onset vs plateau\n(same k_AMPA, "
                 f"> {FAR_S:g} s before onset); {WIN_S:g} s windows", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_vs_time(rows, conds, path, bin_s=30.0):
    keys = FEATURES
    ncol = 3
    nrow = math.ceil(len(keys) / ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(16, 3.0 * nrow), sharex=True)
    ax = ax.ravel()
    colors = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100"]
    edges = np.arange(-TRACK_S, 1e-9, bin_s)
    for a, (k, lab) in zip(ax, keys):
        for cnd, c in zip(conds, colors):
            R = [r for r in rows if r["condition"] == cnd]
            if not R:
                continue
            # per seizure binned, then mean +- SEM over seizures
            curves = []
            for key in sorted({(r["seed"], r["seizure"]) for r in R}):
                S = [r for r in R if (r["seed"], r["seizure"]) == key]
                tt = np.array([r["t_to_onset"] for r in S])
                vv = np.array([r[k] for r in S])
                cur = [vv[(tt >= lo) & (tt < hi)].mean() if ((tt >= lo) & (tt < hi)).any() else np.nan
                       for lo, hi in zip(edges[:-1], edges[1:])]
                curves.append(cur)
            C = np.array(curves)
            with np.errstate(all="ignore"):
                mu = np.nanmean(C, 0)
                se = np.nanstd(C, 0) / np.sqrt(np.sum(np.isfinite(C), 0))
            mid = (edges[:-1] + edges[1:]) / 2
            a.fill_between(mid, mu - se, mu + se, color=c, alpha=0.18, lw=0)
            a.plot(mid, mu, color=c, lw=1.4, label=f"{cnd} ({len(C)} seizures)")
        a.axvspan(-PRE_S, 0, color=PRE_C, alpha=0.08, lw=0)
        a.set_ylabel(lab, fontsize=9)
        _style(a)
    for a in ax[len(keys):]:
        a.set_visible(False)
    ax[0].legend(fontsize=8, frameon=False)
    for a in ax[-ncol:]:
        a.set_xlabel("time to seizure onset (s)")
    fig.suptitle(f"LFP proxy features before seizure onset ({WIN_S:g} s windows, {bin_s:g} s bins, "
                 "mean ± SEM over seizures; red band = pre-ictal windows)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_separability(sep, conds, path):
    feats = [k for k, _ in FEATURES]
    labels = dict(FEATURES)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100"]
    w = 0.8 / max(1, len(conds))
    for j, (cnd, c) in enumerate(zip(conds, colors)):
        v = [next((s["auc"] for s in sep if s["condition"] == cnd and s["feature"] == k), np.nan)
             for k in feats]
        ax.bar(np.arange(len(feats)) + (j - (len(conds) - 1) / 2) * w, v, w * 0.9, color=c,
               label=cnd)
    ax.axhline(0.5, color=INK, lw=1)
    ax.text(len(feats) - 0.5, 0.505, "0.5 = no separation", ha="right", va="bottom", fontsize=8,
            color=MUTED)
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels([labels[k] for k in feats], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC, pre-ictal vs plateau windows\n(> 0.5: higher before onset)")
    ax.legend(fontsize=9, frameon=False)
    _style(ax)
    fig.suptitle(f"Can one LFP feature tell the last {PRE_S:g} s before a seizure from the plateau?",
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(out_dir=None):
    out_dir = Path(out_dir) if out_dir else THIS_DIR / "results" / "lfp_cycle"
    raw = sorted((out_dir / "raw").glob("*.npz"))
    if not raw:
        raise SystemExit(f"no runs in {out_dir / 'raw'}")
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    rows, psd_by_run = [], {}
    for p in raw:
        r, (f, P), sz, m = analyse_run(p)
        rows += r
        psd_by_run[(m["seed"], m["condition"])] = (f, P, r)
        n = {c: sum(x["cls"] == c for x in r) for c in ("preictal", "plateau")}
        print(f"{p.stem:24s} {len(sz):2d} seizures; windows pre-ictal {n['preictal']:4d}, "
              f"plateau {n['plateau']:4d}", flush=True)
    order = [c["name"] for c in cond_mod.LFP_CONDITIONS if any(r["condition"] == c["name"] for r in rows)]
    order += sorted({r["condition"] for r in rows} - set(order))
    if not rows:
        raise SystemExit("no seizure with enough history to analyse")
    with open(out_dir / "windows.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    sep = separability(rows, order)
    with open(out_dir / "separability.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sep[0]))
        wr.writeheader()
        wr.writerows(sep)
    print(f"\n{'condition':10s} {'feature':18s} {'AUC':>5s} {'paired diff':>12s} {'p':>8s} (n seizures)")
    for s in sep:
        print(f"{s['condition']:10s} {s['feature']:18s} {s['auc']:5.2f} {s['paired_mean_diff']:12.4g} "
              f"{s['wilcoxon_p']:8.2g} ({s['n_seizures']})")
    plot_psd(rows, psd_by_run, order, out_dir / "plots" / "psd_preictal_vs_plateau.png")
    plot_vs_time(rows, order, out_dir / "plots" / "features_vs_time.png")
    plot_separability(sep, order, out_dir / "plots" / "separability.png")
    print(f"wrote {out_dir}/windows.csv, separability.csv, plots/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
