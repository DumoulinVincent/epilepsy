"""Entropy of the LFP proxy as a seizure-risk marker (lfp_cycle.py runs).

    python lfp_entropy.py

Runs: results/lfp_tau250_15k/raw (all conditions, 15000 s) and, for benzo and
DBS 20 pA, results/lfp_tau250_20k/raw (20000 s), as in lfp_treatment.py.

Features (same windows as the power analysis, non-overlapping 50 s, windows
containing a seizure left out):
  SpecEn  -- spectral entropy: Shannon entropy of the normalised Welch PSD
             (4 s segments) over 0.25-100 Hz, divided by log(number of bins)
             (1 = flat spectrum, 0 = a single frequency)
  PermEn  -- permutation entropy (Bandt & Pompe 2002) of order PE_M with delay
             PE_TAU_MS, divided by log(PE_M!) (1 = all ordinal patterns equally
             likely). The proxy is a smooth synaptic current, so the delay is
             10 ms rather than one sample.
  p30     -- power 0-30 Hz, for comparison (as in lfp_treatment.py)

Tests:
  1. time course and seizure-free level per condition (dose relation);
  2. early-warning trend: Kendall tau against time in 10 s windows over the
     last 600 s before each untreated onset, vs the same segments of the
     seed's reference (as lfp_ews.py);
  3. pre-ictal discrimination: AUROC of each feature for the last 10 min before
     onset vs mid-interictal windows (> 30 min from any seizure), untreated,
     pooled over seeds; the sign is chosen per feature on the data, so 0.5 is
     chance and 1 is perfect separation.

Outputs: results/lfp_tau250_15k/entropy/{entropy_windows.csv, entropy_ews.csv,
entropy_summary.csv, plots/*.png}
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import signal, stats

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import lfp_tau_plot as lp  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R15 = THIS_DIR / "results" / "lfp_tau250_15k"
R20 = THIS_DIR / "results" / "lfp_tau250_20k"
OUT = R15 / "entropy"
LONG = ("benzo__k_1_0", "DBS_20pA")
WIN_S, EWS_WIN_S, PRE_S, GAP_S = 50.0, 10.0, 600.0, 2.0
PE_M, PE_TAU_MS = 5, 10
SE_BAND = (0.25, 100.0)
PREICTAL_S, FAR_S = 600.0, 1800.0
UNT, REF = "untreated", "reference 40Hz"
DBS_LEVELS = (20, 60, 100, 150, 200, 250, 300)
ORDER = [UNT, "benzo (k=1.0)"] + [f"DBS {a}pA" for a in DBS_LEVELS] + [REF]
LABEL = {UNT: "untreated", "benzo (k=1.0)": "benzo", REF: "reference", **{f"DBS {a}pA": f"DBS {a}" for a in DBS_LEVELS}}
_RAMP = ("#9fdcb5", "#5cc392", "#1baf7a", "#11905f", "#008300", "#006400", "#004d1a")
COL = {UNT: "#eb6834", REF: "#52514e", "benzo (k=1.0)": "#2a78d6", **{f"DBS {a}pA": c for a, c in zip(DBS_LEVELS, _RAMP)}}
FEATS = (("SpecEn", "spectral entropy"), ("PermEn", "permutation entropy"), ("p30", "power 0-30 Hz (pA²)"))


def spectral_entropy(x, fs):
    f, P = signal.welch(x, fs=fs, nperseg=int(4 * fs))
    m = (f >= SE_BAND[0]) & (f <= SE_BAND[1])
    p = P[m] / P[m].sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(m.sum())), float(P[(f > 0) & (f < 30)].sum() * (f[1] - f[0]))


def permutation_entropy(x, fs, m=PE_M, tau_ms=PE_TAU_MS):
    tau = int(round(tau_ms * fs / 1000))
    n = len(x) - (m - 1) * tau
    emb = np.stack([x[i * tau:i * tau + n] for i in range(m)], axis=1)
    ranks = np.argsort(emb, axis=1, kind="stable")
    codes = (ranks * (m ** np.arange(m))).sum(1)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def features(x, fs):
    se, p30 = spectral_entropy(x, fs)
    return dict(SpecEn=se, PermEn=permutation_entropy(x, fs), p30=p30)


def run_file(path):
    r = lp.load(path)
    fs = int(r["fs"])
    cond = r["meta"]["condition"]
    seed = int(Path(path).stem.split("_")[0][4:])
    x = r["lfp"]
    n = int(WIN_S * fs)
    rows = []
    for a in range(0, len(x) - n + 1, n):
        t0 = a / fs
        ictal = any(t0 < b and t0 + WIN_S > s for s, b in r["seizures"])
        row = dict(seed=seed, condition=cond, t=t0 + WIN_S / 2, ictal=int(ictal),
                   rE=float(r["rE"][int(t0):int(t0 + WIN_S)].mean()))
        row.update(features(x[a:a + n], fs) if not ictal else dict(SpecEn=np.nan, PermEn=np.nan, p30=np.nan))
        rows.append(row)
    # early-warning windows before each usable untreated seizure, and the same segments of the reference
    ews = []
    if cond in (UNT, REF):
        ref_like = cond == REF
        src = r if not ref_like else lp.load(Path(path).with_name(f"seed{seed}_untreated.npz"))
        prev = -np.inf
        for s0, s1 in src["seizures"]:
            if s0 - PRE_S >= lp.SETTLE_S and prev <= s0 - PRE_S:
                for e in np.arange(-PRE_S, -GAP_S - EWS_WIN_S + 1e-9, EWS_WIN_S):
                    a, b = int((s0 + e) * fs), int((s0 + e + EWS_WIN_S) * fs)
                    f_ = features(x[a:b], fs)
                    ews.append(dict(seed=seed, onset=s0, condition=cond, t_rel=e + EWS_WIN_S / 2,
                                    SpecEn=f_["SpecEn"], PermEn=f_["PermEn"]))
            prev = s1
    return rows, ews, [(float(a), float(b)) for a, b in r["seizures"]], cond, seed, len(r["rE"])


def files():
    out = []
    for p in sorted((R15 / "raw").glob("seed*.npz")):
        if not any(p.stem.endswith(s) for s in LONG):
            out.append(p)
    out += [p for p in sorted((R20 / "raw").glob("seed*.npz")) if any(p.stem.endswith(s) for s in LONG)]
    return out


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    u = stats.mannwhitneyu(pos, neg).statistic
    a = u / (len(pos) * len(neg))
    return max(a, 1 - a), ("higher" if a >= 0.5 else "lower")


def style(a):
    lp.style(a)


def main(n_jobs=4):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(exist_ok=True)
    res = Parallel(n_jobs=n_jobs)(delayed(run_file)(p) for p in files())
    W = [w for rr in res for w in rr[0]]
    E = [e for rr in res for e in rr[1]]
    SZ = {(rr[4], rr[3]): rr[2] for rr in res}
    RUNLEN = {(rr[4], rr[3]): rr[5] for rr in res}
    with open(OUT / "entropy_windows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(W[0]))
        w.writeheader()
        w.writerows(W)
    with open(OUT / "entropy_ews.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(E[0]))
        w.writeheader()
        w.writerows(E)
    seeds = sorted({w["seed"] for w in W})
    conds = [c for c in ORDER if any(w["condition"] == c for w in W)]

    def series(s, c, k):
        R = [w for w in W if w["seed"] == s and w["condition"] == c]
        return np.array([w["t"] for w in R]), np.array([w[k] for w in R], float), np.array([w["ictal"] for w in R])

    # ---- 1. time courses (seed 1) and seizure-free levels
    fig, ax = plt.subplots(len(FEATS), 1, figsize=(11, 7.5), sharex=True)
    for a, (k, lab) in zip(ax, FEATS):
        for c in conds:
            t, v, _ = series(1, c, k)
            a.plot(t, v, color=COL[c], lw=1.0 if c in (UNT, REF) else 0.8, label=LABEL[c])
            for s0, _ in SZ.get((1, c), []):
                if s0 > lp.SETTLE_S:
                    a.plot([s0], [np.nanmin(v)], "v", color=COL[c], ms=5)
        a.set_ylabel(lab)
        style(a)
    ax[0].legend(ncol=4, fontsize=7, frameon=False)
    ax[-1].set_xlabel("time (s)")
    ax[0].set_title("Seed 1: LFP entropy and power, 50 s windows (seizure windows left out); triangles = onsets",
                    loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "entropy_timecourse_seed1.png", dpi=130)
    plt.close(fig)

    summ = []
    for c in conds:
        for s in seeds:
            row = dict(condition=c, seed=s, seizures=sum(a > lp.SETTLE_S for a, _ in SZ.get((s, c), [])),
                       run_s=RUNLEN.get((s, c)))
            for k, _ in FEATS:
                t, v, ic = series(s, c, k)
                row[k] = float(np.nanmedian(v[(t > lp.SETTLE_S) & (ic == 0)]))
            summ.append(row)
    fig, ax = plt.subplots(1, len(FEATS), figsize=(12, 3.6))
    rng = np.random.default_rng(0)
    for i, c in enumerate(conds):
        for a, (k, lab) in zip(ax, FEATS):
            v = [r[k] for r in summ if r["condition"] == c]
            a.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", ms=4, color=COL[c])
            a.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=COL[c], lw=2)
    for a, (k, lab) in zip(ax, FEATS):
        a.set_xticks(range(len(conds)))
        a.set_xticklabels([LABEL[c] for c in conds], rotation=40, ha="right", fontsize=8)
        a.set_title(f"{lab}, seizure-free median", loc="left", fontsize=9)
        style(a)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "entropy_by_condition.png", dpi=130)
    plt.close(fig)

    # ---- 2. early-warning trend
    ktab = []
    for (s, o) in sorted({(e["seed"], e["onset"]) for e in E if e["condition"] == UNT}):
        for c in (UNT, REF):
            R = [e for e in E if e["seed"] == s and e["onset"] == o and e["condition"] == c]
            if R:
                t = np.array([e["t_rel"] for e in R])
                ktab.append(dict(seed=s, onset=o, condition=c,
                                 **{k: float(stats.kendalltau(t, [e[k] for e in R]).statistic)
                                    for k in ("SpecEn", "PermEn")}))

    # ---- 3. pre-ictal vs mid-interictal AUROC (untreated, pooled)
    disc = {}
    for k, _ in FEATS:
        pos, neg = [], []
        for s in seeds:
            t, v, ic = series(s, UNT, k)
            sz = SZ[(s, UNT)]
            ok = (t > lp.SETTLE_S) & (ic == 0) & np.isfinite(v)
            pre = np.zeros(len(t), bool)
            far = ok.copy()
            for a_, b_ in sz:
                pre |= (t >= a_ - PREICTAL_S) & (t < a_)
                far &= ~((t > a_ - FAR_S) & (t < b_ + FAR_S))
            pos += list(v[ok & pre])
            neg += list(v[far])
        disc[k] = (*auroc(pos, neg), len(pos), len(neg))

    with open(OUT / "entropy_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, k in zip(ax, ("SpecEn", "PermEn")):
        for c, dx in ((UNT, -0.15), (REF, 0.15)):
            v = [r[k] for r in ktab if r["condition"] == c]
            a.plot(np.full(len(v), dx) + rng.uniform(-0.05, 0.05, len(v)), v, "o", color=COL[c], ms=4)
            a.plot([dx - 0.1, dx + 0.1], [np.mean(v)] * 2, color=COL[c], lw=2)
        a.axhline(0, color="#9a9890", lw=0.8)
        a.set_xticks([-0.15, 0.15])
        a.set_xticklabels(["untreated", "reference"])
        a.set_ylim(-1, 1)
        a.set_title(f"{dict(FEATS)[k]}: Kendall tau, last 10 min", loc="left", fontsize=9)
        style(a)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "entropy_ews.png", dpi=130)
    plt.close(fig)

    print("seizure-free medians (mean over seeds):")
    for c in conds:
        R = [r for r in summ if r["condition"] == c]
        print(f"  {c:15s} " + "  ".join(f"{k} {np.mean([r[k] for r in R]):.4f}" for k, _ in FEATS))
    print("early-warning Kendall tau (mean over segments):")
    for c in (UNT, REF):
        R = [r for r in ktab if r["condition"] == c]
        print(f"  {c:15s} n={len(R)}: SpecEn {np.mean([r['SpecEn'] for r in R]):+.3f}, "
              f"PermEn {np.mean([r['PermEn'] for r in R]):+.3f}")
    for k in ("SpecEn", "PermEn"):
        u = [r[k] for r in ktab if r["condition"] == UNT]
        rr = [r[k] for r in ktab if r["condition"] == REF]
        print(f"  {k}: untreated vs 0 p = {stats.wilcoxon(u).pvalue:.3f}; untreated vs reference p = "
              f"{stats.wilcoxon(u, rr).pvalue:.3f}")
    print(f"pre-ictal (last {PREICTAL_S/60:.0f} min) vs mid-interictal (> {FAR_S/60:.0f} min from seizures), untreated:")
    for k, (a, d, npos, nneg) in disc.items():
        print(f"  {k:7s} AUROC {a:.3f} ({d} before seizures; {npos} vs {nneg} windows)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
