"""Diagnostics of the local image decoder (reads <out>/raw/*.npz).

    python image_diagnostics.py results/image_random_c16

Per run and readout duration T, on post-settle trials that do not overlap a
seizure:
  1. spike counts of white vs black pixels: P(0), P(1), P(>=2), mean, SD;
  2. the integer decision boundary each local decoder actually uses (the
     smallest count called white, per pixel): adapted (per outer fold) and
     frozen (fitted once on the seed's 40 Hz reference), and how often the
     two decoders make the same call (per pixel and per whole image);
  3. separation: white - black mean count, d', and a threshold-free AUROC of
     the raw counts (white vs black, all pixels pooled), next to BA, recall
     and specificity.
Writes <out>/diagnostics/diagnostics.csv and count_distributions.png.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import image_analysis as ia  # noqa: E402

MAXC = 40


def boundary_from_fit(fit):
    """Smallest integer count c with P(white | c) >= threshold, per pixel
    (local model); MAXC + 1 if never (the pixel is never called white)."""
    C = np.arange(MAXC + 1, dtype=float)
    Xc = np.repeat(C[:, None], 100, axis=1)            # row c: every cell fired c spikes
    white = ia.predict_proba(fit, Xc) >= fit["threshold"]
    b = np.where(white.any(0), white.argmax(0), MAXC + 1)
    monotone = np.all(np.diff(white.astype(int), axis=0) >= 0, axis=0)
    return b, monotone


def adapted_with_fits(X, Y, tt, g, q):
    """ia.run_adapted for the local model, also returning each fold's fit."""
    post = np.flatnonzero(tt["post"])
    S = np.full(Y.shape, np.nan)
    thr = np.full(len(Y), np.nan)
    fits = []
    for test in [post[c] for c in ia._blocked(len(post), g["outer_folds"])]:
        lo, hi = test[0], test[-1]
        pos_lo, pos_hi = np.searchsorted(post, lo), np.searchsorted(post, hi)
        keep = np.r_[post[:max(pos_lo - g["gap_trials"], 0)], post[pos_hi + 1 + g["gap_trials"]:]]
        tr = keep[tt["train_ok"][keep]]
        fit = ia.fit_selected(X[tr], Y[tr], "local", g, q)
        S[test] = ia.predict_proba(fit, X[test])
        thr[test] = fit["threshold"]
        fits.append(fit)
    return S, thr, fits


def diagnose_seed(paths_by_cond, g):
    q = g["k_white"] / 100.0
    runs = {c: ia.load_run(p) for c, p in paths_by_cond.items()}
    tts = {c: ia.trial_table(r, g) for c, r in runs.items()}
    ref = tts.get(ia.REFERENCE)
    rows = []
    for T in g["readout_ms"]:
        frozen = None
        if ref is not None:
            post = np.flatnonzero(ref["post"])
            n_fit = int(len(post) * g["frozen_train_frac"])
            fit_idx = post[:n_fit][ref["train_ok"][post[:n_fit]]]
            frozen = ia.fit_selected(ref["feats"][T][fit_idx], ref["Y"][fit_idx], "local", g, q)
            fb, fmono = boundary_from_fit(frozen)
        for c, tt in tts.items():
            X, Y = tt["feats"][T], tt["Y"]
            inter = tt["post"] & ~tt["seizure_overlap"]
            w, b = X[inter][Y[inter] == 1], X[inter][Y[inter] == 0]
            S, thr, fits = adapted_with_fits(X, Y, tt, g, q)
            pa = S >= thr[:, None]
            sa = ia.trial_scores(S[inter], Y[inter], thr[inter][:, None], q)
            bounds = np.concatenate([boundary_from_fit(f)[0] for f in fits])
            row = dict(seed=runs[c]["meta"]["seed"], condition=c, T_ms=T,
                       white_P0=np.mean(w == 0), white_P1=np.mean(w == 1), white_P2plus=np.mean(w >= 2),
                       black_P0=np.mean(b == 0), black_P1=np.mean(b == 1), black_P2plus=np.mean(b >= 2),
                       white_mean=w.mean(), black_mean=b.mean(), white_sd=w.std(), black_sd=b.std(),
                       diff=w.mean() - b.mean(),
                       dprime=(w.mean() - b.mean()) / np.sqrt((w.var() + b.var()) / 2),
                       count_AUROC=float(stats.mannwhitneyu(w, b).statistic / (len(w) * len(b))),
                       adapted_BA=sa["BA"].mean(), adapted_recall=sa["recall"].mean(),
                       adapted_spec=sa["spec"].mean(),
                       adapted_boundary_eq1=np.mean(bounds == 1), adapted_boundary_eq2=np.mean(bounds == 2),
                       adapted_boundary_ge3=np.mean((bounds >= 3) & (bounds <= MAXC)),
                       adapted_boundary_never=np.mean(bounds > MAXC),
                       adapted_boundary_median=float(np.median(bounds)))
            if frozen is not None:
                Sf = ia.predict_proba(frozen, X)
                pf = Sf >= frozen["threshold"]
                sf = ia.trial_scores(Sf[inter], Y[inter], frozen["threshold"], q)
                row.update(frozen_BA=sf["BA"].mean(), frozen_recall=sf["recall"].mean(),
                           frozen_spec=sf["spec"].mean(),
                           frozen_boundary_eq1=np.mean(fb == 1), frozen_boundary_eq2=np.mean(fb == 2),
                           frozen_boundary_median=float(np.median(fb)),
                           frozen_boundary_monotone=float(np.mean(fmono)),
                           same_call_pixels=np.mean(pa[inter] == pf[inter]),
                           same_call_images=np.mean(np.all(pa[inter] == pf[inter], axis=1)))
            rows.append(row)
    return rows


def plot(rows, path, T_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from image_plots import _order
    conds = _order({r["condition"] for r in rows})
    fig, ax = plt.subplots(1, len(T_list), figsize=(4.5 * len(T_list), 5), sharey=True, squeeze=False)
    for j, T in enumerate(T_list):
        a = ax[0, j]
        for k, (lab, keys) in enumerate((("white", ("white_P0", "white_P1", "white_P2plus")),
                                         ("black", ("black_P0", "black_P1", "black_P2plus")))):
            for i, c in enumerate(conds):
                v = [np.mean([r[kk] for r in rows if r["condition"] == c and r["T_ms"] == T]) for kk in keys]
                x = i + (k - 0.5) * 0.38
                bottom = 0.0
                for vv, col in zip(v, ("#d9d7cf", "#86b6ef", "#1c5cab") if k == 0 else ("#e8e6df", "#f2b48f", "#c4314b")):
                    a.bar(x, vv, 0.36, bottom=bottom, color=col, edgecolor="white", lw=0.5)
                    bottom += vv
        a.set_xticks(range(len(conds)))
        a.set_xticklabels(conds, rotation=35, ha="right", fontsize=8)
        a.set_title(f"T = {T} ms: left bar white, right bar black\nlight -> dark = 0, 1, >=2 spikes",
                    fontsize=9, loc="left")
    ax[0, 0].set_ylabel("fraction of pixel-trials (interictal, mean over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(out_dir, n_jobs=6):
    from joblib import Parallel, delayed
    out_dir = Path(out_dir)
    raw = sorted((out_dir / "raw").glob("*.npz"))
    g = ia.protocol_from_runs(raw)
    import json
    by_seed = {}
    for p in raw:
        m = json.loads(str(np.load(p)["meta"]))
        by_seed.setdefault(m["seed"], {})[m["condition"]] = p
    res = Parallel(n_jobs=min(n_jobs, len(by_seed)), backend="loky")(
        delayed(diagnose_seed)(v, g) for v in by_seed.values())
    rows = [r for rr in res for r in rr]
    d = out_dir / "diagnostics"
    d.mkdir(exist_ok=True)
    with open(d / "diagnostics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                                                if k in rows[0] else 999))
        w.writeheader()
        w.writerows(rows)
    plot(rows, d / "count_distributions.png", list(g["readout_ms"]))
    return rows, g


if __name__ == "__main__":
    rows, g = main(sys.argv[1] if len(sys.argv) > 1 else THIS_DIR / "results" / "image_random_c16")
    from image_plots import _order
    for T in g["readout_ms"]:
        print(f"\n=== T = {T} ms (interictal, mean over seeds)")
        print(f"{'condition':15s} {'white P0/P1/P2+':>17s} {'black P0/P1/P2+':>17s} {'diff':>6s} {'dprime':>6s} "
              f"{'cntAUC':>6s} | {'BA':>5s} {'rec':>5s} {'spec':>5s} | bound(ad) =1  =2  med | frozen =1  med | same pix  img")
        for c in _order({r["condition"] for r in rows}):
            R = [r for r in rows if r["condition"] == c and r["T_ms"] == T]
            m = lambda k: np.mean([r[k] for r in R if k in r]) if any(k in r for r in R) else np.nan  # noqa: E731
            print(f"{c:15s} {m('white_P0'):.2f}/{m('white_P1'):.2f}/{m('white_P2plus'):.2f}   "
                  f"{m('black_P0'):.2f}/{m('black_P1'):.2f}/{m('black_P2plus'):.2f}   {m('diff'):6.3f} {m('dprime'):6.3f} "
                  f"{m('count_AUROC'):6.3f} | {m('adapted_BA'):.3f} {m('adapted_recall'):.3f} {m('adapted_spec'):.3f} | "
                  f"{m('adapted_boundary_eq1'):.2f} {m('adapted_boundary_eq2'):.2f} {m('adapted_boundary_median'):3.0f} | "
                  f"{m('frozen_boundary_eq1'):.2f} {m('frozen_boundary_median'):3.0f} | "
                  f"{m('same_call_pixels'):.3f} {m('same_call_images'):.3f}")
