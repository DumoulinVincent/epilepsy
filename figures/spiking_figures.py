"""Paper figures from the spiking-network results: LFP cycle, early-warning signals, treatments, image task,
Epileptor outlook (fig1-fig8). Called by make_paper_figures.py.

Sources (all in ../../spiking_network/results/; raw runs are not in git, see the top-level README):
  lfp_tau250_15k/   untreated, reference 40 Hz, benzo, DBS 20/60/100/300 pA;
                    5 seeds x 15000 s, HSP tau 250 s (raw/, plots/, ews/, treatment/)
  lfp_tau250_20k/   benzo and DBS 20 pA, 5 seeds x 20000 s (treatment/)
  image_random_c16/ image recovery, 16 Hz contrast, 6 conditions x 5 seeds
Needs the spiking_network modules (lfp_tau_plot, image_analysis,
report/build_lfp_image_report); the Epileptor figure reads ../../epileptor/results/trajectories.npz.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                     # repository root (this folder: figures/)
V2 = ROOT / "spiking_network"
RES = V2 / "results"
FIGS = HERE / "out"
for p in (V2, V2 / "report"):
    sys.path.insert(0, str(p))
import lfp_tau_plot as lp  # noqa: E402
import image_analysis as ia  # noqa: E402
import build_lfp_image_report as R  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

COL, LABEL, ORDER, INK, MUTED, GRID = R.COL, R.LABEL, R.ORDER, R.INK, R.MUTED, R.GRID
UNT, REF, BZ = R.UNT, R.REF, R.BZ
style = R.style


# ===================================================================== copy
# ================================================================== figures
def fig_cycle(seed, path):
    """Untreated vs reference over a 15000 s run: E rate, HSP gains, LFP power, PSD."""
    runs = {c: lp.load(RES / f"lfp_tau250_15k/raw/seed{seed}_{s}.npz")
            for c, s in ((UNT, "untreated"), (REF, "reference_40Hz"))}
    W = pd.read_csv(RES / "lfp_tau250_15k/treatment/p30_windows.csv")
    fig = plt.figure(figsize=(6.8, 6.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[2.6, 2.6, 1.9], hspace=0.35, wspace=0.55)
    a0 = fig.add_subplot(gs[0, :2])
    a1 = fig.add_subplot(gs[1, :2], sharex=a0)
    a2 = fig.add_subplot(gs[2, :2], sharex=a0)
    a3 = fig.add_subplot(gs[:, 2])
    for c, r in runs.items():
        a0.plot(r["rate_t"], np.maximum(r["rE"], 0.05), color=COL[c], lw=0.4, label=LABEL[c])
        kA, kG = (np.asarray(r[k]) for k in ("kA", "kG"))
        if kA.ndim == 2:
            ax_ = 1 if kA.shape[0] == len(r["t_k"]) else 0
            kA, kG = kA.mean(ax_), kG.mean(ax_)
        ls = "-" if c == UNT else ":"
        a1.plot(r["t_k"], kA, ls, color="#eda100", lw=1.3, label=f"$k_{{AMPA}}$, {LABEL[c]}")
        a1.plot(r["t_k"], kG, ls, color="#4a3aa7", lw=1.3, label=f"$k_{{GABA}}$, {LABEL[c]}")
        x = W[(W.seed == seed) & (W.condition == c)].sort_values("t")
        a2.plot(x.t, np.where(x.ictal == 1, np.nan, x.p30_smoothed), color=COL[c], lw=1.3, label=LABEL[c])
        f, P, n = lp.welch_masked(r["lfp"], lp.valid_mask(r), r["fs"], int(4 * r["fs"]))
        sel = (f >= 0.25) & (f <= 200)
        a3.loglog(f[sel], P[sel], color=COL[c], lw=1.2, label=LABEL[c])
    a0.set_yscale("log")
    a0.set_ylim(0.3, 500)
    a0.set_ylabel("E rate (Hz, 1 s)")
    a0.legend(frameon=False, loc="upper right", ncol=2)
    a1.set_ylabel("mean HSP gain")
    a1.set_ylim(0, 1.05)
    a1.legend(frameon=False, ncol=2, fontsize=6, loc="center right")
    a2.set_ylabel("LFP power 0–30 Hz (pA²)")
    a2.set_xlabel("time (s)")
    a2.set_ylim(bottom=0)
    for a, t in ((a0, "(a) firing rate"), (a1, "(b) homeostatic gains"), (a2, "(c) LFP power 0–30 Hz")):
        a.set_title(t, loc="left")
        a.axvspan(0, lp.SETTLE_S, color=GRID, alpha=0.6, lw=0)
        for s0, _ in runs[UNT]["seizures"]:
            if s0 > lp.SETTLE_S:
                a.axvline(s0, color="#e34948", lw=0.7, alpha=0.7)
        style(a)
    plt.setp(a0.get_xticklabels(), visible=False)
    plt.setp(a1.get_xticklabels(), visible=False)
    a3.set_title("(d) PSD between seizures", loc="left")
    a3.set_xlabel("frequency (Hz)")
    a3.set_ylabel("PSD (pA²/Hz)")
    a3.axvspan(0.25, 30, color=GRID, alpha=0.5, lw=0)
    a3.legend(frameon=False, loc="lower left")
    style(a3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ews_stats():
    E = pd.read_csv(RES / "lfp_tau250_15k/ews/ews_windows.csv")
    K = pd.read_csv(RES / "lfp_tau250_15k/ews/ews_kendall.csv")
    feats = ["mean", "var", "cv", "ac1_100", "ac1_10", "p_lo", "p_mid", "rE"]
    out = {}
    for k in feats:
        u = K[K.condition == UNT].sort_values(["seed", "onset"])[k].to_numpy()
        r = K[K.condition == REF].sort_values(["seed", "onset"])[k].to_numpy()
        out[k] = dict(u=u.mean(), r=r.mean(), p0=stats.wilcoxon(u).pvalue, pd=stats.wilcoxon(u, r).pvalue)
    return E, K, out


def fig_ews(E, K, path):
    feats = (("var", "variance", "ratio"), ("ac1_100", "lag-1 autocorrelation", "diff"),
             ("p_mid", "power 3–30 Hz", "ratio"))
    fig, ax = plt.subplots(1, 4, figsize=(6.8, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
    for a, (k, lab, kind) in zip(ax, feats):
        for c in (REF, UNT):
            C = []
            for (s, o), g in E[E.condition == c].groupby(["seed", "onset"]):
                g = g.sort_values("t_rel")
                v = g[k].to_numpy()
                base = v[g.t_rel.to_numpy() < -300].mean()
                C.append(v / base if kind == "ratio" else v - base)
            C = np.array(C)
            t = np.sort(E.t_rel.unique())
            m, se = C.mean(0), C.std(0) / np.sqrt(len(C))
            a.fill_between(t, m - se, m + se, color=COL[c], alpha=0.2, lw=0)
            a.plot(t, m, color=COL[c], lw=1.1, label=LABEL[c])
        a.axhline(1 if kind == "ratio" else 0, color=MUTED, lw=0.7)
        a.set_title(lab, loc="left")
        a.set_xlabel("time to onset (s)")
        style(a)
    ax[0].set_ylabel("relative to first 5 min")
    ax[0].legend(frameon=False, loc="upper left")
    names = [("mean", "mean"), ("var", "var"), ("cv", "CV"), ("ac1_100", "AC 100"), ("ac1_10", "AC 10"),
             ("p_lo", "P<3"), ("p_mid", "P3–30")]
    rng = np.random.default_rng(0)
    for i, (k, n) in enumerate(names):
        for c, dx in ((UNT, -0.18), (REF, 0.18)):
            v = K[K.condition == c][k].to_numpy()
            ax[3].plot(i + dx + rng.uniform(-0.06, 0.06, len(v)), v, "o", ms=2, color=COL[c], alpha=0.7)
            ax[3].plot([i + dx - 0.12, i + dx + 0.12], [v.mean()] * 2, color=COL[c], lw=1.6)
    ax[3].axhline(0, color=MUTED, lw=0.7)
    ax[3].set_xticks(range(len(names)))
    ax[3].set_xticklabels([n for _, n in names], rotation=60, ha="right", fontsize=6.5)
    ax[3].set_ylim(-0.6, 0.6)
    ax[3].set_title("Kendall τ vs time", loc="left")
    style(ax[3])
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def image_trials(dirname, slug):
    z = np.load(RES / dirname / "analysis" / f"{slug}_trials.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def fig_image_task(path):
    """(a) BA vs readout duration (adapted, interictal); (b) BA across the seizure cycle (untreated, seed 1);
    (c) BA vs time since the previous seizure, pooled over seeds (untreated)."""
    D = R.image_data()
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw=dict(width_ratios=[1, 1.5, 1.1]))
    y = D[(D.phase == "interictal") & (D.regime == "adapted")]
    for c in [c for c in ORDER if c in set(D.condition) and c != "DBS 300pA"]:
        g = y[y.condition == c].groupby("T_ms").BA.mean()
        ax[0].plot(g.index, g.values, "-o", ms=2.5, lw=1.1, color=COL[c], label=LABEL[c])
    ax[0].set_xscale("log")
    ax[0].set_xticks([50, 100, 250, 500])
    ax[0].set_xticklabels(["50", "100", "250", "500"])
    ax[0].set_xlabel("readout duration (ms)")
    ax[0].set_ylabel("balanced accuracy")
    ax[0].legend(frameon=False, fontsize=5.8, loc="upper left")
    ax[0].set_title("(a) between seizures", loc="left")
    # (b) cycle, seed 1 untreated: 50 s bins
    z = image_trials("image_random_c16", "seed1_untreated")
    t, ba, post = z["tt_onset_s"], z["250_local_adapted_BA"], z["tt_post"]
    sz = z["tt_seizure_overlap"]
    edges = np.arange(500, t.max() + 50, 50)
    idx = np.digitize(t, edges)
    mb = [ba[(idx == i) & post].mean() if np.any((idx == i) & post) else np.nan for i in range(1, len(edges))]
    ax[1].plot(edges[:-1] + 25, mb, color=COL[UNT], lw=1.0)
    run = ia.load_run(RES / "image_random_c16/raw/seed1_untreated.npz")
    for s0, _ in np.asarray(run["seizures"]).reshape(-1, 2):
        if s0 > 500:
            ax[1].axvline(s0, color="#e34948", lw=0.7, alpha=0.7)
    ax[1].plot(t[sz & post], ba[sz & post], "x", ms=3, color="#e34948", mew=0.8, label="trials during a seizure")
    ax[1].set_xlabel("time (s)")
    ax[1].set_title("(b) untreated, seed 1 (50 s means)", loc="left")
    ax[1].legend(frameon=False, loc="lower right", fontsize=6)
    ax[1].set_ylim(0.3, 0.85)
    # (c) BA vs time since the previous seizure (untreated, all seeds)
    T_all, B_all, S_all = [], [], []
    for s in range(1, 6):
        z = image_trials("image_random_c16", f"seed{s}_untreated")
        ok = z["tt_post"]
        T_all.append(z["tt_t_since"][ok])
        B_all.append(z["250_local_adapted_BA"][ok])
        S_all.append(z["tt_seizure_overlap"][ok])
    T_all, B_all, S_all = map(np.concatenate, (T_all, B_all, S_all))
    e = np.arange(0, 1801, 100)
    m = [B_all[(T_all >= a) & (T_all < b) & ~S_all].mean() for a, b in zip(e[:-1], e[1:])]
    se = [B_all[(T_all >= a) & (T_all < b) & ~S_all].std() / np.sqrt(((T_all >= a) & (T_all < b) & ~S_all).sum())
          for a, b in zip(e[:-1], e[1:])]
    xc = (e[:-1] + e[1:]) / 2
    ax[2].errorbar(xc, m, yerr=se, color=COL[UNT], lw=1.0, ms=2.5, fmt="-o")
    ax[2].errorbar([-60], [B_all[S_all].mean()], yerr=[B_all[S_all].std() / np.sqrt(S_all.sum())], fmt="x",
                   color="#e34948", ms=5, mew=1.2)
    ax[2].annotate("during\nseizure", (-60, B_all[S_all].mean()), xytext=(60, 0.6), fontsize=6.5, color=MUTED)
    ax[2].set_xlabel("time since previous seizure (s)")
    ax[2].set_title("(c) untreated, 5 seeds", loc="left")
    for a in ax:
        style(a)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return float(B_all[S_all].mean()), int(S_all.sum())


def frozen_predictions(cache):
    """Frozen decoder (seed 1, T = 250 ms) fitted on the reference, applied to DBS 300 pA."""
    if cache.exists():
        z = np.load(cache)
        return z["S"], float(z["thr"])
    raw = [RES / "image_random_c16/raw/seed1_reference_40Hz.npz", RES / "image_random_c16/raw/seed1_DBS_300pA.npz"]
    g = dict(ia.protocol_from_runs(raw))
    g["readout_ms"] = (250,)
    runs = {ia.load_run(p)["meta"]["condition"]: ia.load_run(p) for p in raw}
    ref = ia.trial_table(runs[REF], g)
    post = np.flatnonzero(ref["post"])
    n_fit = int(len(post) * g["frozen_train_frac"])
    idx = post[:n_fit][ref["train_ok"][post[:n_fit]]]
    fit = ia.fit_selected(ref["feats"][250][idx], ref["Y"][idx], "local", g, g["k_white"] / 100)
    d3 = ia.trial_table(runs["DBS 300pA"], g)
    S = ia.predict_proba(fit, d3["feats"][250])
    np.savez(cache, S=S, thr=fit["threshold"])
    return S, float(fit["threshold"])


def fig_examples(path):
    masks = ia.load_run(RES / "image_random_c16/raw/seed1_reference_40Hz.npz")
    M, mid = masks["masks"], masks["mask_id"]
    cols = [(UNT, "image_random_c16", "adapted"), (REF, "image_random_c16", "adapted"),
            ("DBS 60pA", "image_random_c16", "adapted"), ("DBS 300pA", "image_random_c16", "adapted"),
            ("DBS 300pA", "image_random_c16", "frozen")]
    slug = {UNT: "untreated", REF: "reference_40Hz", "DBS 60pA": "DBS_60pA", "DBS 300pA": "DBS_300pA",
            "DBS 20pA": "DBS_20pA"}
    T = {c: image_trials(d, f"seed1_{slug[c]}") for c, d, _ in cols}
    Sf, thr_f = frozen_predictions(FIGS / "frozen_dbs300_seed1.npz")
    ok = np.ones(len(mid), bool)
    for z in T.values():
        ok &= z["tt_post"] & ~z["tt_seizure_overlap"]
    trials = [j for j in (501, 572) if ok[j]]
    trials += [int(j) for j in np.flatnonzero(ok) if j > 3000][:3 - len(trials)]
    # a seizure-overlapping trial (untreated, seed 1)
    zu = T[UNT]
    szu = np.asarray(ia.load_run(RES / "image_random_c16/raw/seed1_untreated.npz")["seizures"]).reshape(-1, 2)
    inside = np.zeros(len(mid), bool)
    for s0, s1 in szu:
        inside |= (zu["tt_onset_s"] >= s0) & (zu["tt_onset_s"] < s1)
    cand = np.flatnonzero(inside & zu["tt_post"])
    ba_c = zu["250_local_adapted_BA"][cand]
    jz = int(cand[np.argmin(np.abs(ba_c - np.median(ba_c)))])      # typical seizure trial
    rows = [(j, False) for j in trials] + [(jz, True)]
    fig, ax = plt.subplots(len(rows), len(cols) + 1, figsize=(6.6, 1.25 * len(rows) + 0.4))

    def draw(a, y, pred=None):
        img = y.reshape(10, 10) if pred is None else pred.reshape(10, 10)
        if pred is None:
            a.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            rgb = np.zeros((10, 10, 3))
            truth = y.reshape(10, 10).astype(bool)
            p = pred.reshape(10, 10).astype(bool)
            rgb[p & truth] = (1, 1, 1)                       # hit
            rgb[p & ~truth] = (0.55, 0.55, 0.55)             # false white
            a.imshow(rgb)
            for (yy, xx) in zip(*np.nonzero(truth & ~p)):    # miss
                a.add_patch(Rectangle((xx - 0.45, yy - 0.45), 0.9, 0.9, fill=False, ec="#eb6834", lw=1.0))
        a.set_xticks([])
        a.set_yticks([])

    for r, (j, ictal) in enumerate(rows):
        y = M[mid[j]].astype(float)
        draw(ax[r, 0], y)
        ax[r, 0].set_ylabel(f"trial {j}" + ("\n(seizure)" if ictal else ""), fontsize=7)
        for k, (c, d, reg) in enumerate(cols):
            a = ax[r, k + 1]
            if ictal and k > 0:
                a.axis("off")
                continue
            if reg == "frozen":
                S, thr = Sf[j], thr_f
            else:
                S, thr = T[c]["250_local_adapted_S"][j], T[c]["250_local_adapted_thr"][j]
            pred = S >= thr
            draw(a, y, pred)
            rec = (pred & (y > 0)).sum() / y.sum()
            spec = (~pred & (y == 0)).sum() / (y == 0).sum()
            a.set_xlabel(f"BA {(rec + spec) / 2:.2f}", fontsize=6.5, labelpad=1)
    heads = ["image"] + [f"{LABEL[c]}\n{reg}" for c, _, reg in cols]
    for k, h in enumerate(heads):
        ax[0, k].set_title(h, fontsize=7)
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ===================================================================== text
def fig_epileptor_outlook(path):
    """Outlook: the HSP state of the 36 Hz (deafferented) and 40 Hz (healthy) networks mapped onto the excitability x0
    of the 2D Epileptor (HSP_Epileptor_trajectory.ipynb export)."""
    Z = np.load(ROOT / "epileptor" / "results" / "trajectories.npz")
    X0C, ZSN, IE = float(Z["X0_CRIT"]), float(Z["Z_SN"]), float(Z["I_EXT"])
    G = lambda x1: -x1**3 - 2 * x1**2 + 1 + IE  # noqa: E731
    seeds = [int(v) for v in Z["seeds"]]
    s = seeds[0]
    t, x0, ic = Z[f"s{s}|{UNT}|t"], Z[f"s{s}|{UNT}|x0"], Z[f"s{s}|{UNT}|ictal"]
    sz = Z[f"s{s}|{UNT}|seizures"]
    sz = sz[sz[:, 0] > lp.SETTLE_S]
    e0, on1 = sz[0, 1], sz[1, 0]                       # end of the first episode, onset of the next
    snaps = [e0 + 300, e0 + 900, e0 + 1800, 0.5 * (e0 + on1) + 600, on1 - 60]
    cols = ["#2a78d6", "#5b9be0", "#9a9890", "#eda100", "#e34948"]
    x0r = float(np.median(Z[f"s{s}|{REF}|x0"][Z[f"s{s}|{REF}|t"] > lp.SETTLE_S]))

    fig = plt.figure(figsize=(7.0, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.25], hspace=0.45, wspace=0.28)
    a0 = fig.add_subplot(gs[0, :]); a1 = fig.add_subplot(gs[1, 0]); a2 = fig.add_subplot(gs[1, 1])
    # (a) x0(t)
    m = t > lp.SETTLE_S
    a0.plot(t[m] / 3600, np.where(ic, np.nan, x0)[m], color=COL[UNT], lw=1.2, label="deafferented (36 Hz)")
    tr = Z[f"s{s}|{REF}|t"]
    a0.plot(tr[tr > lp.SETTLE_S] / 3600, Z[f"s{s}|{REF}|x0"][tr > lp.SETTLE_S], color=COL[REF], lw=1.2,
            label="healthy (40 Hz)")
    for a_, b_ in sz:
        a0.axvspan(a_ / 3600, b_ / 3600, color="#e34948", alpha=0.35, lw=0)
    for ts, col in zip(snaps, cols):
        k = int(np.argmin(np.abs(t - ts)))
        a0.plot(t[k] / 3600, x0[k], "o", color=col, ms=5, mec="white", mew=0.6, zorder=5)
    a0.axhline(X0C, color=INK, ls="--", lw=0.9)
    a0.text(t[m][-1] / 3600, X0C + 0.06, "$x_{0,crit}$", ha="right", fontsize=7)
    a0.set_xlabel("time (h, model)"); a0.set_ylabel("$x_0^{HSP}$")
    a0.legend(frameon=False, fontsize=6.5, loc="lower right", ncol=2)
    a0.set_title("(a) excitability $x_0^{HSP}(t)$ from the HSP gains (seed 1; red: high-rate episodes)", loc="left")
    # (b) phase plane following x0
    x1 = np.linspace(-2.7, -0.7, 300)
    a1.plot(x1, G(x1), color=INK, lw=1.6, label="fast nullcline")
    for ts, col in zip(snaps, cols):
        k = int(np.argmin(np.abs(t - ts)))
        v = float(x0[k])
        a1.plot(x1, 4 * (x1 - v), color=col, lw=1.2, label=f"36 Hz, $x_0$ = {v:.2f}")
    a1.plot(x1, 4 * (x1 - x0r), color=COL[REF], lw=1.4, ls="--", label=f"40 Hz, $x_0$ = {x0r:.2f}")
    xr = [r.real for r in np.roots([-1.0, -2.0, -4.0, 1 + IE + 4 * x0r]) if abs(r.imag) < 1e-9][0]
    a1.plot(xr, 4 * (xr - x0r), "o", ms=5, mfc=COL[REF], mec="white", zorder=6)
    a1.plot(-4 / 3, ZSN, "o", color="#e34948", ms=5, zorder=6)
    a1.annotate("fold", (-4 / 3, ZSN), xytext=(-1.15, 2.55), fontsize=7, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a1.set_xlim(-2.7, -0.7); a1.set_ylim(2.3, 6.5)
    a1.set_xlabel("$x_1$"); a1.set_ylabel("$z$")
    a1.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    a1.set_title("(b) Epileptor phase plane following $x_0^{HSP}$", loc="left")
    # (c) trajectory in (x0, z), 36 Hz all seeds coloured by phase, 40 Hz
    PH = (("build-up", "#9a9890"), ("last 5 min", "#eda100"), ("episode", "#e34948"), ("post-episode", "#2a78d6"))
    for ss in seeds:
        tt, xx, zz, ii = (Z[f"s{ss}|{UNT}|{k}"] for k in ("t", "x0", "z", "ictal"))
        lab = np.full(len(tt), "build-up", dtype=object)
        for a_, b_ in Z[f"s{ss}|{UNT}|seizures"]:
            lab[(tt >= a_ - 300) & (tt < a_)] = "last 5 min"
            lab[(tt >= b_) & (tt < b_ + 600)] = "post-episode"
        lab[ii] = "episode"
        keep = tt > lp.SETTLE_S
        for p_, col in PH:
            k = keep & (lab == p_)
            a2.plot(xx[k], zz[k], ".", ms=1.5, color=col, label=f"36 Hz, {p_}" if ss == seeds[0] else None)
        tr = Z[f"s{ss}|{REF}|t"]
        a2.plot(Z[f"s{ss}|{REF}|x0"][tr > lp.SETTLE_S], Z[f"s{ss}|{REF}|z"][tr > lp.SETTLE_S], ".", ms=4,
                color=COL[REF], label="40 Hz (all seeds)" if ss == seeds[0] else None)
    a2.annotate("40 Hz: fixed point", (x0r, 3.05), xytext=(-3.3, 3.25), fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a2.axvline(X0C, color=INK, ls="--", lw=0.9); a2.axhline(ZSN, color=INK, ls=":", lw=0.9)
    a2.set_xlim(-4.0, -1.95); a2.set_ylim(2.6, 9.6)
    a2.set_xlabel("$x_0^{HSP}$"); a2.set_ylabel("$z$")
    a2.legend(frameon=False, fontsize=6, markerscale=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    a2.set_title("(c) trajectory in $(x_0^{HSP}, z)$, 5 seeds", loc="left")
    for a in (a0, a1, a2):
        style(a)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def tables():
    W, T, run_s = R.lfp_data()
    S = R.lfp_summary(W, T, run_s)
    lt = []
    for r in S:
        c, P = r["cond"], r["per"]
        th = T[(T.condition == c) & (T.role.isin(["held out", "treatment"]))]
        fr = T[T.condition == c].groupby("seed").alarm_time_fraction.first().mean()
        on = th.alarm_on.astype(str).eq("True")
        lead = th[on].alarm_on_for_s.astype(float)
        lt.append(" & ".join([
            LABEL[c], f"{run_s[c]:.0f}", f"{int(P.n_sz.sum())}",
            f"{np.nanmean(P['first']):.0f}" if P.n_sz.sum() else "--",
            f"{P.rel.mean():.2f} $\\pm$ {P.rel.std():.2f}", f"{P.rE.mean():.2f}",
            f"{fr:.2f}" + ("$^a$" if c == UNT else ""), f"{int(on.sum())}/{len(th)}" if len(th) else "--",
            f"{lead.median() / 60:.0f}" if len(lead) else "--"]) + r" \\")
    D = R.image_data()
    x = D[D.T_ms == 250]
    it = []
    for c in [c for c in ORDER if c in set(D.condition)]:
        sel = x[(x.condition == c) & (x.phase == "interictal")]
        m, sd = sel.groupby("regime")[["BA", "recall", "spec", "AUROC"]].mean(), sel.groupby("regime").BA.std()
        n = sel[sel.regime == "adapted"].seed.nunique()
        s = x[(x.condition == c) & (x.phase == "seizure_overlap") & (x.regime == "adapted")].BA
        pm = (lambda k: f" $\\pm$ {sd[k]:.3f}") if n > 1 else (lambda k: "")
        it.append(" & ".join([
            LABEL[c], str(n), f"{m.loc['adapted', 'BA']:.3f}{pm('adapted')}", f"{m.loc['frozen', 'BA']:.3f}{pm('frozen')}",
            f"{m.loc['adapted', 'AUROC']:.3f}",
            f"{m.loc['adapted', 'recall']:.2f} / {m.loc['adapted', 'spec']:.2f}",
            f"{m.loc['frozen', 'recall']:.2f} / {m.loc['frozen', 'spec']:.2f}",
            f"{s.mean():.3f}" if len(s) else "--"]) + r" \\")
    return S, T, D, "\n".join(lt), "\n".join(it)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_cycle(3, FIGS / "fig1_cycle.png")
    E, K, ews = ews_stats()
    fig_ews(E, K, FIGS / "fig2_ews.png")
    W, T_, run_s = R.lfp_data()
    R.fig_lfp_timecourse(W, 1, FIGS / "fig3_treatments.png")
    S, T, D, lfp_rows, img_rows = tables()
    R.fig_lfp_dose(S, FIGS / "fig4_dose.png")
    fig_image_task(FIGS / "fig5_image_task.png")
    R.fig_image(D, FIGS / "fig6_bounds.png")
    fig_examples(FIGS / "fig7_examples.png")
    print("wrote fig1-fig7 to", FIGS)


if __name__ == "__main__":
    main()
