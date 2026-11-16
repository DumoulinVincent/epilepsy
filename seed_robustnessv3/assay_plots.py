"""Figures A-F for the functional assays (called by assay_analysis.main).

  A_state_<run>.png      evolving state: E/I rate, HSP gains, ictal shading,
                         per-checkpoint encoding accuracy (rolling mean)
  B_encoding.png         C_encode per condition: feature views x readout regimes
  C_noise.png            BA(eta) per condition, adapted and fixed readouts
  D_lags.png             held-out R^2 vs delay (ms), d = 0 marked, with nulls
  E_tradeoffs.png        burden S vs C_encode / C_noise / S_memory, seed points,
                         means, Pareto set (frontier only a visual guide)
  F_conditional.png      encoding BA in ictal / recovery / interictal
                         checkpoints, with the native time share of each phase
"""
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, SZ = "#0b0b0b", "#52514e", "#e4e3de", "#d03b3b"
_DBS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]
COLORS = {"untreated": "#eb6834", "benzo (k=1.0)": "#1baf7a", "reference 40Hz": "#52514e",
          **{f"DBS {a}pA": c for a, c in zip(range(10, 101, 10), _DBS)}}


def color(c):
    return COLORS.get(c, "#0d366b" if c.startswith("DBS") else MUTED)


def order(conds):
    def key(c):
        if c == "reference 40Hz":
            return (0, 0)
        if c == "untreated":
            return (1, 0)
        if c.startswith("benzo"):
            return (2, 0)
        if c.startswith("DBS"):
            return (3, float(c.split()[1][:-2]))
        return (4, 0)
    return sorted(conds, key=key)


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _ms(summ, cond, key):
    v = np.array([r[key] for r in summ if r["condition"] == cond and key in r], float)
    v = v[np.isfinite(v)]
    return (v.mean(), v.std(), v) if v.size else (np.nan, np.nan, v)


def fig_state(out_dir, plots):
    from assay_analysis import load_run, phases, encoding_trials
    import csv
    trials = {}
    p = Path(out_dir) / "analysis" / "trials_encoding.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            trials.setdefault((int(r["seed"]), r["condition"]), []).append(r)
    for path in sorted((Path(out_dir) / "raw").glob("*.npz")):
        run = load_run(path)
        g, m = run["g"], run["meta"]
        fig, ax = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
        t = np.arange(len(run["rE"])) + 0.5
        ax[0].plot(t, run["rE"], color="#c4314b", lw=0.6, label="E")
        ax[0].plot(t, run["rI"], color="#3a6fb0", lw=0.6, label="I")
        ax[0].set_yscale("symlog", linthresh=1)
        ax[0].set_ylim(bottom=0)
        ax[0].set_ylabel("rate (Hz, 1 s)")
        ax[0].legend(fontsize=8, frameon=False)
        ax[1].plot(run["kt"], run["kA"], color="#eda100", label="k_AMPA")
        ax[1].plot(run["kt"], run["kG"], color="#1baf7a", label="k_GABA")
        ax[1].set_ylabel("mean k")
        ax[1].legend(fontsize=8, frameon=False)
        ph = phases(run)
        ax[2].scatter(run["cps"], np.zeros(len(ph)), c=[{"ictal": SZ, "recovery": "#eda100",
                                                          "interictal": "#2a78d6"}[x] for x in ph], s=6)
        ax[2].set_yticks([])
        ax[2].set_ylabel("checkpoint\nphase")
        T = trials.get((m["seed"], m["condition"]), [])
        if T:
            cp = np.array([int(r["checkpoint"]) for r in T])
            ok = np.array([int(r["target"]) == int(r["prediction"]) for r in T], float)
            acc = np.array([ok[cp == j].mean() if (cp == j).any() else np.nan for j in range(len(run["cps"]))])
            k = 10
            sm = np.convolve(np.nan_to_num(acc, nan=np.nanmean(acc)), np.ones(k) / k, mode="same")
            ax[3].plot(run["cps"], acc, ".", color="#9a9890", ms=3)
            ax[3].plot(run["cps"], sm, color="#2a78d6", lw=1.4, label=f"{k}-checkpoint mean")
            ax[3].axhline(0.5, color=MUTED, lw=0.8)
            ax[3].legend(fontsize=8, frameon=False)
        ax[3].set_ylabel("encoding accuracy\nper checkpoint (adapted)")
        ax[3].set_xlabel("native time (s)")
        for a in ax:
            _style(a)
            a.axvspan(0, g["settle_s"], color=GRID, alpha=0.5, lw=0)
            for s0, s1 in run["seizures"]:
                a.axvspan(s0, s1, color=SZ, alpha=0.3, lw=0)
        ax[0].set_title(f"A: seed {m['seed']}, {m['condition']}: native trajectory and probe estimates "
                        f"(checkpoint phase: red ictal, yellow recovery, blue interictal)", fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(plots / f"A_state_{path.stem}.png", dpi=110)
        plt.close(fig)


def fig_encoding(summ, plots):
    conds = order({r["condition"] for r in summ})
    xs = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(conds) + 3), 5.5))
    series = [("C_encode_adapted_counts", "adapted, counts", "-", "o"),
              ("C_encode_adapted_proportions", "adapted, proportions", "-", "s"),
              ("C_encode_adapted_total", "adapted, total count", ":", "^"),
              ("C_encode_fixed_counts", "fixed reference, counts", "--", "o"),
              ("C_encode_fixed_proportions", "fixed reference, proportions", "--", "s"),
              ("C_encode_thinned_counts", "adapted, counts thinned to a common spike budget", "-.", "d")]
    cols = ["#2a78d6", "#1baf7a", "#9a9890", "#eb6834", "#e87ba4", "#52514e"]
    for (k, lab, ls, mk), c in zip(series, cols):
        m = [_ms(summ, cc, k) for cc in conds]
        ax.errorbar(xs, [x[0] for x in m], yerr=[x[1] for x in m], fmt=mk, ls=ls, color=c, capsize=3, label=lab)
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(conds, rotation=25, ha="right")
    ax.set_ylabel("held-out balanced accuracy (mean ± SD over seeds)")
    ax.set_title("B: immediate encoding of the +/- left/right pattern", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(plots / "B_encoding.png", dpi=120)
    plt.close(fig)


def fig_noise(noise, plots):
    conds = order({r["condition"] for r in noise})
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for a, regime in zip(ax, ("adapted", "fixed")):
        for c in conds:
            R = [r for r in noise if r["condition"] == c and r["regime"] == regime and r["view"] == "counts"]
            if not R:
                continue
            etas = sorted({r["eta"] for r in R})
            m = [np.nanmean([r["BA"] for r in R if r["eta"] == e]) for e in etas]
            sd = [np.nanstd([r["BA"] for r in R if r["eta"] == e]) for e in etas]
            a.errorbar(etas, m, yerr=sd, fmt="o-", color=color(c), capsize=2, ms=4, label=c)
        a.axhline(0.5, color=MUTED, lw=0.8)
        a.set_xlabel("corruption eta")
        a.set_title(f"C: clean-trained {regime} readout (counts)", fontsize=10, loc="left")
        _style(a)
    ax[0].set_ylabel("balanced accuracy (mean ± SD over seeds)")
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "C_noise.png", dpi=120)
    plt.close(fig)


def fig_lags(lags, plots):
    conds = order({r["condition"] for r in lags})
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[3, 1]))
    for c in conds:
        R = [r for r in lags if r["condition"] == c and r["kind"] == "actual" and r["window"] == "20ms"]
        ds = sorted({r["lag"] for r in R})
        m = [np.nanmean([r["R2"] for r in R if r["lag"] == d]) for d in ds]
        ax[0].plot([d * 50 for d in ds if d > 0], m[1:], "o-", color=color(c), ms=4, label=c)
        ax[1].plot([conds.index(c)], [m[0]], "o", color=color(c), ms=7)
    for kind, ls in (("null_input", "--"), ("null_permute", ":")):
        R = [r for r in lags if r["kind"] == kind and r["lag"] > 0]
        ds = sorted({r["lag"] for r in R})
        ax[0].plot([d * 50 for d in ds], [np.nanmean([r["R2"] for r in R if r["lag"] == d]) for d in ds], ls,
                   color="#9a9890", label=kind.replace("_", " ") + " (all conditions)")
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[0].set_xlabel("delay d (ms; the past symbol occupied a whole 50 ms bin)")
    ax[0].set_ylabel("held-out predictive R² (mean over seeds)")
    ax[0].set_title("D: recovering past symbols from the current bin's last 20 ms", fontsize=10, loc="left")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].set_xticks(range(len(conds)))
    ax[1].set_xticklabels(conds, rotation=40, ha="right", fontsize=8)
    ax[1].set_title("d = 0 (current symbol)", fontsize=10, loc="left")
    for a in ax:
        _style(a)
    fig.tight_layout()
    fig.savefig(plots / "D_lags.png", dpi=120)
    plt.close(fig)


def fig_tradeoffs(summ, pareto, plots):
    conds = order({r["condition"] for r in summ})
    keys = (("C_encode_adapted_counts", "C_encode (BA)"), ("C_noise_adapted_counts", "C_noise (area)"),
            ("S_memory", "S_memory (sum R², d = 1-10)"))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    for a, (k, lab) in zip(ax, keys):
        front = []
        for c in conds:
            S = _ms(summ, c, "burden")
            V = _ms(summ, c, k)
            a.plot(S[2], V[2], ".", color=color(c), alpha=0.5)
            a.errorbar(S[0], V[0], xerr=S[1], yerr=V[1], fmt="o", color=color(c), ms=8, capsize=3, label=c)
            p = next((r for r in pareto if r["condition"] == c), None)
            if p and p.get(f"pareto_{k}"):
                front.append((S[0], V[0], c))
                a.annotate(f"P {p[f'freq_{k}']:.0%}", (S[0], V[0]), xytext=(5, 5), textcoords="offset points",
                           fontsize=7, color=INK)
        front.sort()
        if len(front) > 1:
            a.plot([f[0] for f in front], [f[1] for f in front], ":", color=MUTED, lw=1)
        a.set_xlabel("seizure-like burden S (fraction of time ictal)")
        a.set_ylabel(lab)
        a.set_title("E: upper-left is better; P = Pareto, bootstrap frequency", fontsize=9, loc="left")
        _style(a)
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "E_tradeoffs.png", dpi=120)
    plt.close(fig)


def fig_conditional(summ, plots):
    conds = order({r["condition"] for r in summ})
    xs = np.arange(len(conds))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for k, (p, c) in enumerate((("ictal", SZ), ("recovery", "#eda100"), ("interictal", "#2a78d6"))):
        m = [_ms(summ, cc, f"C_encode_adapted_counts_{p}") for cc in conds]
        ax[0].errorbar(xs + (k - 1) * 0.2, [x[0] for x in m], yerr=[x[1] for x in m], fmt="o", color=c,
                       capsize=3, label=p)
        w = [_ms(summ, cc, f"phase_{p}")[0] for cc in conds]
        ax[1].bar(xs, w, bottom=np.nansum([[_ms(summ, cc, f"phase_{pp}")[0] for cc in conds]
                                           for pp in ("ictal", "recovery", "interictal")[:k]], axis=0),
                  color=c, label=p)
    ax[0].axhline(0.5, color=MUTED, lw=0.8)
    ax[0].set_ylabel("encoding BA by checkpoint phase (adapted, counts)")
    ax[0].set_title("F: conditional performance", fontsize=10, loc="left")
    ax[1].set_ylabel("share of native time (uniform checkpoints)")
    ax[1].set_title("native phase occupancy", fontsize=10, loc="left")
    for a in ax:
        a.set_xticks(xs)
        a.set_xticklabels(conds, rotation=25, ha="right")
        a.legend(fontsize=8, frameon=False)
        _style(a)
    fig.tight_layout()
    fig.savefig(plots / "F_conditional.png", dpi=120)
    plt.close(fig)


def make_all(out_dir, summ, lags, noise, pareto, plots):
    plots = Path(plots)
    fig_state(out_dir, plots)
    fig_encoding(summ, plots)
    fig_noise(noise, plots)
    fig_lags(lags, plots)
    fig_tradeoffs(summ, pareto, plots)
    fig_conditional(summ, plots)
