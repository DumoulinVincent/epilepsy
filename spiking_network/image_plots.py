"""Figures for the image-recovery analysis (called by image_analysis.main).

  reconstructions.png -- true image, response count map, predicted probability
                         map and binary reconstruction (local adapted decoder,
                         primary readout) for the same images across
                         conditions. Examples are fixed by rule before looking
                         at scores: mask IDs 0 and 1, first post-settle
                         non-seizure trial, lowest seed.
  ba_vs_readout.png   -- balanced accuracy vs readout duration: local /
                         population x adapted / frozen, mean +- SD over seeds,
                         permuted-label null in grey; interictal trials.
  recall_spec.png     -- white recall and black specificity per condition, and
                         balanced accuracy vs E rate (per seed).
  cycle_<run>.png     -- E rate, HSP gains, seizures, and balanced accuracy per
                         summary window with the number of trials evaluated.
"""
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID_C, SZ = "#0b0b0b", "#52514e", "#e4e3de", "#d03b3b"
_DBS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]
COLORS = {"untreated": "#eb6834", "benzo (k=1.0)": "#1baf7a", "reference 40Hz": "#52514e",
          **{f"DBS {a}pA": c for a, c in zip(range(10, 101, 10), _DBS)}}


def _style(ax):
    ax.grid(axis="y", color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _order(conds):
    import conditions as C
    names = [c["name"] for c in C.IMAGE_CONDITIONS]
    return sorted(conds, key=lambda c: names.index(c) if c in names else 999)


def plot_reconstructions(per_run, g, path, example_masks=(0, 1)):
    T0 = g["primary_readout_ms"]
    seed = min(o["meta"]["seed"] for o in per_run.values())
    runs = {o["meta"]["condition"]: o for o in per_run.values() if o["meta"]["seed"] == seed}
    conds = _order(runs)
    key = (T0, "local", "adapted")
    rows = [(c, m) for m in example_masks for c in conds]
    fig, ax = plt.subplots(len(rows), 4, figsize=(9, 2.2 * len(rows)), squeeze=False)
    titles = ("true image", f"E counts ({T0} ms)", "P(white), local decoder", "reconstruction")
    for r, (c, m) in enumerate(rows):
        o = runs[c]
        tt = o["tt"]
        cand = np.flatnonzero(tt["post"] & ~tt["seizure_overlap"] & (tt["mask_id"] == m))
        if not cand.size or key not in o["res"]:
            for a in ax[r]:
                a.axis("off")
            continue
        j = cand[0]
        res = o["res"][key]
        panels = (tt["Y"][j], o["example_counts"][T0][j], res["S"][j],
                  (res["S"][j] >= res["thr"][j]).astype(float))
        for k, (a, img) in enumerate(zip(ax[r], panels)):
            a.imshow(np.asarray(img, float).reshape(10, 10), cmap="Greys_r" if k != 1 else "viridis",
                     vmin=0, vmax=1 if k != 1 else None, interpolation="nearest")
            a.set_xticks([])
            a.set_yticks([])
            if r == 0:
                a.set_title(titles[k], fontsize=9)
        ax[r, 0].set_ylabel(f"{c}\nmask {m}, trial {j}", fontsize=8)
        ax[r, 3].set_xlabel(f"BA {res['BA'][j]:.2f}", fontsize=8)
    fig.suptitle(f"Image recovery, seed {seed}: same images across conditions (examples fixed by rule)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _mean_sd(rows, **sel):
    v = [r["BA"] for r in rows if all(r[k] == x for k, x in sel.items())]
    return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)


def plot_ba_vs_readout(rows, g, path):
    conds = _order({r["condition"] for r in rows})
    Ts = list(g["readout_ms"])
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for i, model in enumerate(("local", "population")):
        for j, regime in enumerate(("adapted", "frozen")):
            a = ax[i, j]
            for c in conds:
                ms = [_mean_sd(rows, condition=c, T_ms=T, model=model, regime=regime, phase="interictal")
                      for T in Ts]
                m = np.array([x[0] for x in ms])
                sd = np.array([x[1] for x in ms])
                if np.all(np.isnan(m)):
                    continue
                a.fill_between(Ts, m - sd, m + sd, color=COLORS.get(c, MUTED), alpha=0.15, lw=0)
                a.plot(Ts, m, "o-", color=COLORS.get(c, MUTED), lw=1.6, ms=4, label=c)
            nul = [_mean_sd(rows, T_ms=T, model=model, regime="null0", phase="interictal")[0] for T in Ts]
            a.plot(Ts, nul, "--", color="#9a9890", lw=1.2, label="permuted labels (adapted)")
            a.axhline(0.5, color=MUTED, lw=0.8)
            a.set_title(f"{model} decoder, {regime}", fontsize=10, loc="left")
            a.set_xscale("log")
            a.set_xticks(Ts)
            a.set_xticklabels([str(T) for T in Ts])
            _style(a)
    for a in ax[:, 0]:
        a.set_ylabel("balanced accuracy\n(interictal trials, mean ± SD over seeds)")
    for a in ax[-1]:
        a.set_xlabel("readout duration T (ms, from 20 ms after onset)")
    ax[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_recall_spec(rows, desc, g, path):
    T0 = g["primary_readout_ms"]
    conds = _order({r["condition"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    xs = np.arange(len(conds))
    for a, key, lab in ((ax[0], "recall", "white recall"), (ax[1], "spec", "black specificity")):
        for k, (model, off) in enumerate((("local", -0.15), ("population", 0.15))):
            vals = [[r[key] for r in rows if r["condition"] == c and r["T_ms"] == T0 and r["model"] == model
                     and r["regime"] == "adapted" and r["phase"] == "interictal"] for c in conds]
            a.errorbar(xs + off, [np.mean(v) if v else np.nan for v in vals],
                       yerr=[np.std(v) if v else np.nan for v in vals], fmt="o",
                       color=("#2a78d6", "#eb6834")[k], capsize=3, label=f"{model}, adapted")
        a.set_xticks(xs)
        a.set_xticklabels(conds, rotation=30, ha="right", fontsize=8)
        a.set_title(f"{lab} ({T0} ms, interictal)", fontsize=10, loc="left")
        a.set_ylim(0, 1.02)
        _style(a)
    ax[0].legend(fontsize=8, frameon=False)
    for stem, d in desc.items():
        ba = [r["BA"] for r in rows if r["seed"] == d["seed"] and r["condition"] == d["condition"]
              and r["T_ms"] == T0 and r["model"] == "local" and r["regime"] == "adapted"
              and r["phase"] == "interictal"]
        if ba:
            ax[2].plot(d["E_rate"], ba[0], "o", color=COLORS.get(d["condition"], MUTED), ms=7)
    for c in conds:
        ax[2].plot([], [], "o", color=COLORS.get(c, MUTED), label=c)
    ax[2].set_xlabel(f"E rate in the response window (Hz, {T0} ms, interictal)")
    ax[2].set_ylabel("balanced accuracy (local, adapted)")
    ax[2].set_title("score vs E rate (diagnostic; one point per seed and condition)", fontsize=10, loc="left")
    ax[2].legend(fontsize=7, frameon=False)
    _style(ax[2])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cycle(o, g, path):
    from image_analysis import load_run
    run = load_run(o["path"])
    tt = o["tt"]
    T0 = g["primary_readout_ms"]
    E = run["E_counts"]
    nb = 1000 // g["sched_ms"]
    Tn = len(E) // nb
    rE = E[:Tn * nb].sum(1, dtype=np.int64).reshape(Tn, nb).sum(1) / E.shape[1]
    res = o["res"].get((T0, "local", "adapted"))
    W = g["summary_window_s"]
    edges = np.arange(g["settle_s"], Tn + 1e-9, W)
    mid, ba, n = [], [], []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        s = (tt["onset_s"] >= a_) & (tt["onset_s"] < b_) & ~np.isnan(res["BA"]) if res else None
        if s is not None and s.any():
            mid.append((a_ + b_) / 2)
            ba.append(res["BA"][s].mean())
            n.append(int(s.sum()))
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    ax[0].plot(np.arange(Tn) + 0.5, rE, color="#c4314b", lw=0.6)
    ax[0].set_yscale("symlog", linthresh=1)
    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel("E rate (Hz, 1 s)")
    ax[1].plot(run["t_k"], run["k_AMPA"], color="#eda100", label="k_AMPA")
    ax[1].plot(run["t_k"], run["k_GABA"], color="#1baf7a", label="k_GABA")
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_ylabel("mean k")
    ax[2].plot(mid, ba, "o-", color="#2a78d6", ms=3, lw=1)
    for x, y, k in zip(mid, ba, n):
        ax[2].annotate(str(k), (x, y), fontsize=5, color=MUTED, ha="center", va="bottom")
    ax[2].axhline(0.5, color=MUTED, lw=0.8)
    ax[2].set_ylabel(f"balanced accuracy\n(local adapted, {T0} ms; numbers = trials)")
    ax[2].set_xlabel("time (s)")
    for a in ax:
        _style(a)
        a.axvspan(0, g["settle_s"], color=GRID_C, alpha=0.5, lw=0)
        for s0, s1 in np.asarray(run["seizures"]).reshape(-1, 2):
            a.axvspan(s0, s1, color=SZ, alpha=0.3, lw=0)
    m = o["meta"]
    ax[0].set_title(f"seed {m['seed']}, {m['condition']}: {len(run['seizures'])} seizures (red), "
                    f"settling (grey), every trial scored out of sample", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_all(per_run, frozen, rows, desc, g, plot_dir):
    plot_dir = Path(plot_dir)
    plot_reconstructions(per_run, g, plot_dir / "reconstructions.png")
    plot_ba_vs_readout(rows, g, plot_dir / "ba_vs_readout.png")
    plot_recall_spec(rows, desc, g, plot_dir / "recall_spec.png")
    for stem, o in per_run.items():
        plot_cycle(o, g, plot_dir / f"cycle_{stem}.png")
