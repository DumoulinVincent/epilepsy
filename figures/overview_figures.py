"""Overview figures shared with the AES 2026 poster, in the poster's style (Lato if installed, large fonts):
  network     -> out/network.pdf          the network schematic
  story       -> out/cycle_slow_fast.pdf  healthy vs deafferented with fast and slow HSP (long_tau_cycle.py runs)
  conditions  -> out/treatments_slow.pdf  E rate per condition, slow-HSP runs
  epi         -> out/epileptor.pdf        HSP mapped onto the Epileptor excitability x0
Called by make_paper_figures.py. Data: ../spiking_network/results/ and ../epileptor/results/trajectories.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
EPI = HERE.parent                      # repository root (this folder: figures/)
V2 = EPI / "spiking_network"
RES = V2 / "results"
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "report"))
import lfp_tau_plot as lp  # noqa: E402
import build_lfp_image_report as R  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = HERE / "out"
NAMES = dict(F_network="network", F_story="cycle_slow_fast", F_conditions="treatments_slow", F_epi="epileptor")
LATO = Path("/usr/local/texlive/2026/texmf-dist/fonts/truetype/typoland/lato")
from matplotlib import font_manager  # noqa: E402
for _w in ("Regular", "Bold", "Italic", "Semibold"):
    if (LATO / f"Lato-{_w}.ttf").exists():
        font_manager.fontManager.addfont(str(LATO / f"Lato-{_w}.ttf"))
UNT, REF, BZ = R.UNT, R.REF, R.BZ
C_UNT, C_REF, C_BZ, C_DBS, C_FROZ = "#eb6834", "#52514e", "#2a78d6", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#e4e3de", "#e34948"
WIN = "#d9f2e6"            # therapeutic window band
plt.rcParams.update({"font.family": "Lato", "mathtext.fontset": "custom", "mathtext.rm": "Lato",
                     "mathtext.it": "Lato:italic", "mathtext.bf": "Lato:bold", "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 11.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "lines.linewidth": 2.0, "axes.titleweight": "bold",
                     "pdf.fonttype": 42})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    name = NAMES.get(name, name)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def dose(c):
    return 0.0 if c == UNT else float(c[4:-2]) if c.startswith("DBS") else np.nan


# ================================================================ cycle (slow HSP)
LONG = RES / "long_tau" / "raw"


def episodes(t, r, thr=10.0, merge_s=None):
    """(onset, end) of runs of 1 s bins above thr; runs closer than merge_s are one episode."""
    hi = r > thr
    on = np.where(hi & ~np.r_[False, hi[:-1]])[0]
    off = np.where(hi & ~np.r_[hi[1:], False])[0]
    ep = []
    for a, b in zip(on, off):
        if ep and merge_s and t[a] - ep[-1][1] < merge_s:
            ep[-1][1] = t[b]
        else:
            ep.append([t[a], t[b]])
    return np.array(ep).reshape(-1, 2)


def load_long(name):
    d = np.load(LONG / name)
    return dict(t=d["rate_t"], rE=d["rate_E"], tk=d["t_k"], kA=d["k_AMPA"], kG=d["k_GABA"],
                meta=json.loads(str(d["meta"])))


def load_run(tau, cond, seed=1, run_s=None, slow=None, tau_hsp=None):
    """tau 250 s: the 15000 s LFP runs; otherwise long_tau_cycle.py output (slow: tau_int_AMPA =
    tau_ins_GABA; tau_hsp: rate-sensor time constant)."""
    if tau == 250 and not slow and not tau_hsp:
        d = np.load(RES / f"lfp_tau250_15k/raw/seed{seed}_{'untreated' if cond == UNT else 'reference_40Hz'}.npz")
        return dict(t=d["rate_t"], rE=d["rate_E"], tk=d["t_k"], kA=d["k_AMPA"], kG=d["k_GABA"])
    tag = f"tau{tau:g}" + (f"_intA{slow:g}_insG{slow:g}" if slow else "") + (f"_tauhsp{tau_hsp:g}" if tau_hsp else "")
    return load_long(f"{tag}_seed{seed}_{'untreated' if cond == UNT else 'reference_40Hz'}_{run_s:g}s.npz")


def periods():
    """Cycle period (onset to onset) per run, against the HSP time constant: tau 250 s from the LFP runs,
    tau >= 1000 s from long_tau_cycle.py (all four taus equal)."""
    rows = []
    for s in range(1, 6):
        f = RES / f"lfp_tau250_15k/raw/seed{s}_untreated.npz"
        d = np.load(f)
        t = np.arange(len(d["rate_E"])) + 0.5
        ep = episodes(t, d["rate_E"], merge_s=300)
        rows += [dict(tau=250, seed=s, period=p) for p in np.diff(ep[:, 0])]
    for f in sorted(LONG.glob("tau*_seed*_untreated_*s.npz")):
        if "_intA" in f.name:
            continue
        r = load_long(f.name)
        ep = episodes(r["t"], r["rE"], merge_s=0.5 * r["meta"]["taus"]["tau_ins_AMPA_E"])
        rows += [dict(tau=r["meta"]["taus"]["tau_ins_AMPA_E"], seed=r["meta"]["seed"], period=p)
                 for p in np.diff(ep[:, 0])]
    return pd.DataFrame(rows)


def fig_cycle(tau=2500, run_s=180000, slow=None, seed=1):
    u = load_run(tau, UNT, seed, run_s, slow)
    ref = load_run(tau, REF, seed, run_s)
    run_s = len(u["rE"])
    P = periods()
    fig = plt.figure(figsize=(8.4, 6.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.85, 1.05], width_ratios=[1.25, 1], hspace=0.62, wspace=0.38)
    a0 = fig.add_subplot(gs[0, :]); a1 = fig.add_subplot(gs[1, :], sharex=a0)
    a2 = fig.add_subplot(gs[2, 0]); a3 = fig.add_subplot(gs[2, 1])
    h = 3600.0
    # 10 s means keep the plot light; the episodes are marked from the 1 s bins
    for r, col, lab in ((ref, C_REF, "healthy input (40 Hz)"), (u, C_UNT, "deafferented (36 Hz)")):
        n = len(r["rE"]) // 10 * 10
        a0.plot(r["t"][:n].reshape(-1, 10).mean(1) / h, np.maximum(r["rE"][:n].reshape(-1, 10).mean(1), 0.05),
                color=col, lw=1.0, label=lab)
    ep = episodes(u["t"], u["rE"], merge_s=0.5 * tau)
    for a_, b_ in ep:
        for a in (a0, a1):
            a.axvspan(a_ / h, max(b_, a_ + 0.004 * run_s) / h, color=RED, alpha=0.18, lw=0)
    a0.set_yscale("log"); a0.set_ylim(0.3, 300)
    a0.set_ylabel("firing rate (Hz)")
    a0.legend(frameon=False, loc="upper left", ncol=2)
    a0.set_title("A  Deafferentation + HSP: recurring high-risk periods", loc="left")
    a1.plot(u["tk"] / h, u["kA"], color=C_UNT, lw=2.2, label="excitation $k_E$")
    a1.plot(u["tk"] / h, u["kG"], color=C_BZ, lw=2.2, label="inhibition $k_I$")
    a1.set_ylim(-0.03, 1.05); a1.set_ylabel("synaptic\nstrength")
    a1.set_xlabel("time (hours, model)")
    a1.legend(frameon=False, loc="center right", fontsize=11)
    a1.set_title("B  Slow homeostatic gains (36 Hz)", loc="left")
    # C: one cycle zoom on the episode (1 s bins)
    if len(ep):
        a_, b_ = ep[min(1, len(ep) - 1)]
        k = (u["t"] > a_ - 0.15 * tau) & (u["t"] < b_ + 0.6 * tau)
        a2.plot((u["t"][k] - a_) / 60, np.maximum(u["rE"][k], 0.05), color=C_UNT, lw=0.8)
        a2.set_yscale("log"); a2.set_ylim(0.3, 400)
        a2.set_xlabel("min from onset"); a2.set_ylabel("firing rate (Hz)")
        a2.set_title("C  One high-risk period", loc="left")
    # D: period vs tau
    g = P.groupby("tau").period
    taus = np.array(sorted(P.tau.unique()), float)
    m = g.mean().reindex(taus).values / h; sd = g.std().reindex(taus).fillna(0).values / h
    a3.errorbar(taus, m, yerr=sd, color=C_UNT, marker="o", ms=8, capsize=4, ls="none")
    slope = np.sum(taus * m) / np.sum(taus * taus)
    xx = np.array([0, taus.max() * 1.1])
    a3.plot(xx, slope * xx, color=MUTED, lw=1.2, ls="--")
    a3.set_xlabel("HSP time constant τ (s)"); a3.set_ylabel("cycle period (h)")
    a3.set_xlim(0, xx[-1]); a3.set_ylim(bottom=0)
    a3.text(0.05, 0.95, f"period ≈ {slope * h:.0f} τ", transform=a3.transAxes, va="top", fontsize=12, color=INK)
    a3.set_title("D  Period scales with τ", loc="left")
    save(fig, "F_cycle")
    print(P.groupby("tau").period.describe())
    return slope * h


# ================================================================ story: one picture, slower HSP -> wider high-risk periods
STORY = dict(  # file in results/long_tau/raw
    healthy="tau250_intA5000_insG5000_tauhsp1000_seed1_reference_40Hz_20000s.npz",
    fast="tau250_seed1_untreated_20000s.npz",
    slow="tau250_intA5000_insG5000_tauhsp1000_seed1_untreated_20000s.npz",
)
C_FAST = "#9a9890"


def fig_story(smooth_s=60, show_fast=True):
    """One axis: healthy (40 Hz) and deafferented (36 Hz) firing with slow HSP (+ fast HSP for comparison)."""
    h = 3600.0
    R_ = {k: load_long(f) for k, f in STORY.items()}

    def trace(r):
        n = len(r["rE"]) // smooth_s * smooth_s
        return (r["t"][:n].reshape(-1, smooth_s).mean(1) / h,
                np.maximum(r["rE"][:n].reshape(-1, smooth_s).mean(1), 0.5))

    fig, (a, b) = plt.subplots(2, 1, figsize=(5.57, 4.45), sharex=True, gridspec_kw=dict(height_ratios=[1.45, 1]))
    ep = episodes(R_["slow"]["t"], R_["slow"]["rE"], merge_s=600)
    for a_, b_ in ep:
        a.axvspan(a_ / h, b_ / h, color=RED, alpha=0.13, lw=0)
        b.axvspan(a_ / h, b_ / h, color=RED, alpha=0.13, lw=0)
    if show_fast:
        b.plot(R_["fast"]["tk"] / h, R_["fast"]["kA"], color=C_FAST, lw=1.2)
    b.plot(R_["healthy"]["tk"] / h, R_["healthy"]["kA"], color=C_REF, lw=2.2)
    b.plot(R_["slow"]["tk"] / h, R_["slow"]["kA"], color=C_UNT, lw=2.6)
    b.set_ylim(0, 1.05); b.set_yticks([0, 0.5, 1]); b.set_ylabel("excitation $k_E$")
    b.grid(axis="x", visible=False)
    if show_fast:
        a.plot(*trace(R_["fast"]), color=C_FAST, lw=1.2, label="deafferented (36 Hz), fast HSP")
    a.plot(*trace(R_["healthy"]), color=C_REF, lw=2.2, label="healthy input (40 Hz)")
    a.plot(*trace(R_["slow"]), color=C_UNT, lw=2.6, label="deafferented (36 Hz), slow HSP")
    a.set_yscale("log"); a.set_ylim(0.5, 1500); a.set_yticks([1, 10, 100]); a.set_yticklabels(["1", "10", "100"])
    a.set_xlim(0, R_["slow"]["t"][-1] / h)
    b.set_xlabel("time (hours, model)"); a.set_ylabel("firing rate (Hz)")
    a.grid(axis="x", visible=False)
    if len(ep) >= 2:
        (a0, b0), (a1, _) = ep[0], ep[1]
        a.annotate("", (b0 / h, 25), (a1 / h, 25), arrowprops=dict(arrowstyle="<->", color=C_BZ, lw=1.8))
        a.text(0.5 * (b0 + a1) / h, 32, "low-risk period", ha="center", va="bottom", color=C_BZ, fontsize=13)
        a.text(0.5 * (a0 + b0) / h, 480, "high-risk\nperiod", ha="center", va="bottom", color=RED, fontsize=12,
               fontweight="bold")
        a.text(0.5 * (ep[1][0] + ep[1][1]) / h, 480, "high-risk\nperiod", ha="center", va="bottom", color=RED,
               fontsize=12, fontweight="bold")
    hs, ls = a.get_legend_handles_labels()
    order = [ls.index(k) for k in ("healthy input (40 Hz)", "deafferented (36 Hz), slow HSP",
                                     "deafferented (36 Hz), fast HSP") if k in ls]
    b.legend([hs[i] for i in order], [ls[i] for i in order], frameon=False, loc="upper center", fontsize=11.5,
             bbox_to_anchor=(0.5, -0.42), ncol=2)
    fig.tight_layout(h_pad=0.3)
    save(fig, "F_story")

# ================================================================ network schema
def fig_network():
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch
    fig, a = plt.subplots(figsize=(5.92, 5.74))
    a.set_aspect("equal"); a.axis("off")
    cE, cI = C_UNT, C_BZ
    # receptive areas of one E cell (E->E, radius 2) and one I cell (I->E, radius 2)
    e0, i0 = (3, 3), (6.5, 6.5)
    a.add_patch(Rectangle((e0[0] - 2.5, e0[1] - 2.5), 5, 5, fc=cE, alpha=0.13, ec=cE, lw=1.5, ls="--", zorder=0))
    a.add_patch(Rectangle((i0[0] - 2.5, i0[1] - 2.5), 5, 5, fc=cI, alpha=0.10, ec=cI, lw=1.5, ls="--", zorder=0))
    for x in range(10):
        for y in range(10):
            a.add_patch(Circle((x, y), 0.27, fc=cE, ec="white", lw=0.8, zorder=2))
    for x in range(5):
        for y in range(5):
            a.add_patch(Rectangle((2 * x + 0.5 - 0.2, 2 * y + 0.5 - 0.2), 0.4, 0.4, fc=cI, ec="white", lw=0.8, zorder=3))
    # connections from the two highlighted cells
    for x in range(e0[0] - 2, e0[0] + 3):
        for y in range(e0[1] - 2, e0[1] + 3):
            if (x, y) != e0:
                a.plot([e0[0], x], [e0[1], y], color=cE, lw=0.9, alpha=0.7, zorder=1)
    for x in range(int(i0[0] - 2), int(i0[0] + 3)):
        for y in range(int(i0[1] - 2), int(i0[1] + 3)):
            a.plot([i0[0], x], [i0[1], y], color=cI, lw=0.9, alpha=0.6, zorder=1)
    a.add_patch(Circle(e0, 0.34, fc=cE, ec=INK, lw=2, zorder=4))
    a.add_patch(Rectangle((i0[0] - 0.26, i0[1] - 0.26), 0.52, 0.52, fc=cI, ec=INK, lw=2, zorder=4))
    # input spike trains on the left
    rng = np.random.default_rng(3)
    for y in (1, 4, 7):
        ts = np.sort(rng.uniform(-4.6, -1.6, 7))
        for t in ts:
            a.plot([t, t], [y - 0.25, y + 0.25], color=INK, lw=1.4)
        a.add_patch(FancyArrowPatch((-1.4, y), (-0.45, y), arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.6))
    a.text(-3.0, 9.3, "random input\nto every E cell", ha="center", va="bottom", fontsize=15, fontweight="bold")
    a.text(-3.0, 9.15, "40 Hz healthy\n36 Hz deafferented", ha="center", va="top", fontsize=13, color=MUTED)
    # DBS above the grid
    a.add_patch(FancyBboxPatch((3.6, 10.0), 1.8, 0.55, boxstyle="round,pad=0.05", fc="#9a9890", ec="none"))
    for x in (0.5, 2.5, 4.5, 6.5, 8.5):
        a.add_patch(FancyArrowPatch((4.5, 10.0), (x, 9.45), arrowstyle="-|>", mutation_scale=12, color=C_DBS, lw=1.6))
    a.text(4.5, 10.8, "DBS: 120 Hz current pulses into all E cells", ha="center", va="bottom", fontsize=15,
           color=C_DBS, fontweight="bold")
    # labels below
    a.text(-0.9, -2.45, "E→E (AMPA) × $k_E$", ha="left", va="center", fontsize=14, color=cE, fontweight="bold")
    a.text(5.0, -2.45, "I→E (GABA) × $k_I$", ha="left", va="center", fontsize=14, color=cI, fontweight="bold")
    a.text(4.5, -3.45, "(also E→I and I→I connections)", ha="center", va="center", fontsize=12.5, color=MUTED)
    # legend
    a.add_patch(Circle((0.0, -1.3), 0.27, fc=cE, ec="white")); a.text(0.5, -1.3, "100 excitatory", va="center", fontsize=14)
    a.add_patch(Rectangle((5.4, -1.5), 0.4, 0.4, fc=cI, ec="white")); a.text(6.0, -1.3, "25 inhibitory", va="center", fontsize=14)
    a.set_xlim(-5.2, 10.4); a.set_ylim(-3.9, 11.6)
    save(fig, "F_network")


# ================================================================ conditions: E(t) per condition, dose summary
SLOW = "tau250_intA5000_insG5000_tauhsp1000"          # slow-HSP runs (long_tau_cycle.py, 20 000 s)
COND_SLUG = {REF: "reference_40Hz", UNT: "untreated", BZ: "benzo__k_1_0",
             **{f"DBS {a}pA": f"DBS_{a}pA" for a in (20, 40, 60, 100, 150, 200, 250, 300)}}
COND_ROWS = (  # condition, label, colour
    (REF, "healthy input (40 Hz)", C_REF),
    (UNT, "deafferented, untreated", C_UNT),
    (BZ, "benzodiazepine", C_BZ),
    ("DBS 20pA", "DBS 20 pA", "#5cc392"),
    ("DBS 60pA", "DBS 60 pA", C_DBS),
)


def cond_file(cond, seed, src):
    """Raw run of one (condition, seed): src "slow" = slow-HSP 20 000 s runs; "fast" = tau 250 s LFP runs."""
    if src == "slow":
        return LONG / f"{SLOW}_seed{seed}_{COND_SLUG[cond]}_20000s.npz"
    long_ = RES / "lfp_tau250_20k/raw" / f"seed{seed}_{COND_SLUG[cond]}.npz"
    return long_ if long_.exists() else RES / "lfp_tau250_15k/raw" / f"seed{seed}_{COND_SLUG[cond]}.npz"


def fig_conditions(src="slow", seed=1, smooth_s=60):
    """All conditions on one axis: E firing rate (60 s means) against time, one network."""
    h = 3600.0
    rows = [(c, lab, col, np.load(cond_file(c, seed, src))) for c, lab, col in COND_ROWS
            if cond_file(c, seed, src).exists()]
    tmax = max(len(d["rate_E"]) for *_, d in rows) / h
    fig, a = plt.subplots(figsize=(5.57, 3.7))
    lw = {REF: 4.0, "DBS 60pA": 2.0}
    for c, lab, col, d in rows:
        r = d["rate_E"]; t = np.arange(len(r)) + 0.5
        n = len(r) // smooth_s * smooth_s
        a.plot(t[:n].reshape(-1, smooth_s).mean(1) / h, np.maximum(r[:n].reshape(-1, smooth_s).mean(1), 0.5),
               color=col, lw=lw.get(c, 2.0), label=lab, alpha=0.55 if c == REF else 1.0)
        for a_, b_ in episodes(t, r, merge_s=600):
            if a_ > 300:
                a.axvspan(a_ / h, b_ / h, color=col, alpha=0.12, lw=0)
    a.set_yscale("log"); a.set_ylim(0.6, 1200); a.set_yticks([1, 10, 100]); a.set_yticklabels(["1", "10", "100"])
    a.set_xlim(0, tmax); a.grid(axis="x", visible=False)
    a.set_xlabel("time (hours, model)", fontsize=14); a.set_ylabel("firing rate (Hz)", fontsize=14)
    a.legend(frameon=False, ncol=3, fontsize=11.5, loc="upper left", columnspacing=0.8, handlelength=1.4)
    fig.tight_layout()
    save(fig, "F_conditions")


def dose_stats(src="slow", seeds=(1, 2, 3, 4, 5)):
    """Per condition and seed: high-risk episodes per hour and median firing between episodes (after 500 s)."""
    out = {}
    for c in COND_SLUG:
        per = []
        for s in seeds:
            f = cond_file(c, s, src)
            if not f.exists():
                continue
            r = np.load(f)["rate_E"]; t = np.arange(len(r)) + 0.5
            ep = episodes(t, r, merge_s=600); ep = ep[ep[:, 0] > 500]
            quiet = (t > 500) & (r < 10)
            per.append((len(ep) / ((len(r) - 500) / 3600), float(np.median(r[quiet]))))
        if per:
            out[c] = np.array(per)
    return out


def fig_dose(src="slow"):
    S = dose_stats(src)
    dbs = sorted([c for c in S if c.startswith("DBS")], key=dose)
    XB = 385.0
    fig, ax = plt.subplots(1, 2, figsize=(5.57, 2.61))
    for a, k, lab, ttl in ((ax[0], 0, "episodes per hour (model)", "High-risk periods"),
                           (ax[1], 1, "firing rate (Hz)", "Firing between episodes")):
        xs = [0.0] + [dose(c) for c in dbs]
        m = np.array([S[UNT][:, k].mean()] + [S[c][:, k].mean() for c in dbs])
        sd = np.array([S[UNT][:, k].std()] + [S[c][:, k].std() for c in dbs])
        a.errorbar(xs[1:], m[1:], yerr=sd[1:], color=C_DBS, marker="o", ms=8, capsize=3, label="DBS")
        a.plot(xs[:2], m[:2], color=C_DBS, lw=1.2, ls=":")
        a.errorbar([0], [m[0]], yerr=[sd[0]], color=C_UNT, marker="o", ms=9, capsize=3, ls="none", label="untreated")
        if REF in S:
            a.axhline(S[REF][:, k].mean(), color=C_REF, lw=1.6, label="healthy")
        if BZ in S:
            a.errorbar([XB], [S[BZ][:, k].mean()], yerr=[S[BZ][:, k].std()], color=C_BZ, marker="s", ms=9,
                       capsize=3, ls="none", label="benzo")
        a.set_xticks([0, 60, 150, 300, XB]); a.set_xticklabels(["0", "60", "150", "300", "BZD"])
        a.set_xlim(-20, XB + 30); a.set_ylim(bottom=0)
        a.set_xlabel("DBS amplitude (pA)", fontsize=14); a.set_ylabel(lab, fontsize=12.5)
        a.set_title(ttl, loc="left", fontsize=14)
    ax[0].legend(frameon=False, fontsize=11.5, loc="upper right")
    ax[1].annotate("60 pA ≈ healthy", (60, S["DBS 60pA"][:, 1].mean()), xytext=(95, 0.55), fontsize=12,
                   color="#11905f", arrowprops=dict(arrowstyle="-", color="#11905f", lw=1))
    fig.tight_layout(w_pad=1.5)
    save(fig, "F_dose")


# ================================================================ treatments
def fig_treat():
    W, T, run_s = R.lfp_data()
    S = {r["cond"]: r["per"] for r in R.lfp_summary(W, T, run_s)}
    dbs = sorted([c for c in S if c.startswith("DBS")], key=dose)
    XB = 350.0
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 6.4))
    ax = axs.ravel()
    panels = ((ax[0], "sz_h", "episodes per hour (model)", "A  High-risk episodes"),
              (ax[1], "rel", "power <30 Hz / healthy", "B  LFP power"),
              (ax[2], "rE", "firing rate (Hz)", "C  Firing rate"))
    for a, k, lab, ttl in panels:
        xs = [0.0] + [dose(c) for c in dbs]
        m = np.array([S[UNT][k].mean()] + [S[c][k].mean() for c in dbs])
        sd = np.array([S[UNT][k].std()] + [S[c][k].std() for c in dbs])
        a.errorbar(xs[1:], m[1:], yerr=sd[1:], color=C_DBS, marker="o", ms=7, capsize=3, label="DBS")
        a.plot(xs[:2], m[:2], color=C_DBS, lw=1.2, ls=":")
        a.errorbar([0], [m[0]], yerr=[sd[0]], color=C_UNT, marker="o", ms=8, capsize=3, ls="none", label="untreated")
        ref = S[REF][k]
        a.axhline(ref.mean(), color=C_REF, lw=1.5, label="healthy")
        bz = S[BZ][k]
        a.errorbar([XB], [bz.mean()], yerr=[bz.std()], color=C_BZ, marker="s", ms=8, capsize=3, ls="none",
                   label="benzodiazepine")
        a.set_ylabel(lab); a.set_title(ttl, loc="left"); a.set_ylim(bottom=0)
    # D: image recovery, upper (recalibrated reader) and lower (healthy-tuned reader) bound
    d = pd.read_csv(RES / "image_random_c16/analysis/summary.csv")
    d = d[(d.model == "local") & (d.T_ms == 250) & (d.phase == "interictal")]
    conds = sorted([c for c in d.condition.unique() if c.startswith("DBS")], key=dose)

    def ba(c, reg):
        return d[(d.condition == c) & (d.regime == reg)].groupby("seed").BA.mean().mean()

    xs = [0.0] + [dose(c) for c in conds]
    for reg, col, mk, lab in (("adapted", C_DBS, "o", "re-tuned reader"), ("frozen", C_FROZ, "s", "healthy-tuned reader")):
        ax[3].plot(xs, [ba(UNT, reg)] + [ba(c, reg) for c in conds], color=col, marker=mk, ms=7, label=lab)
    ax[3].axhline(ba(REF, "adapted"), color=C_REF, lw=1.5)
    ax[3].plot([XB], [ba(BZ, "adapted")], "s", color=C_BZ, ms=8)
    ax[3].set_ylim(0.6, 0.72); ax[3].set_ylabel("image recovery (bal. acc.)")
    ax[3].legend(frameon=False, loc="lower left", fontsize=11)
    ax[3].set_title("D  Input processing", loc="left")
    for a in ax:
        a.axvspan(60, 150, color=WIN, lw=0, zorder=0)
        a.set_xticks([0, 60, 150, 300, XB]); a.set_xticklabels(["0", "60", "150", "300", "BZD"])
        a.set_xlim(-20, XB + 25)
    for a in ax[2:]:
        a.set_xlabel("DBS amplitude (pA)")
    ax[0].legend(frameon=False, loc="upper right", fontsize=11)
    ax[0].text(105, ax[0].get_ylim()[1] * 0.5, "window", ha="center", color="#11905f", fontsize=11)
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, "F_treat")


# ================================================================ prediction
def fig_pred(seed=1):
    W, T, run_s = R.lfp_data()
    thr = T[(T.seed == seed)].threshold.iloc[0]
    x = W[(W.seed == seed) & (W.condition == UNT)].sort_values("t")
    y = np.where(x.ictal == 1, np.nan, x.p30_smoothed).astype(float)
    on = (y >= thr) & np.isfinite(y)
    fig = plt.figure(figsize=(8.4, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.25], wspace=0.36)
    a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    th = x.t.values / 3600
    a0.fill_between(th, 0, np.nanmax(y) * 1.08, where=on, color=RED, alpha=0.14, lw=0, label="alarm on")
    a0.plot(th, y, color=C_UNT, lw=2.4, label="LFP power <30 Hz")
    a0.axhline(thr, color=INK, ls="--", lw=1.4, label="threshold (2 episodes)")
    r = lp.load(RES / f"lfp_tau250_15k/raw/seed{seed}_untreated.npz")
    for i, (s0, _) in enumerate([s for s in r["seizures"] if s[0] > lp.SETTLE_S]):
        a0.axvline(s0 / 3600, color=RED, lw=2, label="episode onset" if i == 0 else None)
    a0.set_xlabel("time (hours, model)"); a0.set_ylabel("power (pA²)"); a0.set_ylim(0, np.nanmax(y) * 1.45)
    a0.legend(frameon=False, fontsize=10, loc="upper left", ncol=2, borderaxespad=0.1, columnspacing=1.0)
    # AUROC: last 10 min before onset vs mid-interictal, untreated, 5 seeds pooled
    E = pd.read_csv(RES / "lfp_tau250_15k/entropy/entropy_windows.csv")
    feats = (("p30", "power\n<30 Hz"), ("SpecEn", "spectral\nentropy"), ("PermEn", "perm.\nentropy"))
    aucs = []
    for k, _ in feats:
        pos, neg = [], []
        for s in sorted(E.seed.unique()):
            q = E[(E.seed == s) & (E.condition == UNT)].sort_values("t")
            t, v, ic = q.t.values, q[k].values, q.ictal.values
            sz = lp.load(RES / f"lfp_tau250_15k/raw/seed{s}_untreated.npz")["seizures"]
            ok = (t > lp.SETTLE_S) & (ic == 0) & np.isfinite(v)
            pre = np.zeros(len(t), bool); far = ok.copy()
            for s0, s1 in sz:
                pre |= (t >= s0 - 600) & (t < s0)
                far &= ~((t > s0 - 1800) & (t < s1 + 1800))
            pos += list(v[ok & pre]); neg += list(v[far])
        u = stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))
        aucs.append(max(u, 1 - u))
    a1.bar(range(3), aucs, color=[C_UNT, "#9a9890", "#9a9890"], width=0.62)
    for i, v in enumerate(aucs):
        a1.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=12)
    a1.axhline(0.5, color=MUTED, lw=1, ls=":")
    a1.set_xticks(range(3)); a1.set_xticklabels([f for _, f in feats], fontsize=10)
    a1.set_ylim(0.4, 1.0); a1.set_ylabel("AUROC")
    a1.set_title("B  High vs low risk", loc="left")
    a0.set_title("A  Alarm on LFP power (one network)", loc="left")
    save(fig, "F_pred")


# ================================================================ Epileptor
EPI_REPORT = EPI / "epileptor" / "report"


def epileptor_lfp(x0s, T_ms=10000.0, dt=0.05, sigma=0.02, seed=7):
    """Full Epileptor (TVB parameters) at fixed x0 with additive noise (sigma on x1, y1; sigma/4 on population 2):
    x1, x2 and the LFP-like x2 - x1, all from the same run. Cached in figs/epileptor_lfp.npz."""
    cache = FIG / "epileptor_lfp.npz"
    key = np.array([x0s[k] for k in sorted(x0s)] + [T_ms, dt, sigma, seed, 0.25, 6.0])
    if cache.exists():
        c = np.load(cache)
        if c["key"].shape == key.shape and np.allclose(c["key"], key):
            return {k: (c["t"], c[k + "_x1"], c[k + "_x2"], c[k + "_lfp"]) for k in x0s} | {k + "_x0": x0s[k] for k in x0s}
    sys.path.insert(0, str(EPI_REPORT))
    import theory_figures as TH
    n = int(T_ms / dt); every = 20
    out = {"t": np.arange(n // every) * dt * every / 1000.0}
    for i_, (name, x0) in enumerate(sorted(x0s.items())):
        rng = np.random.default_rng(seed + i_)
        (xf, zf, _, _), = TH.fixed_points(x0)
        y = np.array([xf, 1 - 5 * xf**2, zf, -0.9, 0.0, 0.1 * xf])     # full model
        p = dict(TH.P6, x0=x0)
        nz = rng.standard_normal((n, 5)) * sigma * np.sqrt(dt)
        X1 = np.empty(n // every); X2 = np.empty(n // every); L = np.empty(n // every)
        for k_ in range(n):
            y = y + np.array(TH.epi6d(0, y, p)) * dt
            y[0] += nz[k_, 1]; y[1] += nz[k_, 2]; y[3] += 0.25 * nz[k_, 3]; y[4] += 0.25 * nz[k_, 4]
            if k_ % every == 0:
                X1[k_ // every] = y[0]; X2[k_ // every] = y[3]; L[k_ // every] = y[3] - y[0]
        out[name + "_x1"] = X1; out[name + "_x2"] = X2; out[name + "_lfp"] = L
    np.savez_compressed(cache, key=key, **out)
    return {k: (out["t"], out[k + "_x1"], out[k + "_x2"], out[k + "_lfp"]) for k in x0s} | {k + "_x0": x0s[k] for k in x0s}


def fig_epi():
    Z = np.load(EPI / "epileptor" / "results" / "trajectories.npz")
    X0C, ZSN, IE = float(Z["X0_CRIT"]), float(Z["Z_SN"]), float(Z["I_EXT"])
    G = lambda x1: -x1**3 - 2 * x1**2 + 1 + IE  # noqa: E731
    seeds = [int(v) for v in Z["seeds"]]
    s = seeds[0]
    t, x0, ic = Z[f"s{s}|{UNT}|t"], Z[f"s{s}|{UNT}|x0"], Z[f"s{s}|{UNT}|ictal"]
    sz = Z[f"s{s}|{UNT}|seizures"]
    sz = sz[sz[:, 0] > lp.SETTLE_S]
    e0, on1 = sz[0, 1], sz[1, 0]
    snaps = [e0 + 300, e0 + 900, e0 + 1800, 0.5 * (e0 + on1) + 600, on1 - 60]
    cols = ["#2a78d6", "#5b9be0", "#9a9890", "#eda100", RED]
    x0r = float(np.median(Z[f"s{s}|{REF}|x0"][Z[f"s{s}|{REF}|t"] > lp.SETTLE_S]))

    fig = plt.figure(figsize=(11.2, 3.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.85, 1.05], wspace=0.3)
    a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[0, 1])
    m = t > lp.SETTLE_S
    a0.plot(t[m] / 3600, np.where(ic, np.nan, x0)[m], color=C_UNT, lw=2, label="deafferented (36 Hz)")
    tr = Z[f"s{s}|{REF}|t"]
    a0.plot(tr[tr > lp.SETTLE_S] / 3600, Z[f"s{s}|{REF}|x0"][tr > lp.SETTLE_S], color=C_REF, lw=2,
            label="healthy (40 Hz)")
    for a_, b_ in sz:
        a0.axvspan(a_ / 3600, b_ / 3600, color=RED, alpha=0.35, lw=0)
    for ts, col in zip(snaps, cols):
        k = int(np.argmin(np.abs(t - ts)))
        a0.plot(t[k] / 3600, x0[k], "o", color=col, ms=9, mec="white", mew=1, zorder=5)
    a0.axhline(X0C, color=INK, ls="--", lw=1.2)
    a0.text(0.2, X0C + 0.08, "seizure threshold $x_{0,crit}$", fontsize=11, va="bottom")
    a0.set_xlabel("time (hours, model)"); a0.set_ylabel("$x_0$ (from HSP)")
    a0.set_ylim(-4.0, -1.75)
    a0.legend(frameon=False, loc="center", bbox_to_anchor=(0.62, 0.42), ncol=1, fontsize=12)
    a0.set_title("A  HSP sets the excitability $x_0(t)$", loc="left")
    x1 = np.linspace(-2.7, -0.7, 300)
    att = x1 <= -4 / 3                                   # attracting (stable) part; the rest is the repelling branch
    a1.plot(x1[att], G(x1[att]), color=INK, lw=2.6, label="fast nullcline")
    a1.plot(x1[~att], G(x1[~att]), color="#c3c2bc", lw=2.4, ls=(0, (4, 2)))
    a1.text(-0.71, 3.45, "unstable", fontsize=9, color="#8f8e88", ha="right", va="bottom")
    for ts, col in zip(snaps, cols):
        k = int(np.argmin(np.abs(t - ts)))
        a1.plot(x1, 4 * (x1 - float(x0[k])), color=col, lw=2)
    a1.plot(x1, 4 * (x1 - x0r), color=C_REF, lw=2.2, ls="--", label="healthy (40 Hz)")
    xr = [q.real for q in np.roots([-1.0, -2.0, -4.0, 1 + IE + 4 * x0r]) if abs(q.imag) < 1e-9][0]
    a1.plot(xr, 4 * (xr - x0r), "o", ms=10, mfc=C_REF, mec="white", zorder=6)
    a1.plot(-4 / 3, ZSN, "o", color=RED, ms=10, zorder=6)
    a1.annotate("fold:\nseizure onset", (-4 / 3, ZSN), xytext=(-1.25, 2.4), fontsize=11,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a1.annotate("healthy:\nstable rest", (xr, 4 * (xr - x0r)), xytext=(-2.65, 2.4), fontsize=11, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a1.set_xlim(-2.7, -0.7); a1.set_ylim(2.0, 6.5)
    a1.set_xlabel("$x_1$ (fast activity)"); a1.set_ylabel("$z$ (slow permittivity)")
    a1.set_title("B  Phase plane", loc="left")
    traces = epileptor_lfp({"blue": float(x0[int(np.argmin(np.abs(t - snaps[0])))]), "healthy": x0r,
                            "red": float(x0[int(np.argmin(np.abs(t - snaps[-1])))])})
    sub = gs[0, 2].subgridspec(3, 1, hspace=0.62)
    rows = (("blue", cols[0], "after a high-rate period"), ("healthy", C_REF, "healthy"),
            ("red", RED, "onset"))
    for k_, (key, col, lab) in enumerate(rows):
        ax_ = fig.add_subplot(sub[k_])
        tt, x1f, x2f, _ = traces[key]
        ax_.plot(tt, x1f, color=col, lw=1.4)
        ax_.set_ylim(-2.6, 1.9); ax_.set_xlim(0, tt[-1]); ax_.set_yticks([])
        ax_.set_title(f"{lab}: $x_0$ = {traces[key + '_x0']:.2f}", fontsize=11.5, color=col, loc="left",
                      fontweight="bold", pad=1)
        ax_.grid(False)
        if k_ == 1:
            ax_.set_ylabel("$x_1$ (fast activity)", fontsize=13)
        if k_ < 2:
            ax_.set_xticklabels([])
        if k_ == 0:
            ax_.text(0.0, 1.62, "C  Epileptor activity $x_1(t)$", transform=ax_.transAxes, fontsize=14, fontweight="bold")
    ax_.set_xlabel("time (s, Epileptor units)")
    save(fig, "F_epi")


if __name__ == "__main__":
    which = sys.argv[1:] or ["network", "story", "conditions", "epi"]
    for w in which:
        if w.split(":")[0] in ("conditions", "dose"):    # conditions[:fast], dose[:fast]
            globals()[f"fig_{w.split(':')[0]}"](*(w.split(":")[1:] or ["slow"]))
        elif w.startswith("cycle"):            # cycle or cycle:<tau>:<run_s>[:<slow>]
            a = [float(v) for v in w.split(":")[1:]]
            fig_cycle(*a[:2], *(a[2:] or [None])) if a else fig_cycle()
        else:
            globals()[f"fig_{w}"]()
