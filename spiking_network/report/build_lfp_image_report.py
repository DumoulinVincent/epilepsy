"""Build lfp_image_results.pdf: LFP power < 30 Hz across treatments (seizure
risk and alarm threshold) and image recovery (balanced accuracy, adapted
decoder = upper bound, frozen decoder = lower bound).

    python report/build_lfp_image_report.py

Reads results/lfp_tau250_15k/treatment/{p30_windows,threshold}.csv (untreated,
reference, DBS 60/100/300 pA; 15000 s), results/lfp_tau250_20k/treatment/
(benzo and DBS 20 pA; 20000 s), results/lfp_tau250_15k/ews/ews_kendall.csv,
results/image_random_c16/analysis/summary.csv and
results/image_c16_dbs300/analysis/summary.csv (DBS 300 pA, seed 1 only).
Figures go to report/figs/; needs pdflatex.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parent
V2 = REPORT.parent
RES = V2 / "results"
FIGS = REPORT / "figs"
NAME = "lfp_image_results"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UNT, REF, BZ = "untreated", "reference 40Hz", "benzo (k=1.0)"
LONG = (BZ, "DBS 20pA")                      # taken from the 20000 s runs
DBS_LEVELS = (20, 60, 100, 150, 200, 250, 300)
ORDER = [UNT, BZ] + [f"DBS {a}pA" for a in DBS_LEVELS] + [REF]
LABEL = {UNT: "untreated", BZ: "benzo", REF: "reference", **{f"DBS {a}pA": f"DBS {a}" for a in DBS_LEVELS}}
# DBS: one hue, light -> dark with amplitude
_DBS_RAMP = ("#9fdcb5", "#5cc392", "#1baf7a", "#11905f", "#008300", "#006400", "#004d1a")
COL = {UNT: "#eb6834", REF: "#52514e", BZ: "#2a78d6", **{f"DBS {a}pA": c for a, c in zip(DBS_LEVELS, _DBS_RAMP)}}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
SETTLE = 500.0
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def style(a):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)


# ------------------------------------------------------------------ LFP data
def lfp_data():
    w15 = pd.read_csv(RES / "lfp_tau250_15k/treatment/p30_windows.csv")
    w20 = pd.read_csv(RES / "lfp_tau250_20k/treatment/p30_windows.csv")
    t15 = pd.read_csv(RES / "lfp_tau250_15k/treatment/threshold.csv")
    t20 = pd.read_csv(RES / "lfp_tau250_20k/treatment/threshold.csv")
    W = pd.concat([w15[~w15.condition.isin(LONG)], w20[w20.condition.isin(LONG)]])
    T = pd.concat([t15[~t15.condition.isin(LONG)], t20[t20.condition.isin(LONG)]])
    run_s = {c: (20000.0 if c in LONG else 15000.0) for c in ORDER}
    return W, T, run_s


def lfp_summary(W, T, run_s):
    rows = []
    for c in [c for c in ORDER if c in set(W.condition)]:      # conditions without runs are skipped
        per = []
        for s in sorted(W.seed.unique()):
            x = W[(W.seed == s) & (W.condition == c) & (W.t > SETTLE) & (W.ictal == 0)]
            r = W[(W.seed == s) & (W.condition == REF) & (W.t > SETTLE) & (W.ictal == 0)]
            ons = T[(T.seed == s) & (T.condition == c)].onset.dropna()
            ons = ons[ons != ""].astype(float)
            per.append(dict(seed=s, rel=x.p30.median() / r.p30.median(), p30=x.p30.median(), rE=x.rE.median(),
                            n_sz=len(ons), first=ons.min() if len(ons) else np.nan,
                            sz_h=len(ons) / ((run_s[c] - SETTLE) / 3600)))
        rows.append(dict(cond=c, per=pd.DataFrame(per)))
    return rows


def fig_lfp_timecourse(W, seed, path):
    rows = [UNT, BZ, "DBS 20pA", "DBS 60pA", "DBS 100pA", "DBS 300pA"]
    fig, ax = plt.subplots(len(rows), 1, figsize=(6.6, 5.6), sharex=True, sharey=True)
    r = W[(W.seed == seed) & (W.condition == REF)]
    for a, c in zip(ax, rows):
        a.plot(r.t, r.p30_smoothed, color=COL[REF], lw=0.8, label="reference 40 Hz")
        x = W[(W.seed == seed) & (W.condition == c)].sort_values("t")
        y = np.where(x.ictal == 1, np.nan, x.p30_smoothed)
        a.plot(x.t, y, color=COL[c], lw=1.3, label=LABEL[c])
        # seizure onsets: first ictal window of each run of ictal windows
        ic = x.ictal.to_numpy()
        on = x.t.to_numpy()[(ic == 1) & (np.r_[0, ic[:-1]] == 0)]
        on = on[on > SETTLE]
        a.plot(on, np.zeros_like(on), "v", color=COL[c], ms=6, clip_on=False, zorder=4)
        a.axvspan(0, SETTLE, color=GRID, alpha=0.6, lw=0)
        a.set_ylim(0, 110)
        a.set_yticks([0, 50, 100])
        a.text(0.06, 0.97, LABEL[c], transform=a.transAxes, ha="left", va="top", color=INK, fontsize=8)
        style(a)
    fig.supylabel("LFP power 0–30 Hz (pA²)", fontsize=8, x=0.01)
    ax[-1].set_xlabel("time (s)")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.99, bottom=0.08, hspace=0.18)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_lfp_dose(S, path):
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.6))
    rng = np.random.default_rng(0)
    for i, r in enumerate(S):
        c, P = r["cond"], r["per"]
        for a, k in zip(ax, ("rel", "sz_h", "rE")):
            v = P[k].to_numpy()
            a.plot(i + rng.uniform(-0.13, 0.13, len(v)), v, "o", ms=3.5, color=COL[r["cond"]], alpha=0.85)
            a.plot([i - 0.3, i + 0.3], [np.mean(v)] * 2, color=COL[c], lw=1.8)
    for a, t in zip(ax, ("(a) LFP power / reference", "(b) seizures per hour", "(c) E rate (Hz)")):
        a.set_xticks(range(len(S)))
        a.set_xticklabels([LABEL[r["cond"]] for r in S], rotation=45, ha="right")
        a.set_title(t, loc="left")
        a.set_ylim(bottom=0)
        style(a)
    ax[0].axhline(1, color=MUTED, lw=0.8, ls="--")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- image data
def image_data():
    a = pd.read_csv(RES / "image_random_c16/analysis/summary.csv")
    b = pd.read_csv(RES / "image_c16_dbs300/analysis/summary.csv")
    if "DBS 300pA" not in set(a.condition):                   # older layout: DBS 300 pA in its own folder
        a = pd.concat([a, b[b.condition == "DBS 300pA"]])
    return a[a.model == "local"]


def fig_image(d, path):
    conds = [c for c in ORDER if c in set(d.condition)]
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    # (a) bounds at 250 ms, interictal, + seizure-overlapping trials
    x = d[(d.T_ms == 250)]
    for i, c in enumerate(conds):
        g = x[(x.condition == c) & (x.phase == "interictal")].groupby("regime").BA.mean()
        ax[0].plot([i - 0.14, i + 0.14], [g["adapted"], g["frozen"]], color=COL[c], lw=0.9)
        ax[0].plot(i - 0.14, g["adapted"], "o", color=COL[c], ms=4.5)
        ax[0].plot(i + 0.14, g["frozen"], "o", mfc="white", mec=COL[c], mew=1.3, ms=4.5)
        s = x[(x.condition == c) & (x.phase == "seizure_overlap") & (x.regime == "adapted")].BA
        if len(s):
            ax[0].plot(i, s.mean(), "x", color=COL[c], ms=5, mew=1.3)
    ax[0].set_xticks(range(len(conds)))
    ax[0].set_xticklabels([LABEL[c] for c in conds], rotation=45, ha="right")
    ax[0].set_title("(a) bounds at 250 ms", loc="left")
    ax[0].set_ylim(0.55, 0.73)
    ax[0].set_ylabel("balanced accuracy")
    # (b) BA vs readout duration: DBS 300 adapted vs frozen, reference
    y = d[d.phase == "interictal"]
    for c, reg, ls in ((REF, "adapted", "-"), ("DBS 300pA", "adapted", "-"), ("DBS 300pA", "frozen", "--"),
                       (UNT, "adapted", "-")):
        g = y[(y.condition == c) & (y.regime == reg)].groupby("T_ms").BA.mean()
        ax[1].plot(g.index, g.values, ls, marker="o", ms=3, color=COL[c], lw=1.4, label=f"{LABEL[c]}, {reg}")
    ax[1].set_xscale("log")
    ax[1].set_xticks([50, 100, 250, 500])
    ax[1].set_xticklabels(["50", "100", "250", "500"])
    ax[1].set_xlabel("readout duration (ms)")
    ax[1].set_ylabel("balanced accuracy")
    ax[1].set_title("(b) BA vs readout duration", loc="left")
    ax[1].legend(frameon=False, loc="upper left", fontsize=6.5)
    # (c) frozen recall vs specificity at 250 ms
    z = x[(x.phase == "interictal")]
    for c in conds:
        for reg, mk in (("adapted", "o"), ("frozen", "s")):
            g = z[(z.condition == c) & (z.regime == reg)][["recall", "spec"]].mean()
            ax[2].plot(g["spec"], g["recall"], mk, color=COL[c], ms=4.5 if mk == "o" else 4,
                       mfc=COL[c] if reg == "adapted" else "none", mew=1.2)
    ax[2].annotate("DBS 300\nfrozen", (0.307, 0.933), xytext=(0.38, 0.86), fontsize=7, color=MUTED,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax[2].set_xlabel("black specificity")
    ax[2].set_ylabel("white recall")
    ax[2].set_xlim(0.25, 0.85)
    ax[2].set_ylim(0.55, 1.0)
    ax[2].set_title("(c) recall vs specificity", loc="left")
    for a in ax:
        style(a)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def tex_num(v, f="{:.2f}"):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def main():
    FIGS.mkdir(exist_ok=True)
    W, T, run_s = lfp_data()
    S = lfp_summary(W, T, run_s)
    fig_lfp_timecourse(W, 1, FIGS / "lfp_timecourse_seed1.png")
    fig_lfp_dose(S, FIGS / "lfp_dose.png")
    D = image_data()
    fig_image(D, FIGS / "image_bounds.png")
    ews = pd.read_csv(RES / "lfp_tau250_15k/ews/ews_kendall.csv")
    ek = ews.groupby("condition")[["mean", "var", "cv", "ac1_100", "p_lo", "p_mid"]].mean()

    # ---- LFP table
    lt = []
    for r in S:
        c, P = r["cond"], r["per"]
        th = T[(T.condition == c) & (T.role.isin(["held out", "treatment"]))]
        fr = T[T.condition == c].groupby("seed").alarm_time_fraction.first().mean()
        caught = f"{int(th.alarm_on.astype(str).eq('True').sum())}/{len(th)}" if len(th) else "--"
        lead = th[th.alarm_on.astype(str) == "True"].alarm_on_for_s.astype(float)
        lt.append(" & ".join([
            LABEL[c], f"{run_s[c]:.0f}", f"{int(P.n_sz.sum())}",
            tex_num(np.nanmean(P['first']), "{:.0f}") if P.n_sz.sum() else "--",
            f"{P.rel.mean():.2f}", f"{P.rE.mean():.2f}",
            f"{fr:.2f}" if c != UNT else f"{fr:.2f}$^a$", caught,
            tex_num(lead.median() / 60, "{:.0f}") if len(lead) else "--"]) + r" \\")
    # ---- image table
    it = []
    x = D[(D.T_ms == 250)]
    for c in [c for c in ORDER if c in set(D.condition)]:
        g = x[(x.condition == c) & (x.phase == "interictal")].groupby("regime")[["BA", "recall", "spec", "AUROC"]]
        m, sd = g.mean(), g.std()
        n = x[(x.condition == c) & (x.phase == "interictal") & (x.regime == "adapted")].seed.nunique()
        s = x[(x.condition == c) & (x.phase == "seizure_overlap") & (x.regime == "adapted")].BA
        sdtxt = lambda k: "" if n < 2 else f" $\\pm$ {sd.loc[k, 'BA']:.3f}"  # noqa: E731
        it.append(" & ".join([
            LABEL[c], str(n), f"{m.loc['adapted', 'BA']:.3f}{sdtxt('adapted')}",
            f"{m.loc['frozen', 'BA']:.3f}{sdtxt('frozen')}",
            f"{m.loc['adapted', 'AUROC']:.3f}",
            f"{m.loc['frozen', 'recall']:.2f} / {m.loc['frozen', 'spec']:.2f}",
            f"{s.mean():.3f}" if len(s) else "--"]) + r" \\")
    s_unt = next(r for r in S if r["cond"] == UNT)["per"]
    s_d20 = next(r for r in S if r["cond"] == "DBS 20pA")["per"]
    g300 = x[(x.condition == "DBS 300pA") & (x.phase == "interictal")].groupby("regime").BA.mean()
    rel = {r["cond"]: r["per"].rel.mean() for r in S}
    ews_u = ek.loc[UNT]
    ews_max = float(np.abs(ews_u).max())

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=2.0cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,enumitem,hyperref,float,xcolor,caption}
\captionsetup{font=small,labelfont=bf}
\setlist{itemsep=1pt,topsep=3pt}
\title{HSP seizure model: LFP power and image recovery across treatments}
\date{\today}
\begin{document}
\maketitle

\section*{Model and conditions}
Spiking network of 100 excitatory (E) and 25 inhibitory (I) neurons (Brian2) with
homeostatic synaptic plasticity (HSP): per-cell gains $k_\text{AMPA}$, $k_\text{GABA}\in[0,1]$ move toward a
target firing range (1--3\,Hz) with insertion/internalisation time constants $\tau$.
Five network seeds. Conditions: \textbf{untreated} (36\,Hz Poisson drive, seizure-prone),
\textbf{reference} (40\,Hz drive, healthy, never seizes), \textbf{benzo} ($k_\text{GABA}$ fixed at 1) and
\textbf{DBS} (120\,Hz Poisson current to every E cell, 20--300\,pA, on the 36\,Hz drive).
A seizure is an E rate above 10\,Hz in 1\,s bins.

\section{LFP power below 30\,Hz}
\textbf{Setup.} LFP proxy $=|\overline{I}_\text{AMPA}|+|\overline{I}_\text{GABA}|$ onto E cells, every 1\,ms.
HSP $\tau=250$\,s; runs of 15\,000\,s (benzo and DBS 20\,pA: 20\,000\,s). Power at 0--30\,Hz
(Welch, 4\,s segments) in 50\,s windows, smoothed with a trailing 250\,s mean (past only);
windows containing a seizure are left out.

\begin{figure}[tbp]\centering
\includegraphics[width=0.9\linewidth]{figs/lfp_timecourse_seed1.png}
\caption{Seed 1. LFP power 0--30\,Hz over the run, one row per condition (colour), each against the
40\,Hz reference (grey). Triangles: seizure onsets; shaded: settling period (500\,s). Untreated, reference
and DBS 60--300\,pA ran for 15\,000\,s, benzo and DBS 20\,pA for 20\,000\,s.}
\end{figure}

\begin{figure}[tbp]\centering
\includegraphics[width=\linewidth]{figs/lfp_dose.png}
\caption{Per condition, one dot per seed, bar = mean. (a) Seizure-free median LFP power 0--30\,Hz relative
to the same seed's reference (dashed = 1). (b) Seizure rate. (c) Seizure-free median E rate.}
\end{figure}

\textbf{Results.}
\begin{itemize}
\item \textbf{Untreated: slow build-up, abrupt seizure, deep collapse.} Between seizures, power rises over
thousands of seconds as $k_\text{AMPA}$ climbs to its ceiling; seizures (about 1\,min long) start when
$k_\text{AMPA}\approx0.95$, followed by a post-ictal trough of about 1500\,s. First seizure at
""" + f"{s_unt['first'].mean():.0f}" + r"""\,s on average; """ + f"{int(s_unt.n_sz.sum())}" + r""" seizures in 5 seeds.
\item \textbf{DBS lowers LFP power with dose.} Seizure-free power relative to the reference:
DBS 20 """ + f"{rel['DBS 20pA']:.2f}" + r""", \textbf{DBS 60 """ + f"{rel['DBS 60pA']:.2f}" + r"""}
(almost identical to the reference), DBS 100 """ + f"{rel['DBS 100pA']:.2f}" + r""",
DBS 300 """ + f"{rel['DBS 300pA']:.2f}" + r""". From 60\,pA there are no seizures, and the E rate
sits at (60\,pA) or above (100, 300\,pA) the reference.
\item \textbf{DBS 20\,pA delays seizures but does not stop them:} every seed eventually seizes, first at
""" + f"{s_d20['first'].mean():.0f}" + r"""\,s on average (untreated: """ + f"{s_unt['first'].mean():.0f}" + r"""\,s), by the same mechanism
($k_\text{AMPA}\approx0.95$).
\item \textbf{Benzo: highest power, no seizures} in $5\times20\,000$\,s. With $k_\text{GABA}=1$,
$k_\text{AMPA}$ saturates at 0.999 without destabilising the network. The proxy includes the
inhibitory current, so high power under benzo does not mean high risk.
\item \textbf{No short-horizon early warning.} Over the last 10\,min before each untreated onset
(12 seizures, 10\,s windows), the Kendall $\tau$ of LFP mean, variance, CV, lag-1 autocorrelation
and band power against time is $|\tau|\le""" + f"{ews_max:.2f}" + r"""$ on average, the same as the paired
reference: the onset is a noise-triggered jump from a long, fragile plateau.
\end{itemize}

\textbf{Alarm threshold (per seed).} Threshold $=0.95\times$ the lowest mean power over the last
500\,s before onset of that seed's first two untreated seizures. Alarm on when the smoothed power exceeds it.

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrrrr}
\toprule
condition & run (s) & seizures & 1st onset (s) & power/ref & E (Hz) & time in alarm & caught & alarm lead (min)\\
\midrule
""" + "\n".join(lt) + r"""
\bottomrule
\end{tabular}
\caption{LFP and alarm results, 5 seeds. ``Caught'': held-out untreated seizures (after the two used for
calibration) or seizures under treatment that started with the alarm on. Lead: how long the alarm had been on
at onset (median). $^a$Calibration seizures included in the time-in-alarm share.}
\end{table}

The threshold catches every held-out seizure, but as a risk-state marker (alarm on for tens of minutes to
hours before onset), not as a countdown. It does not transfer across treatments: benzo is in alarm most of
the time without seizures, and under DBS 20 the alarm is on for hours before onset. It would have to be
recalibrated after any treatment change.

\section{Image recovery: balanced accuracy, upper and lower bounds}
\textbf{Setup.} The drive carries a binary image: 10 of the 100 E cells (white) get $f+0.9a$, the others
$f-0.1a$ ($f=36$\,Hz, contrast $a=16$\,Hz; total input unchanged); 600\,ms images separated by
200--600\,ms blanks, 8000\,s per run, HSP $\tau=25$\,s, about 7500 images per run. Each pixel is decoded from
its own cell's spike count in a window starting 20\,ms after onset (logistic regression, one shared
threshold). Score: balanced accuracy BA $=($white recall $+$ black specificity$)/2$, chance $=0.5$.
\begin{itemize}
\item \textbf{Upper bound, adapted decoder:} retrained within each condition (blocked cross-validation).
What a downstream reader could extract after full recalibration.
\item \textbf{Lower bound, frozen decoder:} trained once on the healthy 40\,Hz reference of the same seed,
then applied unchanged. What a reader tuned to the healthy state gets without recalibration.
\end{itemize}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{figs/image_bounds.png}
\caption{(a) Interictal BA at 250\,ms per condition: filled = adapted decoder (upper bound), open = frozen
decoder (lower bound); the two are equal except at DBS 300\,pA. $\times$ = adapted BA on trials overlapping a seizure.
(b) Interictal BA against readout duration. (c) White recall against black specificity at 250\,ms;
filled = adapted, open = frozen. Mean over seeds; DBS 300\,pA: seed 1 only.}
\end{figure}

\begin{table}[H]\centering\small
\begin{tabular}{lrrrrrr}
\toprule
condition & seeds & BA adapted (upper) & BA frozen (lower) & AUROC & frozen recall / spec & BA during seizures\\
\midrule
""" + "\n".join(it) + r"""
\bottomrule
\end{tabular}
\caption{Readout 250\,ms, interictal trials, mean $\pm$ SD over seeds. AUROC: threshold-free separation of
white from black pixels (adapted). Only untreated and DBS 20\,pA had seizures during the image runs.}
\end{table}

\textbf{Results.}
\begin{itemize}
\item \textbf{Between seizures, recovery is about the same in every condition} (BA 0.69--0.70 at 250\,ms;
all conditions within 0.01). Longer readouts help every condition (about 0.55 at 50\,ms to 0.73--0.78 at 500\,ms).
\item \textbf{The deficit is ictal:} trials overlapping a seizure drop to BA about 0.60. Treatments help mainly by
preventing seizures.
\item \textbf{Upper and lower bounds coincide up to DBS 100\,pA} (identical per seed; reference within
$\pm$0.002, because its frozen decoder is scored on held-out trials only). With 0--2 spikes per cell in 250\,ms, both
decoders end up with the same integer rule (``$\ge1$ spike = white''), so they make the same calls: a reader tuned
to the healthy network needs no recalibration. The conditions differ in the kind of error, not the score: higher drive (DBS) trades
black specificity for white recall.
\item \textbf{At DBS 300\,pA the bounds separate:} adapted BA """ + f"{g300['adapted']:.3f}" + r""" (information intact,
highest AUROC) vs frozen """ + f"{g300['frozen']:.3f}" + r""". The raised background firing floods a
healthy-tuned reader with false whites (specificity 0.31). This is a calibration cost, not an information loss
(one seed; to be confirmed on 5 seeds and intermediate doses, 150--250\,pA).
\end{itemize}

\section*{Summary}
\begin{itemize}
\item LFP power below 30\,Hz tracks the hidden HSP state: it builds up for tens of minutes before untreated
seizures, and a per-seed threshold flags the high-risk state, but only without a change of treatment.
\item DBS lowers this power with dose; at 60\,pA it is close to the healthy reference and seizures stop.
DBS 20\,pA only delays them. Benzo prevents seizures while raising power.
\item Image recovery is preserved between seizures in every condition and lost during seizures. Up to
100\,pA, DBS costs nothing even for a reader that is not recalibrated. At 300\,pA, the information is intact
but a fixed reader is miscalibrated: a possible therapeutic window between about 60 and 300\,pA.
\end{itemize}
\end{document}
"""
    (REPORT / f"{NAME}.tex").write_text(tex)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{NAME}.tex"], cwd=REPORT,
                           capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:])
        raise SystemExit("pdflatex failed")
    print(f"wrote {REPORT / (NAME + '.pdf')}")


if __name__ == "__main__":
    main()
