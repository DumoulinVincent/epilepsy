"""LFP power < 30 Hz across treatments, and a per-patient (per-seed) alarm
threshold calibrated on the untreated seizures (lfp_cycle.py runs).

    python lfp_treatment.py results/lfp_tau250_15k

Feature: p30 = LFP power at 0-30 Hz (DC excluded; Welch, 4 s segments) in
non-overlapping 50 s windows, smoothed causally (trailing mean of the last
SMOOTH_W non-ictal windows = 250 s), so the alarm uses only the past.

Threshold, per seed: calibrated on the first N_CAL untreated seizures. For
each, the smoothed p30 averaged over the last PRE_S seconds before onset;
threshold = the lowest of those x MARGIN. Alarm = smoothed p30 >= threshold.
Evaluated on (i) the later untreated seizures of the same seed (held out) and
(ii) every treatment run: share of seizure-free time in alarm, seizures that
started during an alarm, and how long the alarm had been on at onset.

Outputs (<dir>/treatment/): p30_windows.csv, threshold.csv, and plots/
  p30_seed{N}.png -- one row per treatment: smoothed p30 vs time for the
      treatment, untreated and the 40 Hz reference; seed threshold dashed;
      seizure onsets marked.
  rates_seed{N}.png -- same layout, E and I firing rates (50 s means).
  rates_summary.png -- seizure-free median E and I rate per condition.
  p30_summary.png -- per condition: seizures per hour, median seizure-free
      p30 relative to the reference, share of time in alarm, alarm hits.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import lfp_tau_plot as lp  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WIN_S, SMOOTH_W, PRE_S, N_CAL, MARGIN = 50.0, 5, 500.0, 2, 0.95
UNT, REF = "untreated", "reference 40Hz"
FIXED = {UNT: "#eb6834", REF: "#52514e", "benzo (k=1.0)": "#2a78d6"}
DBS_RAMP = ["#9fdcb5", "#5cc392", "#1baf7a", "#11905f", "#008300", "#006400", "#004d1a"]   # light -> dark = weak -> strong


def colours(conds):
    dbs = sorted((c for c in conds if c.startswith("DBS")), key=lambda c: float(c[4:-2]))
    col = dict(FIXED)
    col.update({c: DBS_RAMP[min(i, len(DBS_RAMP) - 1)] for i, c in enumerate(dbs)})
    return col


def order(conds):
    dbs = sorted((c for c in conds if c.startswith("DBS")), key=lambda c: float(c[4:-2]))
    rest = sorted(c for c in conds if c not in dbs and c not in (UNT, REF))
    return [c for c in (UNT, REF) if c in conds] + rest + dbs


def file_features(p):
    """(seed, condition, windows, seizures) for one raw file; raw LFP dropped."""
    r = lp.load(p)
    seed = int(p.stem.split("_")[0][4:])
    cond = r["meta"]["condition"]
    cond = cond["name"] if isinstance(cond, dict) else str(cond)
    F = lp.window_features(r, WIN_S)
    t = np.array([w["t"] for w in F])
    p30 = np.array([w["p_lo"] + w["p_mid"] for w in F])
    ictal = np.array([w["ictal"] for w in F])
    sm = np.full(len(t), np.nan)
    hist = []
    for i in range(len(t)):
        if ictal[i]:
            hist = []                      # a seizure resets the trailing history
            continue
        hist = (hist + [p30[i]])[-SMOOTH_W:]
        sm[i] = np.mean(hist)
    n = int(WIN_S)
    rI = np.array([r["rI"][int(x - WIN_S / 2):int(x - WIN_S / 2) + n].mean() for x in t])
    return dict(seed=seed, cond=cond, t=t, p30=p30, sm=sm, ictal=ictal, rE=np.array([w["rE"] for w in F]), rI=rI,
                seizures=np.asarray(r["seizures"], float).reshape(-1, 2), run_s=len(r["rE"]))


def pre_level(run, onset):
    m = (run["t"] < onset) & (run["t"] >= onset - PRE_S) & np.isfinite(run["sm"])
    return float(np.mean(run["sm"][m])) if m.any() else np.nan


def alarm_stats(run, thr):
    """Share of seizure-free time in alarm; per seizure: alarm on at onset and for how long."""
    ok = (run["t"] > lp.SETTLE_S) & np.isfinite(run["sm"])
    on = ok & (run["sm"] >= thr)
    hits = []
    for a, _ in run["seizures"]:
        if a < lp.SETTLE_S + WIN_S:
            continue
        before = np.where(ok & (run["t"] < a))[0]
        if not len(before):
            continue
        last = before[-1]
        if not on[last]:
            hits.append((a, False, 0.0))
            continue
        j = last
        while j > 0 and on[j - 1] and ok[j - 1]:
            j -= 1
        hits.append((a, True, float(a - run["t"][j] + WIN_S / 2)))
    return float(on[ok].mean()) if ok.any() else np.nan, hits


def main(d):
    d = Path(d)
    files = sorted((d / "raw").glob("seed*.npz"))
    if not files:
        raise SystemExit(f"no runs in {d / 'raw'}")
    runs = Parallel(n_jobs=min(8, len(files)))(delayed(file_features)(p) for p in files)
    R = {(r["seed"], r["cond"]): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    conds = order({r["cond"] for r in runs})
    col = colours(conds)
    out = d / "treatment"
    (out / "plots").mkdir(parents=True, exist_ok=True)
    with open(out / "p30_windows.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "condition", "t", "p30", "p30_smoothed", "ictal", "rE"])
        for r in runs:
            for i in range(len(r["t"])):
                w.writerow([r["seed"], r["cond"], r["t"][i], f"{r['p30'][i]:.4g}", f"{r['sm'][i]:.4g}",
                            int(r["ictal"][i]), f"{r['rE'][i]:.4g}"])
    # ------------------------------------------------------------ thresholds
    thr, cal, rows = {}, {}, []
    for s in seeds:
        u = R.get((s, UNT))
        if u is None:
            continue
        usable = [a for a, _ in u["seizures"] if a - PRE_S >= lp.SETTLE_S]
        if len(usable) < N_CAL:
            print(f"seed {s}: only {len(usable)} untreated seizure(s); no threshold")
            continue
        levels = [pre_level(u, a) for a in usable[:N_CAL]]
        thr[s] = MARGIN * min(levels)
        cal[s] = set(usable[:N_CAL])
        for c in conds:
            r = R.get((s, c))
            if r is None:
                continue
            frac, hits = alarm_stats(r, thr[s])
            for a, hit, lead in hits:
                if c == UNT and a in cal[s]:
                    role = "calibration"
                else:
                    role = "held out" if c == UNT else "treatment"
                rows.append(dict(seed=s, condition=c, onset=round(a), role=role, alarm_on=hit,
                                 alarm_on_for_s=round(lead), threshold=round(thr[s], 2),
                                 alarm_time_fraction=round(frac, 3)))
            if not hits:
                rows.append(dict(seed=s, condition=c, onset="", role="no seizure", alarm_on="", alarm_on_for_s="",
                                 threshold=round(thr[s], 2), alarm_time_fraction=round(frac, 3)))
    if rows:
        with open(out / "threshold.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    # ------------------------------------------------------- per-seed figure
    treat = [c for c in conds if c not in (UNT, REF)] or [UNT]
    for s in seeds:
        fig, ax = plt.subplots(len(treat), 1, figsize=(15, 2.3 * len(treat) + 1), sharex=True, sharey=True,
                               squeeze=False)
        ax = ax[:, 0]
        for a, c in zip(ax, treat):
            for cc, lw, z in ((REF, 0.9, 1), (UNT, 0.9, 2), (c, 1.6, 3)):
                r = R.get((s, cc))
                if r is None or (cc == c and c == UNT and z == 3):
                    continue
                a.plot(r["t"], r["sm"], color=col[cc], lw=lw, zorder=z, label=cc)
                for on_ in [a_ for a_, _ in r["seizures"] if a_ >= lp.SETTLE_S]:
                    a.plot([on_], [0], marker="v", ms=8, color=col[cc], zorder=4, clip_on=False)
            if s in thr:
                a.axhline(thr[s], color=lp.INK, lw=0.8, ls="--")
            a.set_ylabel("p30, 250 s\ntrailing (pA²)", fontsize=9)
            a.set_title(c, fontsize=10, loc="left", color=lp.INK)
            a.legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
            lp.style(a)
            a.axvspan(0, lp.SETTLE_S, color=lp.GRID, alpha=0.5, lw=0)
            a.set_ylim(bottom=0)
        ax[-1].set_xlabel("time (s)")
        fig.suptitle(f"seed {s}: LFP power 0-30 Hz (50 s windows, trailing 250 s mean, seizure windows left out); "
                     f"triangles = seizure onsets; dashed = alarm threshold from the first {N_CAL} untreated seizures",
                     fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / "plots" / f"p30_seed{s}.png", dpi=110)
        plt.close(fig)
    # ------------------------------------------------------ firing rates
    jit_r = np.random.default_rng(1)
    for s in seeds:
        fig, ax = plt.subplots(len(treat), 2, figsize=(16, 2.3 * len(treat) + 1), sharex=True, squeeze=False)
        for i, c in enumerate(treat):
            for j, (k, lab) in enumerate((("rE", "E rate (Hz)"), ("rI", "I rate (Hz)"))):
                a = ax[i, j]
                for cc, lw, z in ((REF, 0.9, 1), (UNT, 0.9, 2), (c, 1.6, 3)):
                    r = R.get((s, cc))
                    if r is None or (c == UNT and z == 3):
                        continue
                    v = np.where(r["ictal"], np.nan, r[k])
                    a.plot(r["t"], v, color=col[cc], lw=lw, zorder=z, label=cc)
                    for on_ in [a_ for a_, _ in r["seizures"] if a_ >= lp.SETTLE_S]:
                        a.plot([on_], [0], marker="v", ms=8, color=col[cc], zorder=4, clip_on=False)
                a.set_ylim(bottom=0)
                a.set_ylabel(lab, fontsize=9)
                a.set_title(c, fontsize=10, loc="left", color=lp.INK)
                lp.style(a)
                a.axvspan(0, lp.SETTLE_S, color=lp.GRID, alpha=0.5, lw=0)
            ax[i, 1].legend(fontsize=8, frameon=False, loc="upper right", ncol=3)
        for a in ax[-1]:
            a.set_xlabel("time (s)")
        fig.suptitle(f"seed {s}: firing rates, 50 s window means (windows containing a seizure left out); "
                     f"triangles = seizure onsets", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / "plots" / f"rates_seed{s}.png", dpi=110)
        plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for i, c in enumerate(conds):
        for a, k in zip(ax, ("rE", "rI")):
            v = [np.median(R[(s, c)][k][(R[(s, c)]["t"] > lp.SETTLE_S) & ~R[(s, c)]["ictal"]])
                 for s in seeds if (s, c) in R]
            if v:
                a.plot(i + jit_r.uniform(-0.12, 0.12, len(v)), v, "o", color=col[c], ms=6, alpha=0.85)
                a.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=col[c], lw=2)
    for a, tl in zip(ax, ("E rate, seizure-free median (Hz)", "I rate, seizure-free median (Hz)")):
        a.set_xticks(range(len(conds)))
        a.set_xticklabels(conds, rotation=35, ha="right", fontsize=8)
        a.set_title(tl, fontsize=10, loc="left")
        a.set_ylim(bottom=0)
        lp.style(a)
    fig.tight_layout()
    fig.savefig(out / "plots" / "rates_summary.png", dpi=110)
    plt.close(fig)
    # -------------------------------------------------------- summary figure
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    jit = np.random.default_rng(0)
    for i, c in enumerate(conds):
        sz, rel, frac, hit = [], [], [], []
        for s in seeds:
            r = R.get((s, c))
            if r is None:
                continue
            n = sum(a >= lp.SETTLE_S for a, _ in r["seizures"])
            sz.append(n / ((r["run_s"] - lp.SETTLE_S) / 3600))
            ref = R.get((s, REF))
            q = (r["t"] > lp.SETTLE_S) & ~r["ictal"]
            if ref is not None:
                qr = (ref["t"] > lp.SETTLE_S) & ~ref["ictal"]
                rel.append(np.median(r["p30"][q]) / np.median(ref["p30"][qr]))
            if s in thr:
                f_, h_ = alarm_stats(r, thr[s])
                frac.append(f_)
                hit += [h for a_, h, _ in h_ if not (c == UNT and a_ in cal[s])]
        for k, v in enumerate((sz, rel, frac)):
            if v:
                ax[k].plot(i + jit.uniform(-0.12, 0.12, len(v)), v, "o", color=col[c], ms=6, alpha=0.85)
                ax[k].plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=col[c], lw=2)
        if hit:
            ax[3].bar(i, np.mean(hit), color=col[c], width=0.6)
            ax[3].text(i, np.mean(hit) + 0.02, f"{sum(hit)}/{len(hit)}", ha="center", fontsize=8, color=lp.MUTED)
    titles = ("seizures per hour (per seed)", "seizure-free p30, median / reference", "share of seizure-free time in alarm",
              "held-out seizures starting during an alarm")
    for a, tl in zip(ax, titles):
        a.set_xticks(range(len(conds)))
        a.set_xticklabels(conds, rotation=35, ha="right", fontsize=8)
        a.set_title(tl, fontsize=10, loc="left")
        lp.style(a)
    ax[1].axhline(1, color=lp.MUTED, lw=0.8)
    ax[2].set_ylim(0, 1)
    ax[3].set_ylim(0, 1.1)
    fig.suptitle(f"LFP power 0-30 Hz across treatments ({len(seeds)} seeds); alarm threshold per seed = {MARGIN:g} x the "
                 f"lowest pre-onset level (last {PRE_S:g} s) over its first {N_CAL} untreated seizures",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "plots" / "p30_summary.png", dpi=110)
    plt.close(fig)
    # ----------------------------------------------------------------- print
    for s in seeds:
        if s in thr:
            print(f"seed {s}: threshold {thr[s]:.1f} pA²")
    for c in conds:
        fr = [alarm_stats(R[(s, c)], thr[s])[0] for s in thr if (s, c) in R]
        hits = [r for r in rows if r["condition"] == c and r["role"] in ("held out", "treatment")]
        on = [r["alarm_on_for_s"] for r in hits if r["alarm_on"]]
        print(f"{c:15s} time in alarm {np.mean(fr) if fr else np.nan:.2f}; held-out seizures {len(hits)}, "
              f"in alarm {len(on)}" + (f", alarm on for {np.median(on):.0f} s (median)" if on else ""))
    print(f"wrote {out}/p30_windows.csv, threshold.csv, plots/p30_seed*.png, rates_seed*.png, p30_summary.png, rates_summary.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else THIS_DIR / "results" / "lfp_tau250_15k")
