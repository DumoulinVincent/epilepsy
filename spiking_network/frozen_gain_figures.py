"""Figure of the frozen-HSP analysis (frozen_gain.py full).

    python frozen_gain_figures.py      # -> results/frozen_gain/figures/frozen_gain.{png,pdf}

(a) regime map in (input rate, k_E): where rest survives, where rest and the high-rate state coexist, where rest
    is lost within 50 s; on top, where HSP (gains free) actually takes the network at each input, and hatched the
    part of the map it never reaches.
(b) the two branches at 36 Hz (frozen-gain sweeps up from rest and down from a kicked high state, 5 networks).
(c) risk R_Delta(k_E) = P(escape from rest within Delta | k_E), 36 Hz (dt 1 and 0.5 ms) and 40 Hz, Wilson 95%.
(d) survival from rest (fraction of runs still at rest) against time, 36 Hz, per k_E.
(e) k_E(t) of the 36 Hz cycle (tau 250 s) coloured by the measured escape hazard.
(f) k_E(t) at 38 Hz with HSP running (tau 50 s, 30000 s, 5 networks) and the high-rate onsets.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
D = HERE / "results" / "frozen_gain" / "full"
C = HERE / "results" / "frozen_gain" / "checks"
OUT = HERE / "results" / "frozen_gain" / "figures"
HIGH = 10.0
INK, MUTED, BLUE, RED, ORANGE, GREY = "#0b0b0b", "#52514e", "#2a78d6", "#e34948", "#eb6834", "#9a9890"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def sweep(path):
    z = np.load(path); m = json.loads(str(z["meta"])); r = z["rate_E"][m.get("kick_bins", 0):]; h = int(m["hold_s"])
    ks = np.array(m["kAs"])
    return ks, np.array([r[i * h + h // 2:(i + 1) * h].mean() for i in range(len(ks))])


def edges(f):
    """Upper edge: lowest k at which an up-sweep from rest is high (None if never); lower edge: lowest k at which a
    down-sweep from the high state is still high. Both noise seeds."""
    up, lo = [], []
    for ns in (31, 32):
        ks, r = sweep(D / "sweeps" / f"s1_f{f}_up_n{ns}.npz")
        up.append(ks[r > HIGH].min() if (r > HIGH).any() else np.nan)
        ks, r = sweep(D / "sweeps" / f"s1_f{f}_down_n{ns}.npz")
        lo.append(ks[r > HIGH].min())
    return up, lo


def trials():
    """First-escape time per frozen-gain trial, keyed (seed, f_in, k_E, dt_ms); the default dt (1 ms) is stored as 1.0.
    Trials of the main batch and of the checks batch (risk below 0.93 at 36 Hz, risk at 40 Hz, dt 0.5 ms)."""
    T = defaultdict(list)
    paths = sorted((D / "trials").glob("*.npz"))
    for tag in ("risk36_low", "risk40", "dt05"):
        paths += [p for p in sorted((C / tag).glob("*.npz")) if "fine" not in p.name]
    for p in paths:
        z = np.load(p); m = json.loads(str(z["meta"])); r = z["rate_E"][m["kick_bins"]:]
        hi = np.flatnonzero(r > HIGH)
        T[(m["seed"], m["f_in"], round(m["kA"], 3), m.get("dt_ms") or 1.0)].append(
            (float(hi[0]) if len(hi) else np.inf, m["T"]))
    return T


def hsp_runs():
    """Where HSP actually takes the network (gains free): 36 Hz cycle (tau 250 s, LFP runs), 40 Hz reference
    (5 networks), 38 Hz (tau 50 s, 30000 s, 5 networks)."""
    R = {}
    z = np.load(HERE / "results" / "lfp_tau250_15k" / "raw" / "seed1_untreated.npz")
    k = z["k_AMPA"]; R[36] = (k[len(k) // 4:].min(), k.max())
    ref = [np.load(p)["k_AMPA"] for p in sorted((HERE / "results" / "lfp_tau250_15k" / "raw").glob("seed*_reference_40Hz.npz"))]
    R[40] = np.mean([x[len(x) // 2:].mean() for x in ref])
    R[38] = []
    for p in sorted((HERE / "results" / "long_tau" / "raw").glob("tau50_seed*_untreated_38Hz_30000s.npz")):
        z = np.load(p); r = z["rate_E"]; hi = r > HIGH
        R[38].append(dict(t=z["t_k"], k=z["k_AMPA"], on=np.flatnonzero(hi & ~np.r_[False, hi[:-1]])))
    return R


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def risk(T, key, Dt):
    v = T[key]
    return wilson(sum(e <= Dt for e, _ in v), len(v))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = (32, 34, 35, 36, 37, 38, 40, 42)
    E = {f: edges(f) for f in fs}
    T = trials()
    R = hsp_runs()
    PURPLE, GREEN = "#7b4fbf", "#1baf7a"
    fig = plt.figure(figsize=(13.5, 12.5))
    gs = fig.add_gridspec(3, 3, hspace=0.58, wspace=0.34, height_ratios=[1.15, 1, 0.8])

    # (a) regime map + where HSP takes the network
    a = fig.add_subplot(gs[0, 0])
    fa = np.array(fs, float)
    lo = np.array([np.mean(E[f][1]) for f in fs])
    up = np.array([np.nanmin(E[f][0]) if np.isfinite(E[f][0]).any() else 1.02 for f in fs])
    a.fill_between(fa, 0, lo, color=BLUE, alpha=0.10, lw=0)
    a.fill_between(fa, lo, up, color=GREY, alpha=0.25, lw=0)
    a.fill_between(fa, up, 1.02, color=RED, alpha=0.12, lw=0)
    a.plot(fa, lo, "s-", color=RED, ms=3.5, lw=1, label="high state disappears (down-sweep)")
    a.plot(fa[up < 1.01], up[up < 1.01], "o-", color=BLUE, ms=3.5, lw=1, label="rest lost within 50 s (up-sweep)")
    # highest k_E HSP reaches at each input (gains free); above it the map is never visited
    reach = np.array([1.02, 1.02, up[2], R[36][1], up[4], max(x["k"].max() for x in R[38]), R[40], 0.45])
    a.fill_between(fa, reach, 1.02, facecolor="none", edgecolor=MUTED, hatch="///", lw=0, alpha=0.6,
                   label="never reached with HSP running")
    for f in (32, 34):
        a.annotate("", (f, 1.0), (f, 0.55), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    a.text(33, 0.35, "HSP\nsaturates\nat 1.0,\nno episode", ha="center", fontsize=7.5)
    a.plot([36, 36], R[36], color=ORANGE, lw=4, solid_capstyle="butt")
    k38 = np.concatenate([x["k"][np.searchsorted(x["t"], x["on"][0]):] for x in R[38]])
    a.plot([38, 38], [k38.min(), k38.max()], color=PURPLE, lw=4, solid_capstyle="butt")
    for f in (35, 37):
        a.plot([f, f], [0.02, up[fs.index(f)]], color=ORANGE, lw=1.5, ls=(0, (2, 2)))
    a.text(36.15, 0.20, "36 Hz\ncycle", color=ORANGE, fontsize=8)
    a.text(38.15, 0.07, "38 Hz\ncycle", color=PURPLE, fontsize=8)
    a.plot(40, R[40], "o", ms=8, mfc="white", mec=INK, mew=1.5, zorder=5)
    a.text(40, R[40] - 0.09, "40 Hz\nfixed point", ha="center", fontsize=7.5)
    a.annotate("", (42, 0.30), (42, 0.50), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
    a.text(41.85, 0.18, "42 Hz: fixed\npoint < 0.5", ha="right", color=GREEN, fontsize=7.5)
    a.text(39.2, 0.97, "rest lost", color=RED, fontsize=7.5)
    a.text(41.9, 0.80, "bistable", ha="right", color=MUTED, fontsize=7.5)
    a.set_xlim(31.6, 42.4); a.set_ylim(0, 1.02)
    a.set_xlabel("input rate $f_{in}$ (Hz)"); a.set_ylabel("$k_E$  ($k_I = 1 - k_E$)")
    a.set_title("(a) Frozen-gain map + where HSP goes", loc="left")
    a.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(-0.02, -0.13), ncol=2)

    # (b) branches at 36 Hz, 5 networks, + dt 0.5 ms for network 1
    b = fig.add_subplot(gs[0, 1])
    for s in range(1, 6):
        ks, r = sweep(D / "sweeps" / f"s{s}_f36_upfine_n41.npz")
        b.plot(ks, np.maximum(r, 0.5), color=BLUE, lw=1, alpha=0.8, label="up, from rest" if s == 1 else None)
        ks, r = sweep(D / "sweeps" / f"s{s}_f36_downfine_n41.npz")
        b.plot(ks, np.maximum(r, 0.5), color=RED, lw=1, alpha=0.8, label="down, from high state" if s == 1 else None)
    for nm in ("upfine", "downfine"):
        ks, r = sweep(C / "dt05" / f"s1_f36_{nm}_n41.npz")
        b.plot(ks, np.maximum(r, 0.5), color=INK, lw=1.2, ls="--", label="network 1, dt 0.5 ms" if nm == "upfine" else None)
    b.axvline(0.971, color=INK, ls=":", lw=1); b.text(0.968, 0.6, "$k_E^c$", ha="right", fontsize=8)
    b.axvspan(0.75, 0.99, color=GREY, alpha=0.15, lw=0)
    b.set_yscale("log"); b.set_ylim(0.5, 600)
    b.set_xlabel("frozen $k_E$"); b.set_ylabel("E rate (Hz), last 25 s of each 50 s step")
    b.set_title("(b) Hysteresis at 36 Hz (5 networks)", loc="left")
    b.legend(frameon=False, fontsize=7.5, loc="center left")

    # (c) R_Delta(k_E): 36 Hz (dt 1 and 0.5 ms), 40 Hz
    c = fig.add_subplot(gs[0, 2])
    k36 = sorted(k for (s, f, k, dt) in T if s == 1 and f == 36.0 and dt == 1.0)
    for Dt, col, lw in ((100, "#9cc2ee", 1), (300, "#5b9be0", 1), (1000, BLUE, 1.8)):
        p = np.array([risk(T, (1, 36.0, k, 1.0), Dt) for k in k36])
        c.plot(k36, p[:, 0], "o-", color=col, ms=3.5, lw=lw, label=f"36 Hz, $\\Delta$ = {Dt} s")
        if Dt == 1000:
            c.fill_between(k36, p[:, 1], p[:, 2], color=col, alpha=0.12, lw=0)
    k05 = sorted(k for (s, f, k, dt) in T if f == 36.0 and dt == 0.5)
    c.plot(k05, [risk(T, (1, 36.0, k, 0.5), 1000)[0] for k in k05], "o", mfc="white", mec=INK, ms=6,
           label="36 Hz, dt 0.5 ms, $\\Delta$ = 1000 s")
    k40 = sorted(k for (s, f, k, dt) in T if f == 40.0 and k < 0.9)
    p = np.array([risk(T, (1, 40.0, k, 1.0), 1000) for k in k40])
    c.plot(k40, p[:, 0], "D-", color=GREEN, ms=4, lw=1.8, label="40 Hz, $\\Delta$ = 1000 s")
    c.fill_between(k40, p[:, 1], p[:, 2], color=GREEN, alpha=0.12, lw=0)
    c.axvline(R[40], color=GREEN, ls=":", lw=1.2)
    c.text(R[40] + 0.004, 0.62, "40 Hz HSP\nfixed point", color=GREEN, fontsize=7)
    on38 = [float(np.interp(o, x["t"], x["k"])) for x in R[38] for o in x["on"]]
    c.axvspan(min(on38), max(on38), color=PURPLE, alpha=0.15, lw=0)
    c.text(np.mean(on38), 1.06, "38 Hz onsets", color=PURPLE, fontsize=7, ha="center")
    c.axvline(0.971, color=INK, ls=":", lw=1)
    c.set_xlim(0.735, 1.0); c.set_ylim(-0.03, 1.1)
    c.set_xlabel("frozen $k_E$"); c.set_ylabel("$R_\\Delta$ = P(escape from rest within $\\Delta$)")
    c.set_title("(c) Risk (network 1; 36 Hz: 20 runs, 40 Hz: 10 runs)", loc="left")
    c.legend(frameon=False, fontsize=6.8, loc="center left", bbox_to_anchor=(0.0, 0.42))

    # (d) survival curves, 36 Hz, k_E >= 0.92
    d = fig.add_subplot(gs[1, 0])
    ksurv = [k for k in k36 if k >= 0.92]
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(ksurv)))
    tt = np.arange(0, 1001, 5)
    for k, col in zip(ksurv, cm):
        v = T[(1, 36.0, k, 1.0)]
        d.step(tt, [np.mean([e > t for e, _ in v]) for t in tt], where="post", color=col, lw=1.4, label=f"{k:.2f}")
    d.text(30, 0.05, "$k_E$ = 0.80-0.90: no escape\nin 20 x 1000 s", fontsize=7.5, color=MUTED)
    d.set_xlabel("time at frozen gains (s)"); d.set_ylabel("fraction of runs still at rest")
    d.set_title("(d) Survival from rest, 36 Hz", loc="left")
    d.legend(frameon=False, fontsize=7, title="$k_E$", title_fontsize=7.5, ncol=2, loc="upper right")

    # hazard (escapes per time at risk) at 36 Hz, all tested k_E
    haz = {k: sum(e < np.inf for e, _ in T[(1, 36.0, k, 1.0)]) / sum(min(e, Tm) for e, Tm in T[(1, 36.0, k, 1.0)])
           for k in k36}

    # (e) the 36 Hz cycle coloured by hazard
    e = fig.add_subplot(gs[1, 1:])
    z = np.load(HERE / "results" / "lfp_tau250_15k" / "raw" / "seed1_untreated.npz")
    t, k = z["t_k"] / 3600, z["k_AMPA"]
    kh = np.array(k36); hv = np.array([haz[x] * 1000 for x in kh])
    h = np.where(k >= kh[0], np.interp(k, kh, hv), np.nan)
    pts = np.c_[t, k][:, None, :]; segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = plt.get_cmap("magma_r").copy(); cmap.set_bad("#c9c7c0")
    lc = LineCollection(segs, cmap=cmap, norm=matplotlib.colors.LogNorm(0.3, 20), lw=2.5)
    hh = h[:-1].copy(); hh[hh <= 0] = np.nan
    lc.set_array(np.ma.masked_invalid(hh)); e.add_collection(lc)
    rE = z["rate_E"]; tr = np.arange(len(rE)) / 3600
    hi = rE > HIGH
    for i in np.flatnonzero(hi & ~np.r_[False, hi[:-1]]):
        e.axvline(tr[i], color=RED, lw=1)
    e.axhspan(0.75, 0.99, color=GREY, alpha=0.12, lw=0)
    e.axhline(0.90, color=MUTED, ls="--", lw=0.8)
    e.text(t[-1], 0.86, "below 0.90: no escape measured", ha="right", fontsize=7.5, color=MUTED)
    e.set_xlim(t[0], t[-1]); e.set_ylim(0, 1.02)
    e.set_xlabel("time (h, model; HSP $\\tau$ = 250 s)"); e.set_ylabel("$k_E$")
    cb = fig.colorbar(lc, ax=e, pad=0.01); cb.set_label("escape hazard (per 1000 s)")
    e.set_title(f"(e) 36 Hz cycle coloured by measured risk (grey: zero); $k_E \\geq$ 0.92 for "
                f"{np.mean(k[500:] >= 0.92):.0%} of the time; red: high-rate onsets", loc="left")

    # (f) 38 Hz with HSP running
    g = fig.add_subplot(gs[2, :])
    cols = plt.cm.Purples(np.linspace(0.45, 0.95, len(R[38])))
    for i, (x, col) in enumerate(zip(R[38], cols)):
        g.plot(x["t"] / 3600, x["k"], color=col, lw=1)
        g.plot(x["on"] / 3600, np.interp(x["on"], x["t"], x["k"]), "v", color=RED, ms=6,
               label="high-rate onset" if i == 0 else None)
    g.axhspan(0.75, 0.99, color=GREY, alpha=0.12, lw=0)
    g.set_xlim(0, R[38][0]["t"][-1] / 3600); g.set_ylim(0, 1.02)
    g.set_xlabel("time (h, model; HSP $\\tau$ = 50 s)"); g.set_ylabel("$k_E$")
    n_on = sum(len(x["on"]) for x in R[38])
    g.set_title(f"(f) 38 Hz with HSP running (5 networks): {n_on} episodes in 5 x 30000 s; rest fires ~1.65 Hz, so $k_E$ creeps "
                f"up slowly and onsets come at $k_E$ = {min(on38):.2f}-{max(on38):.2f}", loc="left")
    g.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.05))

    fig.savefig(OUT / "frozen_gain.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "frozen_gain.pdf", bbox_inches="tight")
    print("wrote", OUT / "frozen_gain.png")
    for kk in kh:
        print(f"  k_E {kk:.2f}: hazard {haz[kk] * 1000:.2f} per 1000 s")
    print("40 Hz HSP k_E", R[40], " 36 Hz range", R[36], " 38 Hz onsets", np.round(sorted(on38), 3))


if __name__ == "__main__":
    main()
