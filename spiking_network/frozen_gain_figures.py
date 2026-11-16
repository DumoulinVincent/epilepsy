"""Figure of the frozen-HSP analysis (frozen_gain.py full).

    python frozen_gain_figures.py      # -> results/frozen_gain/figures/frozen_gain.{png,pdf}

(a) regime map in (input rate, k_E): where rest survives, where rest and the high-rate state coexist, where rest
    is lost within 50 s; the 36 Hz HSP cycle and the 40 Hz healthy fixed point on top.
(b) the two branches at 36 Hz (frozen-gain sweeps up from rest and down from a kicked high state, 5 networks).
(c) risk R_Delta(k_E) = P(escape from rest within Delta | k_E), 36 Hz, Wilson 95% intervals.
(d) survival from rest (fraction of runs still at rest) against time, 36 Hz, per k_E.
(e) k_E(t) of the 36 Hz cycle (tau 250 s) coloured by the measured escape hazard.
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
    T = defaultdict(list)
    for p in sorted((D / "trials").glob("*.npz")):
        z = np.load(p); m = json.loads(str(z["meta"])); r = z["rate_E"][m["kick_bins"]:]
        hi = np.flatnonzero(r > HIGH)
        T[(m["seed"], m["f_in"], round(m["kA"], 3))].append((float(hi[0]) if len(hi) else np.inf, m["T"]))
    return T


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = (32, 34, 35, 36, 37, 38, 40, 42)
    E = {f: edges(f) for f in fs}
    T = trials()
    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)

    # (a) regime map
    a = fig.add_subplot(gs[0, 0])
    fa = np.array(fs, float)
    lo = np.array([np.mean(E[f][1]) for f in fs])
    up = np.array([np.nanmin(E[f][0]) if np.isfinite(E[f][0]).any() else 1.02 for f in fs])
    a.fill_between(fa, 0.5, lo, color=BLUE, alpha=0.12, lw=0)
    a.fill_between(fa, lo, up, color=GREY, alpha=0.25, lw=0)
    a.fill_between(fa, up, 1.02, color=RED, alpha=0.12, lw=0)
    a.plot(fa, lo, "s-", color=RED, ms=4, label="high state disappears (down-sweep)")
    a.plot(fa[up < 1.01], up[up < 1.01], "o-", color=BLUE, ms=4, label="rest lost within 50 s (up-sweep)")
    for f in (32, 34):
        a.annotate("", (f, 1.015), (f, 0.97), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1))
    a.text(33, 0.93, "no escape\nup to 1.0", ha="center", fontsize=7.5, color=BLUE)
    a.plot([40, 42], [0.78, 0.50], "D", color="#1baf7a", ms=5, label="rest fires 2 Hz (HSP fixed point)")
    a.annotate("", (36, 0.97), (36, 0.62), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2))
    a.text(36.25, 0.66, "36 Hz:\nHSP build-up", color=ORANGE, fontsize=8)
    a.plot(40, 0.77, "o", ms=8, mfc="white", mec=INK, mew=1.5)
    a.text(39.6, 0.80, "40 Hz healthy", fontsize=8, ha="right")
    a.text(41.5, 0.55, "rest only", ha="right", color=BLUE, fontsize=8)
    a.text(33.0, 0.84, "bistable", ha="center", color=MUTED, fontsize=8.5)
    a.set_xlim(32, 42); a.set_ylim(0.5, 1.02)
    a.set_xlabel("input rate $f_{in}$ (Hz)"); a.set_ylabel("frozen $k_E$  ($k_I = 1 - k_E$)")
    a.set_title("(a) Regime map, gains frozen (network 1)", loc="left")
    a.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 0.08))

    # (b) branches at 36 Hz, 5 networks
    b = fig.add_subplot(gs[0, 1])
    for s in range(1, 6):
        ks, r = sweep(D / "sweeps" / f"s{s}_f36_upfine_n41.npz")
        b.plot(ks, np.maximum(r, 0.5), color=BLUE, lw=1, alpha=0.8, label="up, from rest" if s == 1 else None)
        ks, r = sweep(D / "sweeps" / f"s{s}_f36_downfine_n41.npz")
        b.plot(ks, np.maximum(r, 0.5), color=RED, lw=1, alpha=0.8, label="down, from high state" if s == 1 else None)
    b.axvline(0.971, color=INK, ls=":", lw=1); b.text(0.968, 0.6, "$k_E^c$", ha="right", fontsize=8)
    b.axvspan(0.75, 0.99, color=GREY, alpha=0.15, lw=0)
    b.set_yscale("log"); b.set_ylim(0.5, 600)
    b.set_xlabel("frozen $k_E$"); b.set_ylabel("E rate (Hz), last 25 s of each 50 s step")
    b.set_title("(b) Hysteresis at 36 Hz (5 networks)", loc="left")
    b.legend(frameon=False, fontsize=8, loc="center left")

    # (c) R_Delta(k_E), 36 Hz
    c = fig.add_subplot(gs[0, 2])
    ks = sorted(k for (s, f, k) in T if s == 1 and f == 36.0)
    for Dt, col in ((100, "#5b9be0"), (300, BLUE), (1000, INK)):
        P = [wilson(sum(e <= Dt for e, _ in T[(1, 36.0, k)]), len(T[(1, 36.0, k)])) for k in ks]
        p = np.array(P)
        c.plot(ks, p[:, 0], "o-", color=col, ms=4, label=f"$\\Delta$ = {Dt} s")
        c.fill_between(ks, p[:, 1], p[:, 2], color=col, alpha=0.12, lw=0)
    for s in range(2, 6):
        kk = sorted(k for (ss, f, k) in T if ss == s)
        c.plot(kk, [np.mean([e <= 500 for e, _ in T[(s, 36.0, k)]]) for k in kk], "x", color=GREY, ms=6,
               label="networks 2-5, $\\Delta$ = 500 s" if s == 2 else None)
    c.axvline(0.971, color=INK, ls=":", lw=1)
    c.set_ylim(-0.03, 1.03); c.set_xlabel("frozen $k_E$"); c.set_ylabel("$R_\\Delta$ = P(escape from rest within $\\Delta$)")
    c.set_title("(c) Risk at 36 Hz (network 1: 20 runs per point)", loc="left")
    c.legend(frameon=False, fontsize=7.5, loc="lower right")

    # (d) survival curves
    d = fig.add_subplot(gs[1, 0])
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(ks)))
    tt = np.arange(0, 1001, 5)
    haz = {}
    for k, col in zip(ks, cm):
        v = T[(1, 36.0, k)]
        d.step(tt, [np.mean([e > t for e, _ in v]) for t in tt], where="post", color=col, lw=1.4, label=f"{k:.2f}")
        haz[k] = sum(e < np.inf for e, _ in v) / sum(min(e, Tm) for e, Tm in v)
    d.set_xlabel("time at frozen gains (s)"); d.set_ylabel("fraction of runs still at rest")
    d.set_title("(d) Survival from rest, 36 Hz", loc="left")
    d.legend(frameon=False, fontsize=7, title="$k_E$", title_fontsize=7.5, ncol=2, loc="upper right")

    # (e) the 36 Hz cycle coloured by hazard
    e = fig.add_subplot(gs[1, 1:])
    z = np.load(HERE / "results" / "lfp_tau250_15k" / "raw" / "seed1_untreated.npz")
    t, k = z["t_k"] / 3600, z["k_AMPA"]
    kh = np.array(sorted(haz)); lh = np.log10([haz[x] * 1000 for x in kh])
    h = np.where(k >= kh[0], 10 ** np.interp(k, kh, lh), np.nan)
    pts = np.c_[t, k][:, None, :]; segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="magma_r", norm=matplotlib.colors.LogNorm(0.5, 20), lw=2.5)
    lc.set_array(h[:-1]); e.add_collection(lc)
    e.plot(t[np.isnan(h)], k[np.isnan(h)], ".", ms=0.8, color=GREY)
    rE = z["rate_E"]; tr = np.arange(len(rE)) / 3600
    hi = rE > HIGH
    for i in np.flatnonzero(hi & ~np.r_[False, hi[:-1]]):
        e.axvline(tr[i], color=RED, lw=1)
    e.axhspan(0.75, 0.99, color=GREY, alpha=0.12, lw=0)
    e.axhline(0.93, color=MUTED, ls="--", lw=0.8)
    e.text(t[-1], 0.935, "lowest $k_E$ tested for risk", ha="right", fontsize=7.5, color=MUTED)
    e.set_xlim(t[0], t[-1]); e.set_ylim(0, 1.02)
    e.set_xlabel("time (h, model; HSP $\\tau$ = 250 s)"); e.set_ylabel("$k_E$")
    cb = fig.colorbar(lc, ax=e, pad=0.01); cb.set_label("escape hazard (per 1000 s)")
    e.set_title(f"(e) The 36 Hz cycle with the measured risk: $k_E \\geq$ 0.93 for {np.mean(k[500:] >= 0.93):.0%} "
                f"of the time; red lines: high-rate onsets; grey band: bistable", loc="left")
    fig.savefig(OUT / "frozen_gain.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "frozen_gain.pdf", bbox_inches="tight")
    print("wrote", OUT / "frozen_gain.png")
    for kk in kh:
        print(f"  k_E {kk:.2f}: hazard {haz[kk] * 1000:.2f} per 1000 s")


if __name__ == "__main__":
    main()
