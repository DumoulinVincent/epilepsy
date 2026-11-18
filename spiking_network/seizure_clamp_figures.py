"""Figures of the seizure clamp (seizure_clamp.py).

    python seizure_clamp_figures.py     # -> results/seizure_clamp/figures/clamp_{overview,inzone}.png

deaff:    from deafferentation (40 -> 36 Hz), resumable long runs; plotted from whatever exists.
overview: start at k_E 0.92, 50000 s (tau 2500 s; tau 25000 s with D 10 and 30 s), fast (100 s) rate sensor.
inzone:   start at k_E 0.95, 100000 s: (a) tau 25000 s, sensor 2500 s; (b) tau 25000 s, sensor 100 s;
          (c) tau 10000 s, sensor 1000 s; (d) drop in k_E after each seizure against the time to the nearest other one.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "seizure_clamp" / "raw"
OUT = HERE / "results" / "seizure_clamp" / "figures"
RED = "#e34948"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})
COLS = plt.cm.viridis(np.linspace(0, 0.85, 5))


def load(name):
    z = np.load(RAW / name)
    r, t = z["rate_E"], z["rate_t"]
    w = t >= 0; r, t = r[w], t[w]; hi = r > 10
    return z["t_k"], z["k_AMPA"], t[np.flatnonzero(hi & ~np.r_[False, hi[:-1]])]


def traces(a, fmt, title, ylim):
    for s, col in zip(range(1, 6), COLS):
        tk, k, on = load(fmt.format(s=s))
        a.plot(tk / 3600, k, color=col, lw=1.2, label=f"network {s}")
        a.plot(on / 3600, np.interp(on, tk, k), "v", color=RED, ms=6.5, mec="k", mew=0.4)
    a.axhspan(0.945, 0.99, color=RED, alpha=0.06, lw=0)
    a.set_ylim(*ylim); a.set_ylabel("$k_E$ (E mean)")
    a.set_title(title, loc="left", fontsize=9.5)


def overview():
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for a, (tau, D) in zip(ax, ((2500, 10), (25000, 10), (25000, 30))):
        traces(a, f"s{{s}}_f36_tau{tau}_D{D}_k0.92_50000s.npz",
               f"HSP $\\tau$ = {tau} s, sensor 100 s, seizure ended after D = {D} s (red: onsets)", (0.80, 0.99))
    ax[0].legend(frameon=False, ncol=5, fontsize=8, loc="lower right")
    ax[-1].set_xlabel("time (h, model)")
    fig.savefig(OUT / "clamp_overview.png", dpi=150, bbox_inches="tight")


def inzone():
    fig = plt.figure(figsize=(12.5, 9.5))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3)
    conds = [("tau25000_sensor2500", "HSP $\\tau$ = 25000 s, slow sensor (2500 s)"),
             ("tau25000", "HSP $\\tau$ = 25000 s, fast sensor (100 s)"),
             ("tau10000_sensor1000", "HSP $\\tau$ = 10000 s, sensor 1000 s")]
    for row, (c, title) in enumerate(conds):
        a = fig.add_subplot(gs[row, :2])
        traces(a, f"s{{s}}_f36_{c}_D10_k0.95_100000s.npz",
               f"({'abc'[row]}) {title}; seizures ended after 10 s (red: onsets)", (0.87, 0.965))
        a.set_xlim(0, 27.8)
        if row == 0:
            a.legend(frameon=False, ncol=5, fontsize=7.5, loc="lower left")
    a.set_xlabel("time (h, model)")
    d = fig.add_subplot(gs[:2, 2])
    for c, mk, col, lab, after in (("tau25000_sensor2500", "o", "#7b4fbf", "slow sensor (2500 s)", 3000),
                                   ("tau25000", "s", "#2a78d6", "fast sensor (100 s)", 1000)):
        G, K = [], []
        for s in range(1, 6):
            tk, k, on = load(f"s{s}_f36_{c}_D10_k0.95_100000s.npz")
            for i, o in enumerate(on):
                G.append(min([abs(o - x) for j, x in enumerate(on) if j != i] or [1e6]) / 3600)
                K.append(np.interp(o - 10, tk, k) - np.interp(o + after, tk, k))
        d.plot(G, K, mk, color=col, ms=6, mfc="none" if mk == "s" else col, alpha=0.85, label=lab)
    d.set_xscale("log"); d.set_yscale("log"); d.set_ylim(1e-4, 0.03)
    d.set_xlabel("time to the nearest other seizure (h)"); d.set_ylabel("drop in $k_E$ after the seizure")
    d.set_title("(d) Is one seizure felt by HSP?", loc="left")
    d.legend(frameon=False, fontsize=8, loc="lower left")
    d.text(0.3, 0.016, "pairs < 1 h apart", fontsize=8, color="#7b4fbf")
    e = fig.add_subplot(gs[2, 2]); e.axis("off")
    e.text(0, 1, "Slow sensor: a lone 10 s seizure raises the\nrate estimate by ~1.2 Hz, staying below the\n"
                 "3 Hz internalization threshold: k$_E$ drops\n~0.0003. Two seizures < 1 h apart add up\n"
                 "above 3 Hz: k$_E$ drops 0.003-0.013.\nFast sensor: every seizure is felt (-0.010).",
           va="top", fontsize=8.5)
    fig.savefig(OUT / "clamp_inzone.png", dpi=150, bbox_inches="tight")


def deaff():
    """Seizure clamp from deafferentation (seizure_clamp.py deaff): k_E and onsets per network and setting. Uses whatever
    has been simulated so far (runs keep partial results)."""
    d = HERE / "results" / "seizure_clamp" / "deaff"
    sets = sorted({p.name.split("_", 1)[1] for p in d.glob("s*_up*.npz")})
    if not sets:
        print("no deafferentation clamp runs yet"); return
    fig, ax = plt.subplots(len(sets), 1, figsize=(12, 3.2 * len(sets)), squeeze=False)
    for a, tag in zip(ax[:, 0], sets):
        for s, col in zip(range(1, 6), COLS):
            p = d / f"s{s}_{tag}"
            if not p.exists():
                continue
            z = np.load(p); r, t = z["rate_E"], z["rate_t"]; w = t >= 0; r, t = r[w], t[w]; hi = r > 10
            on = t[np.flatnonzero(hi & ~np.r_[False, hi[:-1]])]
            a.plot(z["t_k"] / 3600, z["k_AMPA"], color=col, lw=1.1, label=f"network {s}")
            a.plot(on / 3600, np.interp(on, z["t_k"], z["k_AMPA"]), "v", color=RED, ms=6, mec="k", mew=0.4)
        a.axhspan(0.945, 0.99, color=RED, alpha=0.06, lw=0); a.set_ylim(0.72, 0.99); a.set_ylabel("$k_E$")
        a.set_title(f"From deafferentation, seizures ended after 10 s: {tag.replace('_', ' ').replace('.npz', '')}",
                    loc="left", fontsize=9.5)
        a.legend(frameon=False, ncol=5, fontsize=7.5, loc="lower right")
    ax[-1, 0].set_xlabel("time after deafferentation (h, model)")
    fig.savefig(OUT / "clamp_deaff.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "clamp_deaff.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    overview(); inzone(); deaff()
    print("wrote", OUT)
