"""Figures for the step-by-step phase-plane section of the report (figs/pp_*.png).

    python phase_plane_figures.py

  pp_arrows    three phase planes (x0 below, at, above the critical value): branches, the slow drift along them,
               and a simulated trajectory with TVB's r
  pp_slowflow  the slow drift on the interictal branch, dz/dt = r[4(x1 - x0) - G(x1)], and z(t)
  pp_fpmove    the fixed point as x0 increases: path on the cubic, x1*(x0), eigenvalues
  pp_poster3b  poster panel 3B with every element numbered: HSP snapshots of x0 and their fixed points
2D Epileptor with TVB defaults (I_ext = 3.1, r = 0.00035); the HSP data come from
../results/trajectories.npz (seed 1, deafferented and healthy networks).
"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from theory_figures import FIGS, g as G, R, I_EXT, X1_SN, Z_SN, X0_CRIT, fixed_points, INK, MUTED, RED, BLUE, ORANGE

HERE = Path(__file__).resolve().parent
GREY = "#9a9890"
Gp = lambda x: -3 * x**2 - 4 * x  # noqa: E731   slope of the fast nullcline
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.6})


def draw_cubic(a, lo=-2.6, hi=0.9):
    """Fast nullcline with its attracting (solid) and repelling (dashed) parts."""
    x = np.linspace(lo, hi, 600)
    left, mid, right = x <= X1_SN, (x > X1_SN) & (x < 0), x >= 0
    a.plot(x[left], G(x[left]), color=INK, lw=2.2)
    a.plot(x[mid], G(x[mid]), color=GREY, lw=1.6, ls="--")
    a.plot(x[right], G(x[right]), color=INK, lw=2.2)
    a.plot(X1_SN, Z_SN, "s", ms=5, color=RED, zorder=7)
    a.plot(0, G(0), "s", ms=5, color=RED, zorder=7)


def branch_drift(a, x0, xs, color, size=11):
    """Small arrows tangent to the cubic showing the direction of the slow drift (sign of dz/dt on the curve)."""
    for x in xs:
        z = G(x)
        dz = 4 * (x - x0) - z                       # sign of dz/dt on the branch
        if abs(dz) < 0.02:
            continue
        # move along the curve in the direction that changes z with the sign of dz
        dx = 0.06 * np.sign(dz) * np.sign(Gp(x)) if Gp(x) != 0 else 0.06
        a.annotate("", xy=(x + dx, G(x + dx)), xytext=(x, z),
                   arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=size), zorder=6)


def simulate(x0, y0, T, max_step=2.0):
    f = lambda t, y: [G(y[0]) - y[1], R * (4 * (y[0] - x0) - y[1])]  # noqa: E731
    return solve_ivp(f, (0, T), y0, method="LSODA", max_step=max_step, rtol=1e-8, atol=1e-10)


def fp_marker(a, x0, ms=7):
    for x, z, st, _ in fixed_points(x0):
        a.plot(x, z, "o", ms=ms, mfc=(INK if st else "white"), mec=INK, mew=1.3, zorder=8)


# ====================================================================== 1. arrows, branches, trajectories
def fig_arrows():
    cases = [(-2.4, "(a) $x_0=-2.4$\ncrossing on the attracting branch"),
             (X0_CRIT, "(b) $x_0=x_{0,crit}=-2.062$\ncrossing at the fold"),
             (-1.8, "(c) $x_0=-1.8$\ncrossing on the repelling branch")]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True)
    for a, (x0, ttl) in zip(ax, cases):
        X, Zg = np.meshgrid(np.linspace(-2.5, 0.8, 12), np.linspace(2.1, 4.9, 9))
        U = G(X) - Zg; V = R * (4 * (X - x0) - Zg) * 400
        n = np.hypot(U, V) + 1e-12
        a.quiver(X, Zg, U / n, V / n, color="#c9c8c2", scale=30, width=0.004)
        draw_cubic(a)
        xl = np.linspace(-2.6, 0.9, 10)
        a.plot(xl, 4 * (xl - x0), color=BLUE, lw=1.5)
        branch_drift(a, x0, np.linspace(-2.35, -1.45, 5), ORANGE)
        branch_drift(a, x0, np.linspace(0.12, 0.6, 3), ORANGE)
        s = simulate(x0, [-0.6, 4.75], 9000 if x0 > -2 else 25000)
        a.plot(s.y[0], s.y[1], color=RED, lw=1.1, zorder=5)
        a.plot(-0.6, 4.75, "*", ms=8, color=RED, zorder=9)
        fp_marker(a, x0)
        a.set_xlim(-2.6, 0.9); a.set_ylim(2.0, 5.0)
        a.set_title(ttl, loc="left", fontsize=7.6); a.set_xlabel("$x_1$")
    ax[0].set_ylabel("$z$")
    hs = [Line2D([], [], color=INK, lw=2.2), Line2D([], [], color=GREY, ls="--"), Line2D([], [], color=BLUE),
          Line2D([], [], color=ORANGE, marker=">", lw=0), Line2D([], [], color=RED)]
    fig.legend(hs, ["attracting branch", "repelling branch", "slow nullcline", "slow drift on a branch",
                    "trajectory (TVB r)"], ncol=5, frameon=False, fontsize=7.2, loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIGS / "pp_arrows.png", dpi=220, bbox_inches="tight"); plt.close(fig)


# ====================================================================== 2. slow drift on the interictal branch
def fig_slowflow():
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    a = ax[0]
    x = np.linspace(-2.5, X1_SN, 300)
    for x0, col in ((-2.6, "#2a78d6"), (-2.4, "#5b9be0"), (-2.2, GREY), (X0_CRIT, RED), (-1.8, ORANGE)):
        h = 4 * (x - x0) - G(x)                     # (dz/dt)/r on the branch
        a.plot(x, h, color=col, lw=1.6, label=f"$x_0$ = {x0:.2f}")
        k = np.flatnonzero(np.sign(h[:-1]) != np.sign(h[1:]))
        for i in k:
            a.plot(x[i], 0, "o", ms=6, color=col, mec=INK, zorder=6)
    a.axhline(0, color=INK, lw=0.9)
    a.axvline(X1_SN, color=RED, lw=0.8, ls=":")
    a.text(X1_SN - 0.02, 2.0, "fold", ha="right", fontsize=7.5, color=RED)
    a.text(-2.45, 1.2, "$\\dot z>0$: $z$ rises,\nstate moves right on the branch", fontsize=7, color=MUTED)
    a.text(-2.45, -2.2, "$\\dot z<0$: $z$ falls,\nstate moves towards the fold", fontsize=7, color=MUTED)
    a.set_xlabel("$x_1$ on the interictal branch ($x_1<-4/3$)")
    a.set_ylabel("$\\dot z / r = 4(x_1 - x_0) - G(x_1)$")
    a.set_title("(a) slow drift along the interictal branch", loc="left", fontsize=8)
    a.legend(frameon=False, fontsize=6.8, loc="lower right")
    a.set_ylim(-3, 2.5)
    b = ax[1]
    for x0, col, lab in ((-2.4, "#5b9be0", "$x_0=-2.4$: settles at $z^*$"), (-1.8, ORANGE, "$x_0=-1.8$: seizures")):
        s = simulate(x0, [-2.2, 4.6], 9000)
        b.plot(s.t / 1000, s.y[1], color=col, lw=1.3, label=lab)
    b.axhline(Z_SN, color=RED, ls=":", lw=1); b.text(0.1, Z_SN + 0.03, "$z_{SN}$ (onset fold)", fontsize=7, color=RED)
    b.axhline(G(0), color=RED, ls=":", lw=1); b.text(0.6, G(0) + 0.03, "top fold, $z = 4.1$", fontsize=7, color=RED)
    b.set_xlabel("time (thousands of model units)"); b.set_ylabel("$z$")
    b.set_title("(b) $z(t)$ from the same start", loc="left", fontsize=8)
    b.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    b.set_ylim(2.8, 4.7)
    fig.tight_layout()
    fig.savefig(FIGS / "pp_slowflow.png", dpi=220); plt.close(fig)


# ====================================================================== 3. the fixed point as x0 increases
def fig_fpmove():
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.32)
    a = fig.add_subplot(gs[0, 0])
    draw_cubic(a, -2.0, -0.6)
    x0s = np.linspace(-2.5, -1.75, 7)
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(x0s)))
    xl = np.linspace(-2.0, -0.6, 10)
    for x0, c in zip(x0s, cm):
        a.plot(xl, 4 * (xl - x0), color=c, lw=1.1)
        (x, z, st, _), = fixed_points(x0)
        a.plot(x, z, "o", ms=6, mfc=(c if st else "white"), mec=c, mew=1.4, zorder=8)
    a.annotate("", xy=(-1.05, 3.08), xytext=(-1.68, 3.2),
               arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, connectionstyle="arc3,rad=0.35"))
    a.text(-1.22, 3.62, "$x_0$ increases:\nthe line slides right,\nthe crossing slides\nalong the cubic", fontsize=7)
    a.set_xlim(-2.0, -0.6); a.set_ylim(2.6, 4.0)
    a.set_xlabel("$x_1$"); a.set_ylabel("$z$")
    a.set_title("(a) the crossing moves through the fold", loc="left", fontsize=8)
    # x1*(x0)
    b = fig.add_subplot(gs[0, 1])
    xs = np.linspace(-2.4, -0.7, 400)
    x0_of = xs - G(xs) / 4
    st = xs < X1_SN
    b.plot(x0_of[st], xs[st], color=INK, lw=2); b.plot(x0_of[~st], xs[~st], color=GREY, lw=1.6, ls="--")
    b.axvline(X0_CRIT, color=RED, ls="--", lw=0.9); b.axhline(X1_SN, color=RED, ls=":", lw=0.9)
    b.text(X0_CRIT + 0.03, -2.3, "$x_{0,crit}$", color=RED, fontsize=7.5)
    b.text(-2.9, X1_SN + 0.05, "fold, $x_1=-4/3$", color=RED, fontsize=7.5)
    b.set_xlabel("$x_0$"); b.set_ylabel("$x_1^*$")
    b.set_xlim(-3.0, -1.4)
    b.set_title("(b) $x_1^*$ rises steadily with $x_0$", loc="left", fontsize=8)
    # eigenvalues
    c = fig.add_subplot(gs[1, 0])
    x0g = np.linspace(-3.0, -1.4, 801)
    l1, l2 = [], []
    for x0 in x0g:
        (_, _, _, ev), = fixed_points(x0)
        ev = sorted(ev, key=lambda v: v.real)
        l1.append(ev[0].real); l2.append(ev[1].real)
    c.plot(x0g, l1, color=BLUE, lw=1.6, label="Re $\\lambda_1$")
    c.plot(x0g, l2, color=ORANGE, lw=1.6, label="Re $\\lambda_2$")
    c.plot(x0g, [Gp(fixed_points(x0)[0][0]) for x0 in x0g], color=INK, lw=0.8, ls=":", label="$G'(x_1^*)$")
    c.axhline(0, color=INK, lw=0.8); c.axvline(X0_CRIT, color=RED, ls="--", lw=0.9)
    c.set_yscale("symlog", linthresh=1e-3)
    c.set_xlabel("$x_0$"); c.set_ylabel("Re $\\lambda$ (symlog)")
    c.legend(frameon=False, fontsize=7, loc="lower right")
    c.set_title("(c) the eigenvalues change sign at the fold", loc="left", fontsize=8)
    # zoom: complex pair and Hopf
    d = fig.add_subplot(gs[1, 1])
    x0z = np.linspace(X0_CRIT - 0.03, X0_CRIT + 0.03, 1201)
    r1, r2, im = [], [], []
    for x0 in x0z:
        (_, _, _, ev), = fixed_points(x0)
        ev = sorted(ev, key=lambda v: v.real)
        r1.append(ev[0].real); r2.append(ev[1].real); im.append(abs(ev[0].imag))
    d.plot((x0z - X0_CRIT) * 1e3, np.array(r1) * 1e3, color=BLUE, lw=1.6, label="Re $\\lambda_1$")
    d.plot((x0z - X0_CRIT) * 1e3, np.array(r2) * 1e3, color=ORANGE, lw=1.6, label="Re $\\lambda_2$")
    d.plot((x0z - X0_CRIT) * 1e3, np.array(im) * 1e3, color=INK, lw=1.0, ls="--", label="|Im $\\lambda$| (rotation)")
    d.set_ylim(-110, 110)
    d.axhline(0, color=INK, lw=0.8)
    x0H = (X1_SN + R / 4) - G(X1_SN + R / 4) / 4
    d.axvline((x0H - X0_CRIT) * 1e3, color=RED, lw=0.9)
    d.text((x0H - X0_CRIT) * 1e3 + 1.0, -95, "Hopf", color=RED, fontsize=7.5)
    d.set_xlabel("$(x_0 - x_{0,crit})\\times10^3$"); d.set_ylabel("$\\lambda\\times10^3$")
    d.legend(frameon=False, fontsize=6.8, loc="upper left")
    d.set_title("(d) zoom: a narrow focus window, then Hopf", loc="left", fontsize=8)
    fig.savefig(FIGS / "pp_fpmove.png", dpi=220, bbox_inches="tight"); plt.close(fig)


# ====================================================================== 4. poster panel 3B explained
def fig_poster3b():
    Z = np.load(HERE.parent / "results" / "trajectories.npz")
    t, x0, ic = Z["s1|untreated|t"], Z["s1|untreated|x0"], Z["s1|untreated|ictal"]
    sz = Z["s1|untreated|seizures"]; sz = sz[sz[:, 0] > 500]
    e0, on1 = sz[0, 1], sz[1, 0]
    snaps = [e0 + 300, e0 + 900, e0 + 1800, 0.5 * (e0 + on1) + 600, on1 - 60]
    cols = ["#2a78d6", "#5b9be0", GREY, ORANGE, RED]
    x0r = float(np.median(Z["s1|reference 40Hz|x0"][Z["s1|reference 40Hz|t"] > 500]))
    fig = plt.figure(figsize=(7.4, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.3)
    a = fig.add_subplot(gs[0, 0])
    m = t > 500
    a.plot(t[m] / 3600, np.where(ic, np.nan, x0)[m], color="#eb6834", lw=1.4)
    a.axhline(X0_CRIT, color=INK, ls="--", lw=0.9); a.axhline(x0r, color=MUTED, lw=1.2)
    a.text(0.2, x0r - 0.16, "healthy", fontsize=7, color=MUTED)
    for a_, b_ in sz:
        a.axvspan(a_ / 3600, b_ / 3600, color=RED, alpha=0.3, lw=0)
    for i, (ts, c) in enumerate(zip(snaps, cols), 1):
        k = int(np.argmin(np.abs(t - ts)))
        a.plot(t[k] / 3600, x0[k], "o", ms=8, color=c, mec="white", zorder=6)
        a.text(t[k] / 3600 + 0.05, x0[k] - 0.2, str(i), fontsize=8, fontweight="bold", color=c)
    a.set_xlabel("time (h, model)"); a.set_ylabel("$x_0$ from HSP")
    a.set_title("(a) five moments of one HSP cycle", loc="left", fontsize=8)
    b = fig.add_subplot(gs[0, 1])
    draw_cubic(b, -2.7, -0.7)
    xl = np.linspace(-2.7, -0.7, 10)
    for i, (ts, c) in enumerate(zip(snaps, cols), 1):
        k = int(np.argmin(np.abs(t - ts)))
        v = float(x0[k])
        b.plot(xl, 4 * (xl - v), color=c, lw=1.3)
        (xf, zf, st, _), = fixed_points(v)
        b.plot(xf, zf, "o", ms=7, mfc=(c if st else "white"), mec=(INK if st else c), mew=(0.8 if st else 1.6), zorder=8)
        b.text(xf - 0.07, zf + 0.1, str(i), fontsize=8, fontweight="bold", color=c, ha="right")
    b.plot(xl, 4 * (xl - x0r), color=MUTED, lw=1.4, ls="--")
    (xf, zf, _, _), = fixed_points(x0r)
    b.plot(xf, zf, "o", ms=7, mfc=MUTED, mec=INK, zorder=8)
    b.annotate("healthy rest", (xf, zf), xytext=(-2.55, 2.45), fontsize=7, color=MUTED,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    b.annotate("fold: onset", (X1_SN, Z_SN), xytext=(-1.15, 2.45), fontsize=7, color=RED,
               arrowprops=dict(arrowstyle="-", color=RED, lw=0.7))
    b.text(-1.95, 5.6, "as HSP raises $x_0$, the rest\nstate slides down the branch\ntowards the fold (1→5)",
           fontsize=7)
    ins = b.inset_axes([0.5, 0.2, 0.47, 0.36])
    xi = np.linspace(-1.62, -1.22, 200)
    ins.plot(xi[xi <= X1_SN], G(xi[xi <= X1_SN]), color=INK, lw=2); ins.plot(xi[xi > X1_SN], G(xi[xi > X1_SN]), color=GREY, ls="--")
    for i, (ts, c) in enumerate(zip(snaps, cols), 1):
        if i < 3:
            continue
        k = int(np.argmin(np.abs(t - ts))); v = float(x0[k])
        ins.plot(xi, 4 * (xi - v), color=c, lw=1.1)
        (xf, zf, st, _), = fixed_points(v)
        ins.plot(xf, zf, "o", ms=6, mfc=(c if st else "white"), mec=INK, mew=0.8, zorder=8)
        ins.text(xf, zf + 0.012, str(i), fontsize=7.5, fontweight="bold", color=c, ha="center", va="bottom")
    ins.plot(xi, 4 * (xi - x0r), color=MUTED, lw=1.1, ls="--")
    ins.plot(*fixed_points(x0r)[0][:2], "o", ms=6, mfc=MUTED, mec=INK, zorder=8)
    ins.plot(X1_SN, Z_SN, "s", ms=4, color=RED, zorder=9)
    ins.set_xlim(-1.62, -1.22); ins.set_ylim(2.88, 3.08); ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("zoom near the fold", fontsize=6.5, pad=2)
    b.indicate_inset_zoom(ins, edgecolor=MUTED)
    b.set_xlim(-2.7, -0.7); b.set_ylim(2.3, 6.5)
    b.set_xlabel("$x_1$"); b.set_ylabel("$z$")
    b.set_title("(b) poster 3B: the rest state at each moment", loc="left", fontsize=8)
    fig.savefig(FIGS / "pp_poster3b.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    out = []
    for i, ts in enumerate(snaps, 1):
        k = int(np.argmin(np.abs(t - ts))); v = float(x0[k]); (xf, zf, st, ev), = fixed_points(v)
        out.append((i, (ts - e0) / 60, v, xf, zf, max(ev.real)))
    return out, x0r


if __name__ == "__main__":
    fig_arrows(); fig_slowflow(); fig_fpmove()
    rows, x0r = fig_poster3b()
    for i, tm, v, xf, zf, lam in rows:
        print(f"snapshot {i}: {tm:6.1f} min after episode end, x0 = {v:+.3f}, rest at x1 = {xf:+.3f}, z = {zf:.3f}, "
              f"slowest eigenvalue {lam:+.2e}")
    print("healthy x0", x0r, fixed_points(x0r)[0][:2])
    print("wrote pp_arrows, pp_slowflow, pp_fpmove, pp_poster3b")
