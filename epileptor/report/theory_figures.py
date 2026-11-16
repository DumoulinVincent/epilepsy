"""Theory figures for the HSP -> Epileptor report (figs/th_*.png).

    python theory_figures.py

Epileptor (Jirsa et al. 2014; TVB defaults) and its 2D reduction (Proix et al. 2014), integrated with
scipy (LSODA). Time is in model units (TVB: 1 unit = 1 ms, r = 1/tau0 = 0.00035).
"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.6})
INK, MUTED, RED, BLUE, ORANGE, GREEN = "#0b0b0b", "#52514e", "#e34948", "#2a78d6", "#eda100", "#1baf7a"

I_EXT, R = 3.1, 0.00035
g = lambda x1: -x1**3 - 2 * x1**2 + 1 + I_EXT          # fast nullcline z = g(x1)  # noqa: E731
X1_SN = -4.0 / 3.0
Z_SN = g(X1_SN)
X0_CRIT = X1_SN - Z_SN / 4.0


def fixed_points(x0):
    """Fixed points of the 2D Epileptor: g(x1) = 4 (x1 - x0)."""
    rts = np.roots([-1.0, -2.0, -4.0, 1 + I_EXT + 4 * x0])
    xs = sorted(x.real for x in rts if abs(x.imag) < 1e-9)
    out = []
    for x in xs:
        z = 4 * (x - x0)
        J = np.array([[-3 * x**2 - 4 * x, -1.0], [4 * R, -R]])
        ev = np.linalg.eigvals(J)
        out.append((x, z, bool(np.all(ev.real < 0)), ev))
    return out


# ---------------------------------------------------------------- 1. phase planes
def fig_phase_planes():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    x1 = np.linspace(-2.6, 1.3, 400)
    for a, x0, title in zip(ax, (-2.4, X0_CRIT, -1.8),
                            (f"(a) $x_0=-2.4$: stable rest", f"(b) $x_0=x_{{0,crit}}\\approx{X0_CRIT:.3f}$: fold",
                             "(c) $x_0=-1.8$: no rest state")):
        a.plot(x1, g(x1), color=INK, lw=1.8, label="fast nullcline $\\dot x_1=0$")
        a.plot(x1, 4 * (x1 - x0), color=BLUE, lw=1.3, label="slow nullcline $\\dot z=0$")
        for x, z, st, _ in fixed_points(x0):
            a.plot(x, z, "o", ms=6, mfc=(INK if st else "white"), mec=INK, zorder=5)
        # flow arrows (x1 fast: show the direction of dx1/dt on a grid, scaled)
        X, Zg = np.meshgrid(np.linspace(-2.5, 1.2, 14), np.linspace(-0.5, 5.5, 11))
        U = -X**3 - 2 * X**2 + 1 - Zg + I_EXT
        V = R * (4 * (X - x0) - Zg) * 400                  # slow flow exaggerated 400x for visibility
        n = np.hypot(U, V) + 1e-12
        a.quiver(X, Zg, U / n, V / n, color="#9a9890", scale=28, width=0.004, alpha=0.7)
        a.plot(X1_SN, Z_SN, "s", ms=4, color=RED, zorder=6)
        a.set_xlim(-2.6, 1.3)
        a.set_ylim(-0.5, 5.5)
        a.set_title(title, loc="left", fontsize=8)
        a.set_xlabel("$x_1$")
    ax[0].set_ylabel("$z$")
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(FIGS / "th_phase_planes.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- 2. bifurcation diagram
def fig_bifurcation():
    x0s = np.linspace(-3.0, -1.0, 801)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for x0 in x0s:
        for x, z, st, ev in fixed_points(x0):
            c = INK if st else "#9a9890"
            ax[0].plot(x0, x, ".", ms=1.2, color=c)
            ax[1].plot(x0, z, ".", ms=1.2, color=c)
    for a in ax:
        a.axvline(X0_CRIT, color=RED, ls="--", lw=1)
        a.set_xlabel("$x_0$")
    ax[0].set_ylabel("$x_1^*$ (fixed point)")
    ax[1].set_ylabel("$z^*$ (fixed point)")
    ax[1].axhline(Z_SN, color=RED, ls=":", lw=1)
    ax[0].set_title("(a) fixed points: black stable, grey unstable", loc="left", fontsize=8)
    ax[1].set_title("(b) the interictal $z^*$ reaches $z_{SN}$ at $x_{0,crit}$", loc="left", fontsize=8)
    ax[0].annotate("interictal rest\n(lower branch)", (-2.7, -1.75), fontsize=7, color=MUTED)
    ax[0].annotate("$x_{0,crit}$", (X0_CRIT + 0.03, 0.6), fontsize=7, color=RED)
    fig.tight_layout()
    fig.savefig(FIGS / "th_bifurcation.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- 3. 2D simulations
def epi2d(t, y, x0f):
    x1, z = y
    x0 = x0f(t)
    h = 4 * (x1 - x0) - (0.1 * z**7 if z < 0 else 0.0)
    return [-x1**3 - 2 * x1**2 + 1 - z + I_EXT, R * (h - z)]


def fig_2d_sims():
    T = 60000.0
    cases = [("$x_0=-2.2$ (subcritical)", lambda t: -2.2),
             ("$x_0=-1.9$ (supercritical)", lambda t: -1.9),
             ("slow ramp of $x_0$ through $x_{0,crit}$", lambda t: -2.25 + 0.4 * np.sin(np.pi * t / T) ** 2)]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
    for a, (lab, x0f) in zip(ax, cases):
        sol = solve_ivp(epi2d, (0, T), [-1.8, 3.5], args=(x0f,), method="LSODA", max_step=20, rtol=1e-7, atol=1e-9)
        t = sol.t / 1000
        a.plot(t, sol.y[0], color=ORANGE, lw=0.9, label="$x_1$ (fast)")
        a2 = a.twinx()
        a2.plot(t, sol.y[1], color=BLUE, lw=1.2, label="$z$ (slow)")
        a2.axhline(Z_SN, color=BLUE, ls=":", lw=0.8)
        a2.set_ylabel("$z$", color=BLUE)
        a2.spines["right"].set_visible(True)
        a2.grid(False)
        a.set_ylabel("$x_1$", color=ORANGE)
        a.set_title(lab, loc="left", fontsize=8)
        if a is ax[2]:
            a3 = a.inset_axes([0.0, 1.0, 1.0, 0.18])
            tt = np.linspace(0, T, 400)
            a3.plot(tt / 1000, [x0f(v) for v in tt], color=INK, lw=1)
            a3.axhline(X0_CRIT, color=RED, ls="--", lw=0.8)
            a3.set_xticks([])
            a3.set_ylabel("$x_0$", fontsize=7)
            a3.tick_params(labelsize=6)
            a.set_title("")
    ax[-1].set_xlabel("time (s; TVB units, 1 unit = 1 ms, $r=0.00035$)")
    fig.tight_layout(h_pad=1.5)
    fig.savefig(FIGS / "th_2d_sims.png", dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------- 4. full 6D Epileptor
P6 = dict(a=1.0, b=3.0, c=1.0, d=5.0, r=0.00035, s=4.0, x0=-1.6, Iext=3.1, Iext2=0.45, tau=10.0, aa=6.0,
          bb=2.0, slope=0.0)


def epi6d(t, y, p=P6):
    x1, y1, z, x2, y2, gg = y
    if x1 < 0:
        f1 = p["a"] * x1**3 - p["b"] * x1**2
    else:
        f1 = -(p["slope"] - x2 + 0.6 * (z - 4.0) ** 2) * x1
    dx1 = y1 - f1 - z + p["Iext"]
    dy1 = p["c"] - p["d"] * x1**2 - y1
    h = p["s"] * (x1 - p["x0"]) - (0.1 * z**7 if z < 0 else 0.0)
    dz = p["r"] * (h - z)
    dx2 = -y2 + x2 - x2**3 + p["Iext2"] + p["bb"] * gg - 0.3 * (z - 3.5)
    f2 = 0.0 if x2 < -0.25 else p["aa"] * (x2 + 0.25)
    dy2 = (-y2 + f2) / p["tau"]
    dg = -0.01 * (gg - 0.1 * x1)
    return [dx1, dy1, dz, dx2, dy2, dg]


def fig_6d():
    T = 12000.0
    sol = solve_ivp(epi6d, (0, T), [-1.6, -15.0, 3.6, -0.9, 0.0, -0.1], method="LSODA", max_step=0.5,
                    rtol=1e-6, atol=1e-8)
    t = sol.t / 1000
    x1, z, x2 = sol.y[0], sol.y[2], sol.y[3]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True)
    ax[0].plot(t, x2 - x1, color=INK, lw=0.5)
    ax[0].set_ylabel("$x_2-x_1$\n(LFP-like)")
    ax[0].set_title("Full Epileptor (TVB defaults, $x_0=-1.6$)", loc="left", fontsize=8)
    ax[1].plot(t, x1, color=ORANGE, lw=0.5, label="$x_1$ (population 1, fast discharges)")
    ax[1].plot(t, x2, color=GREEN, lw=0.5, label="$x_2$ (population 2, spike-wave)")
    ax[1].legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
    ax[2].plot(t, z, color=BLUE, lw=1.2)
    ax[2].set_ylabel("$z$ (permittivity)")
    ax[2].set_xlabel("time (s; TVB units)")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(FIGS / "th_6d.png", dpi=220)
    plt.close(fig)
    return float(z.min()), float(z.max())


# ---------------------------------------------------------------- 5. relaxation cycle + near-fold scaling
def fig_relaxation():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x0 = -1.9
    sol = solve_ivp(epi2d, (0, 30000), [-1.8, 3.5], args=(lambda t: x0,), method="LSODA", max_step=5,
                    rtol=1e-8, atol=1e-10)
    k = sol.t > 10000
    x1 = np.linspace(-2.4, 1.3, 400)
    ax[0].plot(x1, g(x1), color=INK, lw=1.6, label="fast nullcline")
    ax[0].plot(x1, 4 * (x1 - x0), color=BLUE, lw=1.2, label="slow nullcline, $x_0=-1.9$")
    ax[0].plot(sol.y[0][k], sol.y[1][k], color=ORANGE, lw=1.2, label="limit cycle")
    ax[0].plot(X1_SN, Z_SN, "s", color=RED, ms=4)
    ax[0].plot(0, g(0), "s", color=RED, ms=4)
    ax[0].annotate("onset fold ($z_{SN}$)", (X1_SN, Z_SN), xytext=(-1.05, 2.4), fontsize=7,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax[0].annotate("offset fold\n($z=1+I_{ext}$)", (0, g(0)), xytext=(0.62, 4.45), fontsize=7,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax[0].annotate("interictal: $z$ falls\non the lower branch", (-2.15, 3.55), fontsize=6.5, color=MUTED)
    ax[0].annotate("ictal: $z$ rises\non the upper branch", (0.35, 3.2), fontsize=6.5, color=MUTED)
    ax[0].set_xlim(-2.4, 1.3)
    ax[0].set_ylim(2.2, 4.9)
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$z$")
    ax[0].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0.12, 1.0))
    ax[0].set_title("(a) relaxation oscillation (2D reduction)", loc="left", fontsize=8)
    D = np.logspace(-4, 0, 200)
    exact = []
    for d in D:
        fp = [f for f in fixed_points(X0_CRIT - d) if f[0] <= X1_SN]
        exact.append(fp[0][1] - Z_SN)
    ax[1].loglog(D, exact, color=INK, lw=1.6, label="exact $z^*-z_{SN}$")
    ax[1].loglog(D, 2 * D**2, "--", color=RED, lw=1.2, label="$2D^2$")
    ax[1].loglog(D, 1 + 1 / D, color=BLUE, lw=1.2, label="stiffness $1+1/D$")
    ax[1].set_xlabel("$D = x_{0,crit} - x_0$")
    ax[1].set_title("(b) near the fold: $z^*$ hugs $z_{SN}$, the relaxation\nrate diverges", loc="left", fontsize=8)
    ax[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIGS / "th_relaxation.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    print(f"x1_SN = {X1_SN:.5f}, z_SN = {Z_SN:.5f}, x0_crit = {X0_CRIT:.5f}")
    for x0 in (-2.4, X0_CRIT - 1e-4, X0_CRIT + 1e-4, -1.8):
        print(x0, [(round(x, 3), round(z, 3), st) for x, z, st, _ in fixed_points(x0)])
    fig_phase_planes()
    fig_bifurcation()
    fig_2d_sims()
    print("6D z range", fig_6d())
    fig_relaxation()
