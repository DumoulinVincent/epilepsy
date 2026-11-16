"""Figures for the dynamics / bifurcation textbook (dynamics_textbook.tex).

    python figures.py            # all
    python figures.py hopf snic  # some

Every figure is computed from the equations shown in the text (scipy solve_ivp, LSODA or RK45).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figs"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, RED, GREEN, PURPLE, GREY = "#2a78d6", "#eb6834", "#e34948", "#1baf7a", "#4a3aa7", "#9a9890"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 1.5,
                     "savefig.bbox": "tight", "savefig.dpi": 220})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def stable(ax, x, y, **kw):
    ax.plot(x, y, color=kw.pop("color", INK), lw=kw.pop("lw", 1.8), **kw)


def unstable(ax, x, y, **kw):
    ax.plot(x, y, color=kw.pop("color", INK), lw=kw.pop("lw", 1.4), ls="--", **kw)


def dot(ax, x, y, kind="stable", ms=6, color=INK, z=5):
    fc = {"stable": color, "unstable": "white", "half": GREY}[kind]
    ax.plot(x, y, "o", ms=ms, mfc=fc, mec=color, mew=1.3, zorder=z)


def arrow(ax, x0, y0, x1, y1, color=INK, ms=10, lw=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw))


def flow(ax, f, xlim, ylim, n=22, density=1.0, color=GREY):
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    U, V = f(X, Y)
    ax.streamplot(X, Y, U, V, color=color, density=density, linewidth=0.6, arrowsize=0.8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)


def traj(f, y0, T, max_step=0.01, method="RK45", **kw):
    s = solve_ivp(lambda t, y: f(*y), (0, T), y0, max_step=max_step, method=method, rtol=1e-8, atol=1e-10, **kw)
    return s.t, s.y


# ======================================================================= 1. phase line
def fig_phaseline():
    f = lambda x: x - x**3  # noqa: E731
    x = np.linspace(-1.6, 1.6, 400)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    a = ax[0]
    a.axhline(0, color=MUTED, lw=0.8)
    a.plot(x, f(x), color=BLUE)
    for xs, k in ((-1, "stable"), (0, "unstable"), (1, "stable")):
        dot(a, xs, 0, k)
    for xa, xb in ((-1.55, -1.15), (-0.15, -0.6), (0.15, 0.6), (1.55, 1.15)):
        arrow(a, xa, 0, xb, 0, color=RED, ms=12, lw=1.6)
    a.set_xlabel("$x$"); a.set_ylabel(r"$\dot x = f(x)$")
    a.set_title(r"(a) $\dot x = x - x^3$: read the sign of $f$", loc="left")
    a = ax[1]
    for x0 in np.linspace(-1.5, 1.5, 15):
        t, y = traj(lambda x: [f(x)], [x0], 6.0, max_step=0.02)
        a.plot(t, y[0], color=BLUE, lw=0.9)
    a.axhline(1, color=INK, lw=0.8, ls=":"); a.axhline(-1, color=INK, lw=0.8, ls=":"); a.axhline(0, color=INK, lw=0.8, ls="--")
    a.set_xlabel("$t$"); a.set_ylabel("$x(t)$"); a.set_title("(b) the same flow in time", loc="left")
    fig.tight_layout(); save(fig, "f01_phaseline")


# ======================================================================= 2. trace-determinant plane
def fig_tracedet():
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1], hspace=0.55, wspace=0.45)
    a = fig.add_subplot(gs[:, :2])
    T = np.linspace(-3, 3, 300)
    a.plot(T, T**2 / 4, color=INK, lw=1.4)
    a.axhline(0, color=INK, lw=1.4); a.axvline(0, color=INK, lw=1.0, ymin=0.33)
    a.fill_between(T, -2, 0, color=PURPLE, alpha=0.08)
    a.text(0, -1.2, "saddle\n(det < 0)", ha="center", fontsize=8.5)
    a.text(-2.25, 1.0, "stable\nnode", ha="center", fontsize=8.5); a.text(2.25, 1.0, "unstable\nnode", ha="center", fontsize=8.5)
    a.text(-0.95, 3.2, "stable\nfocus", ha="center", fontsize=8.5); a.text(0.95, 3.2, "unstable\nfocus", ha="center", fontsize=8.5)
    a.text(0.08, 4.3, "centre", fontsize=8, rotation=90, va="top")
    a.text(2.0, 1.6, r"$\det = \mathrm{tr}^2/4$", fontsize=8, color=MUTED, rotation=28)
    a.set_xlim(-3, 3); a.set_ylim(-2, 4.5)
    a.set_xlabel(r"trace $\tau = \lambda_1 + \lambda_2$"); a.set_ylabel(r"determinant $\Delta = \lambda_1\lambda_2$")
    a.set_title("(a) classification of a fixed point", loc="left")
    cases = [("stable node", [[-2, 0], [0, -0.6]]), ("unstable node", [[2, 0], [0, 0.6]]),
             ("stable focus", [[-0.4, -2], [2, -0.4]]), ("unstable focus", [[0.4, -2], [2, 0.4]]),
             ("saddle", [[1, 0], [0, -1]]), ("centre", [[0, -1.5], [1.5, 0]])]
    for i, (name, A) in enumerate(cases):
        b = fig.add_subplot(gs[i // 2, 2 + i % 2])
        A = np.array(A, float)
        flow(b, lambda X, Y: (A[0, 0] * X + A[0, 1] * Y, A[1, 0] * X + A[1, 1] * Y), (-1, 1), (-1, 1), n=15,
             density=0.7, color=BLUE)
        b.plot(0, 0, "o", ms=4, color=INK)
        b.set_xticks([]); b.set_yticks([]); b.set_title(name, fontsize=8.5)
        b.set_aspect("equal")
    save(fig, "f02_tracedet")


# ======================================================================= 3. nullclines, FitzHugh-Nagumo
def fhn(I, eps=0.08, a=0.7, b=0.8):
    return lambda x, y: (x - x**3 / 3 - y + I, eps * (x + a - b * y))


def fig_nullclines():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for a, I, ttl in ((ax[0], 0.0, "(a) $I = 0$: one stable fixed point (excitable)"),
                      (ax[1], 0.5, "(b) $I = 0.5$: unstable fixed point, limit cycle")):
        f = fhn(I)
        flow(a, lambda X, Y: f(X, Y), (-2.5, 2.5), (-1.2, 2.0), n=25, density=1.1)
        x = np.linspace(-2.5, 2.5, 400)
        a.plot(x, x - x**3 / 3 + I, color=ORANGE, lw=2, label=r"$\dot x = 0$")
        a.plot(x, (x + 0.7) / 0.8, color=BLUE, lw=2, label=r"$\dot y = 0$")
        # fixed point
        r = np.roots([-1 / 3, 0, 1 - 1 / 0.8, I - 0.7 / 0.8]); xs = r[np.abs(r.imag) < 1e-9].real[0]
        ys = (xs + 0.7) / 0.8
        J = np.array([[1 - xs**2, -1], [0.08, -0.08 * 0.8]])
        dot(a, xs, ys, "stable" if np.trace(J) < 0 else "unstable", color=RED)
        y0 = [xs + 0.6, ys] if I == 0 else [xs + 0.05, ys]
        t, Y = traj(f, y0, 200, max_step=0.05)
        a.plot(Y[0], Y[1], color=INK, lw=1.4)
        a.set_xlabel("$x$ (fast)"); a.set_ylabel("$y$ (slow)"); a.set_title(ttl, loc="left")
    ax[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "f03_nullclines")


# ======================================================================= 4. saddle-node
def fig_saddlenode():
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    for a, mu, ttl in ((ax[0], -0.4, r"$\mu<0$: node + saddle"), (ax[1], 0.0, r"$\mu=0$: saddle-node"),
                       (ax[2], 0.4, r"$\mu>0$: nothing left")):
        flow(a, lambda X, Y: (mu + X**2, -Y), (-1.2, 1.2), (-1, 1), n=18, density=0.8, color=BLUE)
        if mu < 0:
            dot(a, -np.sqrt(-mu), 0, "stable", color=RED); dot(a, np.sqrt(-mu), 0, "unstable", color=RED)
        elif mu == 0:
            dot(a, 0, 0, "half", color=RED)
        a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=8.5); a.set_aspect("equal")
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$y$")
    a = ax[3]
    m = np.linspace(-1, 0, 200)
    stable(a, m, -np.sqrt(-m), color=RED); unstable(a, m, np.sqrt(-m), color=RED)
    dot(a, 0, 0, "half", color=RED)
    a.set_xlim(-1, 0.6); a.set_xlabel(r"$\mu$"); a.set_ylabel(r"$x^*$")
    a.set_title("bifurcation diagram", fontsize=8.5)
    a.text(-0.95, 0.75, "saddle (unstable)", fontsize=7.5, color=MUTED); a.text(-0.95, -0.82, "node (stable)", fontsize=7.5, color=MUTED)
    fig.suptitle(r"$\dot x = \mu + x^2,\ \ \dot y = -y$", fontsize=9.5, y=1.04)
    fig.tight_layout(); save(fig, "f04_saddlenode")


# ======================================================================= 5. transcritical and pitchfork
def fig_trans_pitch():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    m = np.linspace(-1, 1, 200)
    a = ax[0]
    stable(a, m[m < 0], 0 * m[m < 0], color=RED); unstable(a, m[m > 0], 0 * m[m > 0], color=RED)
    unstable(a, m[m < 0], m[m < 0], color=BLUE); stable(a, m[m > 0], m[m > 0], color=BLUE)
    a.set_title("(a) transcritical\n" + r"$\dot x = \mu x - x^2$", loc="left")
    a = ax[1]
    stable(a, m[m < 0], 0 * m[m < 0], color=RED); unstable(a, m[m > 0], 0 * m[m > 0], color=RED)
    stable(a, m[m > 0], np.sqrt(m[m > 0]), color=BLUE); stable(a, m[m > 0], -np.sqrt(m[m > 0]), color=BLUE)
    a.set_title("(b) supercritical pitchfork\n" + r"$\dot x = \mu x - x^3$", loc="left")
    a = ax[2]
    stable(a, m[m < 0], 0 * m[m < 0], color=RED); unstable(a, m[m > 0], 0 * m[m > 0], color=RED)
    unstable(a, m[m < 0], np.sqrt(-m[m < 0]), color=BLUE); unstable(a, m[m < 0], -np.sqrt(-m[m < 0]), color=BLUE)
    a.set_title("(c) subcritical pitchfork\n" + r"$\dot x = \mu x + x^3$", loc="left")
    for a in ax:
        a.set_xlabel(r"$\mu$"); a.set_ylabel(r"$x^*$"); a.set_ylim(-1.1, 1.1)
    fig.tight_layout(); save(fig, "f05_trans_pitch")


# ======================================================================= 6. Hopf
def hopf_xy(mu, sub):
    if sub:
        return lambda x, y: (mu * x - y + (x * x + y * y) * x - (x * x + y * y) ** 2 * x,
                             x + mu * y + (x * x + y * y) * y - (x * x + y * y) ** 2 * y)
    return lambda x, y: (mu * x - y - (x * x + y * y) * x, x + mu * y - (x * x + y * y) * y)


def fig_hopf():
    fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.9))
    # supercritical
    for a, mu, ttl in ((ax[0, 0], -0.3, r"$\mu=-0.3$: stable focus"), (ax[0, 1], 0.3, r"$\mu=+0.3$: stable cycle")):
        f = hopf_xy(mu, False)
        flow(a, lambda X, Y: f(X, Y), (-1.1, 1.1), (-1.1, 1.1), n=18, density=0.7, color=GREY)
        if mu > 0:
            th = np.linspace(0, 2 * np.pi, 200); a.plot(np.sqrt(mu) * np.cos(th), np.sqrt(mu) * np.sin(th), color=RED, lw=2)
        dot(a, 0, 0, "stable" if mu < 0 else "unstable", color=RED)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=8.5)
    a = ax[0, 2]
    m = np.linspace(-0.5, 0.5, 200)
    stable(a, m[m < 0], 0 * m[m < 0], color=RED); unstable(a, m[m > 0], 0 * m[m > 0], color=RED)
    stable(a, m[m > 0], np.sqrt(m[m > 0]), color=BLUE)
    a.set_xlabel(r"$\mu$"); a.set_ylabel("cycle amplitude $r$"); a.set_title(r"supercritical: $\dot r = \mu r - r^3$", fontsize=8.5)
    a.text(0.08, 0.35, r"$r = \sqrt{\mu}$", color=BLUE, fontsize=8)
    # subcritical
    mu = -0.15
    f = hopf_xy(mu, True)
    a = ax[1, 0]
    flow(a, lambda X, Y: f(X, Y), (-1.3, 1.3), (-1.3, 1.3), n=20, density=0.8, color=GREY)
    rs = np.sqrt((1 - np.sqrt(1 + 4 * mu)) / 2); rl = np.sqrt((1 + np.sqrt(1 + 4 * mu)) / 2)
    th = np.linspace(0, 2 * np.pi, 200)
    a.plot(rs * np.cos(th), rs * np.sin(th), color=RED, lw=1.4, ls="--")
    a.plot(rl * np.cos(th), rl * np.sin(th), color=RED, lw=2)
    dot(a, 0, 0, "stable", color=RED)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(r"$\mu=-0.15$: focus + unstable + stable cycle", fontsize=8.5)
    a = ax[1, 1]
    for r0, col in ((rs * 0.8, BLUE), (rs * 1.25, ORANGE)):
        t, Y = traj(f, [r0, 0], 60, max_step=0.02)
        a.plot(t, np.hypot(*Y), color=col, label=f"start at $r$ = {r0:.2f}")
    a.axhline(rs, color=RED, ls="--", lw=1); a.axhline(rl, color=RED, lw=1)
    a.set_xlabel("$t$"); a.set_ylabel("$r(t)$"); a.legend(frameon=False, fontsize=7.5, loc="center right")
    a.set_title("bistability: the start decides", fontsize=8.5)
    a = ax[1, 2]
    m = np.linspace(-0.25, 0.5, 300)
    stable(a, m[m < 0], 0 * m[m < 0], color=RED); unstable(a, m[m > 0], 0 * m[m > 0], color=RED)
    m2 = np.linspace(-0.25, 0, 200)
    unstable(a, m2, np.sqrt((1 - np.sqrt(1 + 4 * m2)) / 2), color=BLUE)
    stable(a, m, np.sqrt((1 + np.sqrt(1 + 4 * m)) / 2), color=BLUE)
    dot(a, -0.25, np.sqrt(0.5), "half", color=BLUE, ms=5)
    a.axvspan(-0.25, 0, color=PURPLE, alpha=0.08, lw=0)
    a.set_xlabel(r"$\mu$"); a.set_ylabel("cycle amplitude $r$"); a.set_xlim(-0.4, 0.5)
    a.set_title(r"subcritical: $\dot r = \mu r + r^3 - r^5$", fontsize=8.5)
    a.text(-0.39, 0.35, "fold of\ncycles", fontsize=7.5, color=MUTED)
    fig.tight_layout(); save(fig, "f06_hopf")


# ======================================================================= 7. SNIC
def fig_snic():
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
    for a, w, ttl in ((ax[0], 0.85, r"$\omega=0.85$: node + saddle on the circle"), (ax[1], 1.0, r"$\omega=1$: SNIC"),
                      (ax[2], 1.15, r"$\omega=1.15$: limit cycle")):
        f = lambda x, y, w=w: ((1 - x * x - y * y) * x - y * (w - y / max(np.hypot(x, y), 1e-9)) ,  # noqa: E731
                               (1 - x * x - y * y) * y + x * (w - y / max(np.hypot(x, y), 1e-9)))
        X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 21), np.linspace(-1.5, 1.5, 21))
        R = np.hypot(X, Y) + 1e-9; TH = np.arctan2(Y, X)
        dr = R * (1 - R**2); dth = w - np.sin(TH)
        U = dr * np.cos(TH) - R * dth * np.sin(TH); V = dr * np.sin(TH) + R * dth * np.cos(TH)
        a.streamplot(X, Y, U, V, color=GREY, density=0.7, linewidth=0.6, arrowsize=0.7)
        th = np.linspace(0, 2 * np.pi, 300); a.plot(np.cos(th), np.sin(th), color=RED if w > 1 else BLUE, lw=1.8)
        if w < 1:
            t1 = np.arcsin(w); t2 = np.pi - t1
            dot(a, np.cos(t1), np.sin(t1), "stable", color=RED); dot(a, np.cos(t2), np.sin(t2), "unstable", color=RED)
        elif w == 1:
            dot(a, 0, 1, "half", color=RED)
        dot(a, 0, 0, "unstable", color=INK, ms=4)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=8)
        a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5)
    a = ax[3]
    w = np.linspace(1.0005, 1.6, 300)
    a.plot(w, 2 * np.pi / np.sqrt(w**2 - 1), color=INK)
    a.set_yscale("log"); a.set_xlabel(r"$\omega$"); a.set_ylabel("period")
    a.set_title(r"period $= 2\pi/\sqrt{\omega^2-1}\to\infty$", fontsize=8)
    fig.suptitle(r"$\dot r = r(1-r^2),\ \ \dot\theta = \omega - \sin\theta$", fontsize=9.5, y=1.05)
    fig.tight_layout(); save(fig, "f07_snic")


# ======================================================================= 8. homoclinic (Strogatz 8.4.3)
MU_HC = -0.8645


def _escape(t, y):
    return np.hypot(y[0], y[1]) - 4.0


_escape.terminal = True


def homo(mu):
    return lambda x, y: (y, mu * y + x - x * x + x * y)


def cycle_period(mu, T=400.0):
    f = homo(mu)
    s = solve_ivp(lambda t, y: f(*y), (0, T), [1.0 + 0.02, 0.0], max_step=0.02, rtol=1e-9, atol=1e-11,
                  events=_escape)
    x, y, t = s.y[0], s.y[1], s.t
    if s.status == 1:
        return np.nan, s
    # upward zero crossings of y in the last half
    k = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0) & (t[:-1] > T / 2))
    return (np.diff(t[k]).mean() if len(k) > 2 else np.nan), s


def fig_homoclinic():
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
    for a, mu, ttl in ((ax[0], -0.92, r"$\mu=-0.92$: stable cycle"), (ax[1], MU_HC, r"$\mu\approx\mu_c$: homoclinic loop"),
                       (ax[2], -0.82, r"$\mu=-0.82$: no cycle")):
        f = homo(mu)
        flow(a, lambda X, Y: f(X, Y), (-0.6, 1.9), (-1.2, 1.2), n=20, density=0.8, color=GREY)
        if mu != -0.82:
            T = 300 if mu == -0.92 else 120
            s = solve_ivp(lambda t, y: f(*y), (0, T), [1.02, 0], max_step=0.01, rtol=1e-9, atol=1e-11,
                          events=_escape)
            k = s.t > s.t[-1] * 0.6
            a.plot(s.y[0][k], s.y[1][k], color=RED, lw=1.8)
        else:
            s = solve_ivp(lambda t, y: f(*y), (0, 60), [1.02, 0], max_step=0.01, rtol=1e-9, atol=1e-11,
                          events=_escape)
            a.plot(s.y[0], s.y[1], color=RED, lw=1.2)
        dot(a, 0, 0, "unstable", color=INK, ms=5); dot(a, 1, 0, "unstable", color=INK, ms=5)
        a.set_xticks([]); a.set_yticks([]); a.set_title(ttl, fontsize=8)
    a = ax[3]
    mus = MU_HC - np.geomspace(2e-3, 0.12, 12)
    per = [cycle_period(m, T=500)[0] for m in mus]
    a.semilogx(np.abs(mus - MU_HC), per, "o-", color=INK, ms=3)
    a.set_xlabel(r"$|\mu - \mu_c|$"); a.set_ylabel("period")
    a.set_title(r"period $\sim -\ln|\mu-\mu_c|$", fontsize=8)
    fig.suptitle(r"$\dot x = y,\ \ \dot y = \mu y + x - x^2 + xy$ (saddle at 0, focus at 1)", fontsize=9.5, y=1.05)
    fig.tight_layout(); save(fig, "f08_homoclinic")


# ======================================================================= 9. hysteresis
def fig_hysteresis():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a = ax[0]
    x = np.linspace(-1.6, 1.6, 400); m = x**3 - x
    up = (x < -1 / np.sqrt(3)); mid = (np.abs(x) <= 1 / np.sqrt(3)); lo = x > 1 / np.sqrt(3)
    stable(a, m[up], x[up], color=BLUE); unstable(a, m[mid], x[mid], color=BLUE); stable(a, m[lo], x[lo], color=BLUE)
    mf = 2 / (3 * np.sqrt(3))
    dot(a, mf, -1 / np.sqrt(3), "half", color=RED); dot(a, -mf, 1 / np.sqrt(3), "half", color=RED)
    # slow sweep
    eps = 0.02
    f = lambda t, y: [(y[1] + y[0] - y[0] ** 3) / 0.05, eps * np.sign(np.sin(2 * np.pi * t / 160))]  # noqa: E731
    s = solve_ivp(f, (0, 320), [-1.3, -0.0], max_step=0.05)
    mu = 0.8 * np.sin(2 * np.pi * s.t / 160)
    g = lambda t, y: [(0.8 * np.sin(2 * np.pi * t / 160) + y[0] - y[0] ** 3) / 0.05]  # noqa: E731
    s = solve_ivp(g, (0, 320), [-1.3], max_step=0.05)
    mu = 0.8 * np.sin(2 * np.pi * s.t / 160)
    k = s.t > 40
    a.plot(mu[k], s.y[0][k], color=ORANGE, lw=1.4, alpha=0.9)
    arrow(a, 0.3, -1.22, 0.55, -1.12, color=ORANGE); arrow(a, -0.3, 1.22, -0.55, 1.12, color=ORANGE)
    a.text(mf + 0.03, -0.25, "jump up", fontsize=8, color=ORANGE); a.text(-mf - 0.62, -0.25, "jump\ndown", fontsize=8, color=ORANGE)
    a.set_xlabel(r"$\mu$ (swept slowly)"); a.set_ylabel("$x$")
    a.set_title(r"(a) $\dot x = \mu + x - x^3$: two folds $\Rightarrow$ hysteresis", loc="left")
    a.axvspan(-mf, mf, color=PURPLE, alpha=0.07, lw=0)
    a = ax[1]
    eps = 0.002
    def sub(t, y):
        mu = -0.45 + 0.9 * (0.5 - abs((t * eps) % 1 - 0.5)) * 2
        r2 = y[0] ** 2 + y[1] ** 2
        return [mu * y[0] - y[1] + r2 * y[0] - r2**2 * y[0], y[0] + mu * y[1] + r2 * y[1] - r2**2 * y[1]]
    s = solve_ivp(sub, (0, 1 / eps), [1e-3, 0], max_step=0.05, rtol=1e-8, atol=1e-12)
    mu = -0.45 + 0.9 * (0.5 - np.abs((s.t * eps) % 1 - 0.5)) * 2
    a.plot(mu, np.hypot(s.y[0], s.y[1]), color=ORANGE, lw=1.0)
    mm = np.linspace(-0.25, 0.45, 200)
    stable(a, mm, np.sqrt((1 + np.sqrt(1 + 4 * mm)) / 2), color=BLUE, lw=1.0)
    m2 = np.linspace(-0.25, 0, 100); unstable(a, m2, np.sqrt((1 - np.sqrt(1 + 4 * m2)) / 2), color=BLUE, lw=1.0)
    a.set_xlabel(r"$\mu$ (ramped up, then down)"); a.set_ylabel("amplitude $r$")
    a.set_title("(b) subcritical Hopf, $\\mu$ ramped up then down:\nturns on late (slow passage), off near $\\mu=-1/4$", loc="left")
    fig.tight_layout(); save(fig, "f09_hysteresis")


# ======================================================================= 10. slow-fast relaxation
def fig_slowfast():
    eps = 0.02
    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 0.75], hspace=0.45, wspace=0.3)
    for col, aa, ttl in ((0, 0.5, r"(a) $a = 0.5$: fixed point on the repelling branch"),
                         (1, 1.2, r"(b) $a = 1.2$: fixed point on the attracting branch")):
        a = fig.add_subplot(gs[0, col])
        x = np.linspace(-2.4, 2.4, 400); y = x - x**3 / 3
        out = np.abs(x) > 1
        stable(a, x[(x < -1)], y[x < -1], color=ORANGE, lw=2.4); stable(a, x[x > 1], y[x > 1], color=ORANGE, lw=2.4)
        unstable(a, x[~out], y[~out], color=ORANGE, lw=1.8)
        a.axvline(-aa, color=BLUE, lw=1.6)
        dot(a, 1, 2 / 3, "half", color=RED, ms=5); dot(a, -1, -2 / 3, "half", color=RED, ms=5)
        f = lambda t, v: [(v[0] - v[0] ** 3 / 3 - v[1]) / eps, v[0] + aa]  # noqa: E731
        y0 = [-2.0, 0.5] if aa < 1 else [-0.6, -0.6]
        s = solve_ivp(f, (0, 25), y0, max_step=0.005, method="LSODA")
        a.plot(s.y[0], s.y[1], color=INK, lw=1.1)
        if aa < 1:
            arrow(a, 1.0, 0.75, -1.4, 0.75, color=INK, ms=14, lw=1.0, style="-|>")
            arrow(a, -1.0, -0.75, 1.4, -0.75, color=INK, ms=14, lw=1.0, style="-|>")
            a.text(-0.5, 0.86, "fast jump", fontsize=7.5); a.text(-0.2, -0.65, "fast jump", fontsize=7.5)
        xs = -aa; ys = xs - xs**3 / 3
        dot(a, xs, ys, "unstable" if aa < 1 else "stable", color=BLUE)
        a.set_xlim(-2.4, 2.4); a.set_ylim(-1.4, 1.4)
        a.set_xlabel("$x$ (fast)"); a.set_ylabel("$y$ (slow)"); a.set_title(ttl, loc="left", fontsize=8.5)
        a.text(1.55, -0.5, "attracting", fontsize=7, color=ORANGE); a.text(0.05, 0.25, "repelling", fontsize=7, color=ORANGE)
        b = fig.add_subplot(gs[1, col])
        b.plot(s.t, s.y[0], color=INK, lw=1.0); b.set_xlabel("$t$"); b.set_ylabel("$x$")
        if aa > 1:
            b.set_title("no oscillation; a large enough kick gives one excursion", fontsize=8)
        else:
            b.set_title("relaxation oscillation: slow drift, fast jumps", fontsize=8)
    fig.suptitle(r"$\varepsilon\,\dot x = x - x^3/3 - y,\ \ \dot y = x + a,\ \ \varepsilon = 0.02$", fontsize=9.5, y=1.01)
    save(fig, "f10_slowfast")


# ======================================================================= 11. bursting (Hindmarsh-Rose)
HR = dict(I=2.0, r=0.003, s=4.0, xR=-1.6)


def hr(t, v, I=HR["I"], r=HR["r"], s=HR["s"], xR=HR["xR"]):
    x, y, z = v
    return [y - x**3 + 3 * x**2 - z + I, 1 - 5 * x**2 - y, r * (s * (x - xR) - z)]


def fig_bursting():
    I = HR["I"]
    s = solve_ivp(hr, (0, 3500), [-1.5, -10, 2.0], max_step=0.05, method="LSODA", rtol=1e-8, atol=1e-10)
    k = s.t > 800
    t, x, z = s.t[k] - 800, s.y[0][k], s.y[2][k]
    # fast subsystem: equilibria z = -x^3 - 2x^2 + 1 + I; stability from (tr, det)
    xe = np.linspace(-2.2, 2.0, 600); ze = -xe**3 - 2 * xe**2 + 1 + I
    tr = -3 * xe**2 + 6 * xe - 1; det = 3 * xe**2 + 4 * xe
    st = (det > 0) & (tr < 0); sad = det < 0; uf = (det > 0) & (tr > 0)
    # cycle branch of the fast subsystem (z frozen), continued from the right
    zs = np.linspace(min(z) - 0.15, max(z) + 0.3, 70)
    cyc = []; v = [1.0, -5.0]
    for zz in zs:
        f = lambda t_, w: [w[1] - w[0] ** 3 + 3 * w[0] ** 2 - zz + I, 1 - 5 * w[0] ** 2 - w[1]]  # noqa: E731
        so = solve_ivp(f, (0, 200), v, max_step=0.02, rtol=1e-8, atol=1e-10)
        xx = so.y[0][so.t > 100]
        if xx.max() - xx.min() > 0.5:
            cyc.append((zz, xx.min(), xx.max())); v = so.y[:, -1]
        else:
            break
    cyc = np.array(cyc)
    fig = plt.figure(figsize=(7.4, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.5, wspace=0.32)
    a = fig.add_subplot(gs[0, 0]); a.plot(t, x, color=INK, lw=0.7); a.set_ylabel("$x$ (fast)")
    a.set_title("(a) square-wave bursting", loc="left")
    b = fig.add_subplot(gs[1, 0], sharex=a); b.plot(t, z, color=PURPLE); b.set_ylabel("$z$ (slow)"); b.set_xlabel("$t$")
    c = fig.add_subplot(gs[:, 1])
    def seg(mask, style, col, lw):
        idx = np.flatnonzero(mask); groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for g in groups:
            if len(g) > 1:
                c.plot(ze[g], xe[g], style, color=col, lw=lw)
    seg(st, "-", BLUE, 2.2); seg(sad, "--", BLUE, 1.3); seg(uf, ":", BLUE, 1.3)
    if len(cyc):
        c.plot(cyc[:, 0], cyc[:, 1], color=GREEN, lw=2); c.plot(cyc[:, 0], cyc[:, 2], color=GREEN, lw=2)
    c.plot(z, x, color=INK, lw=0.4, alpha=0.6)
    zf = -(-4 / 3) ** 3 - 2 * (-4 / 3) ** 2 + 1 + I
    dot(c, zf, -4 / 3, "half", color=RED, ms=5)
    c.annotate("fold (saddle-node):\nburst starts", (zf, -4 / 3), xytext=(zf + 0.12, -2.15), fontsize=7.5,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    if len(cyc):
        c.annotate("cycle hits the saddle\n(homoclinic): burst ends", (cyc[-1, 0], cyc[-1, 1]), xytext=(cyc[-1, 0] - 0.45, -0.2),
                   fontsize=7.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    c.set_xlabel("$z$ (slow, frozen for the fast subsystem)"); c.set_ylabel("$x$")
    c.set_xlim(min(z) - 0.25, max(z) + 0.25); c.set_ylim(-2.3, 2.3)
    c.set_title("(b) the burst on the fast bifurcation diagram", loc="left", fontsize=8.5)
    from matplotlib.lines import Line2D
    c.legend([Line2D([], [], color=BLUE, lw=2.2), Line2D([], [], color=BLUE, ls="--"), Line2D([], [], color=BLUE, ls=":"),
              Line2D([], [], color=GREEN, lw=2), Line2D([], [], color=INK, lw=0.6)],
             ["stable rest", "saddle", "unstable focus", "stable cycle (min/max)", "burster"],
             frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.62))
    fig.suptitle(r"Hindmarsh–Rose: $\dot x = y - x^3 + 3x^2 - z + I,\ \dot y = 1 - 5x^2 - y,\ \dot z = r[s(x - x_R) - z]$",
                 fontsize=9, y=1.0)
    save(fig, "f11_bursting")


# ======================================================================= 12. the ball-in-a-landscape picture
def fig_landscape():
    fig, ax = plt.subplots(2, 4, figsize=(7.4, 4.0))
    # top: saddle-node V = -mu x - x^3/3
    x = np.linspace(-1.6, 1.4, 400)
    for a, mu, ttl in zip(ax[0], (-0.6, -0.15, 0.0, 0.25),
                          (r"$\mu=-0.6$: valley + hilltop", r"$\mu=-0.15$: shallow valley", r"$\mu=0$: flat ledge",
                           r"$\mu=0.25$: no valley (ghost)")):
        V = -mu * x - x**3 / 3
        a.plot(x, V, color=INK, lw=1.8)
        if mu < 0:
            xs = -np.sqrt(-mu); a.plot(xs, -mu * xs - xs**3 / 3 + 0.07, "o", ms=9, color=ORANGE, mec=INK, zorder=5)
            xu = np.sqrt(-mu); a.plot(xu, -mu * xu - xu**3 / 3, "^", ms=6, color=GREY, zorder=5)
        elif mu == 0:
            a.plot(0, 0.07, "o", ms=9, color=ORANGE, mec=INK, zorder=5)
        else:
            a.plot(-0.6, 0.6 * mu + 0.072 + 0.07, "o", ms=9, color=ORANGE, mec=INK, zorder=5)
            arrow(a, -0.45, 0.13, 0.25, -0.16, color=RED, ms=10)
        a.set_title(ttl, fontsize=7.5); a.set_xticks([]); a.set_yticks([]); a.set_ylim(-1.0, 1.0)
    ax[0, 0].set_ylabel(r"$V(x)$" + "\n" + r"$\dot x=\mu+x^2$")
    # bottom: Epileptor fast landscape V = x^4/4 + 2x^3/3 - (1 + I - z) x
    I = 3.1
    x = np.linspace(-2.6, 1.2, 400)
    for a, z, ttl in zip(ax[1], (3.6, 3.1, 2.915, 4.1),
                         ("$z=3.6$: rest, two valleys", "$z=3.1$: $z$ falls, valley shrinks",
                          "$z=z_{SN}$: ledge, roll right", "$z=4.1$: ledge, roll back")):
        V = x**4 / 4 + 2 * x**3 / 3 - (1 + I - z) * x
        V = V - V.min()
        a.plot(x, V, color=INK, lw=1.8)
        rts = np.roots([-1, -2, 0, 1 + I - z]); rts = np.sort(rts[np.abs(rts.imag) < 1e-6].real)
        Vf = lambda q: q**4 / 4 + 2 * q**3 / 3 - (1 + I - z) * q - (x**4 / 4 + 2 * x**3 / 3 - (1 + I - z) * x).min()  # noqa: E731
        Gp = lambda q: -3 * q**2 - 4 * q  # noqa: E731
        for q in rts:
            if Gp(q) > 1e-3:
                a.plot(q, Vf(q), "^", ms=6, color=GREY, zorder=5)          # hill crest = repelling branch
        left = [q for q in rts if q < -4 / 3 - 1e-3]; right = [q for q in rts if q > 1e-3]
        ball = dict(ms=9, color=ORANGE, mec=INK, zorder=6)
        ghost = dict(ms=9, color=ORANGE, mec=INK, zorder=6, alpha=0.25)
        if z in (3.6, 3.1):                                               # at rest: ball in the left valley
            a.plot(left[0], Vf(left[0]) + 0.18, "o", **ball)
        elif abs(z - 2.915) < 1e-3:                                       # left valley gone: rolls right
            a.plot(-4 / 3, Vf(-4 / 3) + 0.18, "o", **ball); a.plot(right[-1], Vf(right[-1]) + 0.18, "o", **ghost)
            arrow(a, -1.15, Vf(-4 / 3) - 0.1, right[-1] - 0.15, Vf(right[-1]) + 0.45, color=RED, ms=10)
            a.text(-0.9, Vf(-4 / 3) + 0.35, "onset", color=RED, fontsize=7)
        else:                                                             # right valley gone: rolls back left
            a.plot(0, Vf(0) + 0.18, "o", **ball); a.plot(left[0], Vf(left[0]) + 0.18, "o", **ghost)
            arrow(a, -0.15, Vf(0) - 0.1, left[0] + 0.25, Vf(left[0]) + 0.45, color=BLUE, ms=10)
            a.text(-0.95, Vf(0) + 0.3, "offset", color=BLUE, fontsize=7)
        a.set_title(ttl, fontsize=7.5); a.set_xticks([]); a.set_yticks([]); a.set_ylim(-0.3, 3.0)
        a.set_xlabel("$x_1$")
    ax[1, 0].set_ylabel("$V(x_1)$ at fixed $z$\nEpileptor fast variable")
    fig.tight_layout(h_pad=1.2)
    save(fig, "f12_landscape")


if __name__ == "__main__":
    which = sys.argv[1:] or ["phaseline", "tracedet", "nullclines", "saddlenode", "trans_pitch", "hopf", "snic",
                             "homoclinic", "hysteresis", "slowfast", "bursting", "landscape"]
    for w in which:
        globals()[f"fig_{w}"]()
