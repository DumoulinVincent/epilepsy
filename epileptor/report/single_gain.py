"""Single-gain HSP -> Epileptor mapping (report v2): x0 = x0_crit + beta * ln(k_E / k_E^c).

    python single_gain.py

Reads ../results/trajectories.npz (HSP_Epileptor_trajectory.ipynb export: E-mean k_AMPA, k_GABA,
E rate, detector, seizure times, two-gain x0, every 1 s), recomputes x0 with the excitatory gain only, integrates z
with the notebook's backward-Euler scheme, prints every number quoted in the report and writes figs/sg_*.png.
Only the deafferented (36 Hz, "untreated") and healthy (40 Hz) networks are used.
"""
import math
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
Z = np.load(HERE.parent / "results" / "trajectories.npz")
SEEDS = [int(s) for s in Z["seeds"]]
UNT, REF = "untreated", "reference 40Hz"
I_EXT = float(Z["I_EXT"])
X1_SN = -4.0 / 3.0
Z_SN = I_EXT - 5.0 / 27.0
X0_CRIT = X1_SN - Z_SN / 4.0
X1_UP = 0.0                    # the ictal branch is x1 > 0 (local maximum of the fast nullcline at x1 = 0)
R_PER_S = 0.35
Z_DT_S = 0.1
BETA = 1.0
SETTLE_S, PAD_S = 500.0, 60.0
PRE_W, PRE_GAP = 300.0, 2.0
COMPARE_W, MATCHED_OFF, POST_S = 300.0, 1500.0, 600.0
K_FLOOR = 1e-3
C_UNT, C_REF, RED, INK, MUTED, GRID = "#eb6834", "#52514e", "#e34948", "#0b0b0b", "#52514e", "#e4e3de"
PH = (("build-up", "#9a9890"), ("last 5 min", "#eda100"), ("high-rate episode", RED), ("after", "#2a78d6"))
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": MUTED})


def run(seed, cond):
    g = lambda k: Z[f"s{seed}|{cond}|{k}"]  # noqa: E731
    return dict(t=g("t"), kA=np.maximum(g("kA"), K_FLOOR), kG=g("kG"), rE=g("rE"), ictal=g("ictal").astype(bool),
                seizures=g("seizures").reshape(-1, 2), x0_2g=g("x0"))


RUNS = {(s, c): run(s, c) for s in SEEDS for c in (UNT, REF)}


def interictal(r):
    ok = (r["t"] > SETTLE_S) & ~r["ictal"]
    for a, b in r["seizures"]:
        ok &= ~((r["t"] > a - PAD_S) & (r["t"] < b + PAD_S))
    return ok


# ------------------------------------------------------------ threshold gain k_E^c from the onsets
def lnk_per_onset(window=PRE_W):
    out = []
    for s in SEEDS:
        r = RUNS[(s, UNT)]
        for a, _ in r["seizures"]:
            if a - window < SETTLE_S:
                continue
            m = (r["t"] >= a - window) & (r["t"] < a - PRE_GAP) & ~r["ictal"]
            out.append(dict(seed=s, onset=a, lnk=float(np.median(np.log(r["kA"][m])))))
    return out


ONS = lnk_per_onset()
LNK_C = float(np.median([x["lnk"] for x in ONS]))
K_C = math.exp(LNK_C)
for r in RUNS.values():
    r["x0"] = X0_CRIT + BETA * (np.log(r["kA"]) - LNK_C)
    r["D"] = X0_CRIT - r["x0"]


# ------------------------------------------------------------ z (backward Euler + bisection, as in the notebook)
def cubic_real_roots(c):
    a = 2.0
    p, q = -a * a / 3.0, 2 * a**3 / 27.0 + c
    disc = (q / 2) ** 2 + (p / 3) ** 3
    if disc < 0:
        m = 2 * math.sqrt(-p / 3); th = math.acos(max(-1.0, min(1.0, 3 * q / (p * m)))) / 3
        return sorted(m * math.cos(th - 2 * math.pi * k / 3) - a / 3 for k in range(3))
    u = math.copysign(abs(-q / 2 + math.sqrt(disc)) ** (1 / 3), -q / 2 + math.sqrt(disc))
    v = math.copysign(abs(-q / 2 - math.sqrt(disc)) ** (1 / 3), -q / 2 - math.sqrt(disc))
    return [u + v - a / 3]


def x1_branch(z, ictal):
    rts = cubic_real_roots(z - 1 - I_EXT)
    if ictal:
        up = [x for x in rts if x >= X1_UP - 1e-9]
        return (max(up), False) if up else (X1_UP, True)
    lo = [x for x in rts if x <= X1_SN + 1e-9]
    return (min(lo), False) if lo else (X1_SN, True)


def z_fixed_point(x0):
    rts = np.roots([-1.0, -2.0, -4.0, 1 + I_EXT + 4 * x0])
    lo = [x.real for x in rts if abs(x.imag) < 1e-9 and x.real <= X1_SN]
    return 4 * (min(lo) - x0) if lo else Z_SN


def h_rhs(z, x0, ictal):
    x1, _ = x1_branch(z, ictal)
    return 4 * (x1 - x0) - z - (0.1 * z**7 if z < 0 else 0.0)


def implicit_step(z, x0, ictal, dt, rr):
    F = lambda y: y - z - dt * rr * h_rhs(y, x0, ictal)  # noqa: E731
    lo, hi = z - 1.0, z + 1.0
    while F(lo) > 0: lo -= 2 * (hi - lo)  # noqa: E701
    while F(hi) < 0: hi += 2 * (hi - lo)  # noqa: E701
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if F(mid) > 0: hi = mid  # noqa: E701
        else: lo = mid  # noqa: E701
    return 0.5 * (lo + hi)


def integrate_z(r, dt=Z_DT_S, rr=R_PER_S):
    t, x0, ic = r["t"], r["x0"], r["ictal"]
    z = z_fixed_point(x0[0])
    nsub = max(1, int(round(1.0 / dt)))
    Zs = np.empty(len(t)); FLAG = np.zeros(len(t), bool)
    for i in range(len(t)):
        Zs[i] = z
        _, FLAG[i] = x1_branch(z, bool(ic[i]))
        xa, xb = x0[i], x0[min(i + 1, len(t) - 1)]
        for k in range(nsub):
            z = implicit_step(z, xa + (xb - xa) * (k + 1) / nsub, bool(ic[i]), 1.0 / nsub, rr)
    return Zs, FLAG


def main():
    print(f"x1_SN = {X1_SN:.5f}, z_SN = {Z_SN:.5f}, x0_crit = {X0_CRIT:.5f}")
    v = np.array([x["lnk"] for x in ONS])
    print(f"threshold: {len(ONS)} onsets, k_E at onset = " + ", ".join(f"{math.exp(x):.4f}" for x in v))
    print(f"k_E^c = exp(median ln k) = {K_C:.4f}  (onset range {math.exp(v.min()):.4f}-{math.exp(v.max()):.4f}); "
          f"ceiling x0(k_E=1) = {X0_CRIT + BETA * math.log(1 / K_C):+.4f}")
    for w in (100.0, 600.0):
        print(f"  window {w:.0f} s: k_E^c = {math.exp(np.median([x['lnk'] for x in lnk_per_onset(w)])):.4f}")
    for r in RUNS.values():
        r["z"], r["flag"] = integrate_z(r)

    # single- vs two-gain x0 (the two-gain one is centred on its own H_c)
    d = np.concatenate([np.abs(r["x0"] - r["x0_2g"])[r["t"] > SETTLE_S] for r in RUNS.values()])
    d_ii = np.concatenate([np.abs(r["x0"] - r["x0_2g"])[interictal(r)] for (s, c), r in RUNS.items()])
    print(f"|x0 single - x0 two-gain|: interictal max {d_ii.max():.4f}, median {np.median(d_ii):.4f}; "
          f"all samples max {d.max():.4f}")

    # distance to criticality per condition
    for c in (UNT, REF):
        D = np.concatenate([RUNS[(s, c)]["D"][interictal(RUNS[(s, c)])] for s in SEEDS])
        x0 = np.concatenate([RUNS[(s, c)]["x0"][RUNS[(s, c)]["t"] > SETTLE_S] for s in SEEDS])
        fl = np.mean([np.mean(RUNS[(s, c)]["flag"][RUNS[(s, c)]["t"] > SETTLE_S]) for s in SEEDS])
        print(f"{c:15s} D median {np.median(D):+.3f}, min {D.min():+.3f}, share D<0 {np.mean(D < 0):.3f}; "
              f"x0 range {x0.min():+.2f} .. {x0.max():+.2f}; flagged {fl:.3f}")

    # questions answered with a leave-one-out threshold
    rows = []
    for s in SEEDS:
        r = RUNS[(s, UNT)]
        prev_end = SETTLE_S
        for a, b in r["seizures"]:
            if a - MATCHED_OFF - COMPARE_W < prev_end or a < SETTLE_S + COMPARE_W:
                prev_end = b; continue
            lnc = float(np.median([x["lnk"] for x in ONS if not (x["seed"] == s and x["onset"] == a)]))
            Dl = BETA * (lnc - np.log(r["kA"]))
            pre = (r["t"] >= a - COMPARE_W) & (r["t"] < a - PRE_GAP)
            mat = (r["t"] >= a - MATCHED_OFF - COMPARE_W) & (r["t"] < a - MATCHED_OFF)
            post = (r["t"] >= b) & (r["t"] < b + POST_S)
            span = (r["t"] > prev_end) & (r["t"] < a)
            neg = np.flatnonzero(span & (Dl <= 0))
            rows.append(dict(D_pre=np.median(Dl[pre]), D_mat=np.median(Dl[mat]),
                             D_on=Dl[np.flatnonzero(r["t"] < a)[-1]],
                             lead=a - r["t"][neg[0]] if len(neg) else np.nan,
                             dpost=np.median(Dl[post][-60:]) - np.median(Dl[post][:30])))
            prev_end = b
    pre = np.array([x["D_pre"] for x in rows]); mat = np.array([x["D_mat"] for x in rows])
    lead = np.array([x["lead"] for x in rows])
    print(f"{len(rows)} seizures evaluated; D at onset median {np.median([x['D_on'] for x in rows]):+.4f}; "
          f"D<=0 reached before onset in {np.sum(np.isfinite(lead))} (lead {np.nanmin(lead)/60:.0f}-{np.nanmax(lead)/60:.0f} min); "
          f"D at onset > 0 range {np.min([x['D_on'] for x in rows if x['D_on'] > 0] or [np.nan]):.3f}-"
          f"{np.max([x['D_on'] for x in rows]):.3f}")
    print(f"last 5 min vs 25-30 min earlier: median {np.median(pre):+.4f} vs {np.median(mat):+.4f}; "
          f"smaller in {np.sum(pre < mat)}/{len(rows)}; Wilcoxon p = {stats.wilcoxon(pre, mat).pvalue:.4f}")
    print(f"post-episode: D change over first {POST_S:.0f} s median {np.median([x['dpost'] for x in rows]):+.3f}")

    # repeatability: onset-aligned x0
    rel = np.arange(-3000, 1501, 10)
    A = []
    for s in SEEDS:
        r = RUNS[(s, UNT)]
        for a, _ in r["seizures"]:
            if a - 3000 > SETTLE_S and a + 1500 < r["t"][-1]:
                A.append(np.interp(a + rel, r["t"], np.where(r["ictal"], np.nan, r["x0"])))
    A = np.array(A); sd = np.nanstd(A, 0)
    for tt in (-1000, -100, 500):
        print(f"  across-seizure SD of x0 at {tt:+d} s: {sd[rel == tt][0]:.3f}  (n = {len(A)})")
    # z slaved to its fixed point between episodes
    dev = []
    for s in SEEDS:
        r = RUNS[(s, UNT)]; m = interictal(r)
        zs = np.array([z_fixed_point(x) for x in r["x0"][m][::5]])
        dev.append(np.abs(r["z"][m][::5] - zs))
    dev = np.concatenate(dev)
    print(f"|z - z*(x0)| interictal: 99th percentile {np.percentile(dev, 99):.4f}")
    figures(rel, A)


def phase_labels(r):
    lab = np.full(len(r["t"]), "build-up", dtype=object)
    for a, b in r["seizures"]:
        lab[(r["t"] >= a - 300) & (r["t"] < a)] = "last 5 min"
        lab[(r["t"] >= b) & (r["t"] < b + 600)] = "after"
    lab[r["ictal"]] = "high-rate episode"
    return lab


def figures(rel, A):
    FIGS.mkdir(exist_ok=True)
    G = lambda x1: -x1**3 - 2 * x1**2 + 1 + I_EXT  # noqa: E731
    # 1. time series, seed 1
    r, q = RUNS[(1, UNT)], RUNS[(1, REF)]
    h = 3600.0
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 7.0), sharex=True)
    for x, col, lab in ((q, C_REF, "healthy (40 Hz)"), (r, C_UNT, "deafferented (36 Hz)")):
        m = x["t"] > SETTLE_S
        ax[0].plot(x["t"][m] / h, np.maximum(x["rE"][m], 0.3), color=col, lw=0.7, label=lab)
        ax[1].plot(x["t"][m] / h, x["kA"][m], color=col, lw=1.4)
        ax[2].plot(x["t"][m] / h, np.where(x["ictal"], np.nan, x["x0"])[m], color=col, lw=1.4)
        ax[3].plot(x["t"][m] / h, x["z"][m], color=col, lw=1.1)
    ax[1].axhline(K_C, color=INK, ls="--", lw=0.9); ax[1].text(0.15, K_C + 0.02, "$k_E^c$", fontsize=9)
    ax[2].axhline(X0_CRIT, color=INK, ls="--", lw=0.9); ax[2].text(0.15, X0_CRIT + 0.08, "$x_{0,crit}$", fontsize=9)
    ax[3].axhline(Z_SN, color=INK, ls=":", lw=0.9)
    ax[0].set_yscale("log"); ax[0].set_ylabel("E rate (Hz)"); ax[0].legend(frameon=False, fontsize=8, ncol=2)
    ax[1].set_ylabel("$k_E$"); ax[2].set_ylabel("$x_0$"); ax[3].set_ylabel("$z$"); ax[3].set_xlabel("time (h, model)")
    for a in ax:
        for a_, b_ in r["seizures"]:
            a.axvspan(a_ / h, b_ / h, color=RED, alpha=0.25, lw=0)
    fig.tight_layout(); fig.savefig(FIGS / "sg_timeseries.png", dpi=200); plt.close(fig)

    # 2. trajectory (x0, z), all seeds + phase plane at representative x0
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    x1 = np.linspace(-2.7, -0.7, 300)
    a1.plot(x1, G(x1), color=INK, lw=1.8)
    for v, col in ((-3.5, "#2a78d6"), (-2.7, "#5b9be0"), (-2.2, "#9a9890"), (X0_CRIT, RED)):
        a1.plot(x1, 4 * (x1 - v), color=col, lw=1.2, label=f"$x_0$ = {v:.2f}")
    x0r = float(np.median(q["x0"][q["t"] > SETTLE_S]))
    a1.plot(x1, 4 * (x1 - x0r), color=C_REF, lw=1.4, ls="--", label=f"healthy, $x_0$ = {x0r:.2f}")
    a1.plot(X1_SN, Z_SN, "o", color=RED, ms=5)
    a1.set_xlim(-2.7, -0.7); a1.set_ylim(2.3, 6.5); a1.set_xlabel("$x_1$"); a1.set_ylabel("$z$")
    a1.legend(frameon=False, fontsize=7, loc="upper right"); a1.set_title("(a) slow nullclines along the cycle", loc="left", fontsize=10)
    for s in SEEDS:
        x = RUNS[(s, UNT)]; lab = phase_labels(x); keep = x["t"] > SETTLE_S
        for p_, col in PH:
            k = keep & (lab == p_)
            a2.plot(x["x0"][k], x["z"][k], ".", ms=1.3, color=col, label=p_ if s == SEEDS[0] else None)
        y = RUNS[(s, REF)]; k = y["t"] > SETTLE_S
        a2.plot(y["x0"][k], y["z"][k], ".", ms=3, color=C_REF, label="healthy (40 Hz)" if s == SEEDS[0] else None)
    a2.axvline(X0_CRIT, color=INK, ls="--", lw=0.9); a2.axhline(Z_SN, color=INK, ls=":", lw=0.9)
    a2.set_xlabel("$x_0$"); a2.set_ylabel("$z$"); a2.set_xlim(-4.0, -1.95)
    a2.legend(frameon=False, fontsize=7, markerscale=5, loc="upper left")
    a2.set_title("(b) trajectory, 5 seeds", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "sg_trajectory.png", dpi=200); plt.close(fig)

    # 3. onset-aligned x0
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for row in A:
        ax[0].plot(rel / 60, row, color=C_UNT, lw=0.7, alpha=0.7)
    ax[0].axhline(X0_CRIT, color=INK, ls="--", lw=0.9); ax[0].axvline(0, color=RED, lw=1)
    ax[0].set_xlabel("time from onset (min)"); ax[0].set_ylabel("$x_0$")
    ax[0].set_title(f"(a) $x_0$ around {len(A)} onsets", loc="left", fontsize=10)
    ax[1].plot(rel / 60, np.nanstd(A, 0), color=INK, lw=1.2)
    ax[1].set_xlabel("time from onset (min)"); ax[1].set_ylabel("SD across onsets")
    ax[1].set_title("(b) spread of $x_0$", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "sg_onsets.png", dpi=200); plt.close(fig)
    print("wrote figs/sg_timeseries.png, sg_trajectory.png, sg_onsets.png")


if __name__ == "__main__":
    main()
