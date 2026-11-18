"""Reduced event model of the deafferented (36 Hz) network: HSP + rate sensor + seizures as random events.

Calibrated on the spiking network (no free parameter besides the time constants being explored):
  - rest firing r_rest(k_E): frozen-gain up-sweeps at 36 Hz (about 1.1-1.3 Hz);
  - high-state firing r_high(k_E): frozen-gain down-sweeps (130 Hz at 0.76 to 286 Hz at 0.98); the high state exists
    only for k_E >= K_LOW = 0.76;
  - escape hazard h(k_E) from rest: frozen per-cell cycle gains (frozen_gain.py cycle_gains), log-linear fit through
    0.95-0.97 (doubles every ~0.007 in k_E), extrapolated, capped at H_MAX above K_EDGE (rest lost);
  - HSP: the network's equations for the E-mean gains, with the per-cell sigmoid gates applied to the E-mean rate
    estimate. "up" = AMPA insertion and GABA internalization (k_E rises), "down" = AMPA internalization and GABA
    insertion (k_E falls); rate sensor r_hat relaxes to the current rate with tau_s.

A seizure is either ended after D s (the seizure clamp) or, with D = 0, lasts as long as the high state exists
(until HSP has brought k_E below K_LOW), as in the spiking network without a termination mechanism.

    python event_model.py validate      # compare with the spiking runs
    python event_model.py scan          # where do clusters come back periodically?
"""
import sys
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
K_LOW, K_EDGE, H_MAX = 0.76, 0.985, 0.05
_KH = np.array([0.95, 0.96, 0.97]); _HZ = np.array([0.11, 0.43, 0.81]) / 1000.0
HB, HA = np.polyfit(_KH, np.log(_HZ), 1)
KTAB = np.array([0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
RTAB = np.array([129.5, 160.9, 179.2, 204.1, 229.7, 243.9, 250.3, 256.9, 264.6, 272.9, 281.2, 286.4, 286.4])


@njit(cache=True)
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def simulate(T, tau_up, tau_dn, tau_s, D, k0, kG0, r0, seed, ktab, rtab, ha, hb, rec_every=10, dk_post=0.0,
             tau_post=1800.0):
    """1 s steps. Returns (k_E every rec_every s, onset times, seizure end times).
    dk_post, tau_post: short-term post-seizure excitability (e.g. chloride loading weakening GABA): each seizure end adds
    dk_post to the k_E seen by the hazard, decaying with tau_post (0 = HSP only)."""
    np.random.seed(seed)
    kA, kG, rh = k0, kG0, r0
    n_rec = (T + rec_every - 1) // rec_every
    krec = np.empty(n_rec); ons = np.empty(200000); ends = np.empty(200000)
    n_on = 0; n_end = 0; seizing = False; left = 0; u = 0.0
    for t in range(T):
        if t % rec_every == 0:
            krec[t // rec_every] = kA
        if seizing:
            r = np.interp(kA, ktab, rtab)
            if D > 0:
                left -= 1
                if left <= 0:
                    seizing = False
            elif kA < 0.76:
                seizing = False
            if not seizing:
                ends[n_end] = t; n_end += 1
                u += dk_post
        else:
            r = 0.85 + 0.45 * kA
            h = np.exp(ha + hb * (kA + u))
            if kA + u >= 0.985 or h > 0.05:
                h = 0.05
            if np.random.random() < 1.0 - np.exp(-h):
                seizing = True; left = D
                ons[n_on] = t; n_on += 1
                r = np.interp(kA, ktab, rtab)
        rh = r + (rh - r) * np.exp(-1.0 / tau_s)
        u *= np.exp(-1.0 / tau_post)
        gu = _sig(15.0 * ((2.0 - rh) / 2.0 - 0.5))
        gd = _sig(15.0 * ((rh - 2.0) / 2.0 - 0.5))
        kA += (1.0 - kA) * gu / tau_up - kA * gd / tau_dn
        kG += -kG * gu / tau_up + (1.0 - kG) * gd / tau_dn
    return krec, ons[:n_on], ends[:n_end]


def run(T, tau_up, tau_dn, tau_s, D, k0=0.75, kG0=0.21, r0=2.0, seed=0, dk_post=0.0, tau_post=1800.0):
    return simulate(int(T), float(tau_up), float(tau_dn), float(tau_s), int(D), k0, kG0, r0, seed, KTAB, RTAB, HA, HB,
                    10, float(dk_post), float(tau_post))


def clusters(ons, gap):
    """Group onsets separated by less than gap s. Returns list of arrays."""
    if len(ons) == 0:
        return []
    cut = np.flatnonzero(np.diff(ons) > gap) + 1
    return np.split(ons, cut)


def validate():
    print(f"hazard fit: doubles every {np.log(2) / HB:.4f} in k_E; h(0.92) = {np.exp(HA + HB * 0.92) * 1e3:.4f} /1000 s")
    # 1. tau 250 s everywhere, sensor 100 s, no termination: spiking onsets at k_E 0.950-0.978 (mean 0.968),
    #    k_E min after an episode ~0.16, cycle ~6000-6500 s (onsets at ~4600 s, ~11000 s)
    for seed in range(3):
        k, on, en = run(20000, 250, 250, 100, 0, seed=seed)
        kon = np.interp(on, np.arange(len(k)) * 10, k)
        print(f"[tau250, no clamp] onsets {np.round(on).astype(int)}  k_on {np.round(kon, 3)}  "
              f"k_min after 1st {k[int(on[0]) // 10:].min() if len(on) else np.nan:.3f}")
    # 2-4. clamp, start 0.92 / 0.95, compare drop per seizure: spiking 0.1 (tau 2500, sensor 100), 0.010 (tau 25000,
    #      sensor 100), 0.0003 lone / 0.005-0.013 pairs (tau 25000, sensor 2500), 0.035-0.045 (tau 10000, sensor 1000)
    for tau, ts in ((2500, 100), (25000, 100), (25000, 2500), (10000, 1000)):
        drops, gaps = [], []
        for seed in range(20):
            k, on, en = run(100000, tau, tau, ts, 10, k0=0.95, kG0=0.045, r0=1.3, seed=seed)
            tk = np.arange(len(k)) * 10
            for i, o in enumerate(on):
                g = min([abs(o - x) for j, x in enumerate(on) if j != i] or [1e9])
                drops.append(np.interp(o - 10, tk, k) - np.interp(o + 3000, tk, k)); gaps.append(g)
        drops, gaps = np.array(drops), np.array(gaps)
        lone = drops[gaps > 3 * 3600]; pair = drops[gaps < 3600]
        print(f"[clamp tau {tau}, sensor {ts}] seizures/100000 s {len(drops) / 20:.1f}; drop lone (>3 h) "
              f"{np.median(lone) if len(lone) else np.nan:.4f}, pairs (<1 h) {np.median(pair) if len(pair) else np.nan:.4f}")


def scan():
    """Deafferentation from the 40 Hz fixed point, seizures ended after 10 s, 10^6 s, 5 seeds per setting."""
    T, D = 1_000_000, 10
    print(f"{'up':>6} {'down':>6} {'sensor':>6} | {'seiz':>5} {'clust':>5} {'size':>5} {'gap h':>6} {'CV':>5} "
          f"{'kmax':>6} {'%>edge':>6}")
    rows = []
    for tu in (5000, 10000, 25000):
        for td in (1000, 2500, 5000, 25000):
            if td > tu:
                continue
            for ts in (100, 1000, 2500, 5000):
                S = []
                for seed in range(5):
                    k, on, en = run(T, tu, td, ts, D, seed=seed)
                    on = on[on > 20000]
                    cl = clusters(on, gap=max(2 * ts, 3600))
                    sz = np.array([len(c) for c in cl]) if cl else np.array([0])
                    starts = np.array([c[0] for c in cl]) if cl else np.array([])
                    ici = np.diff(starts) / 3600
                    S.append((len(on), len(cl), sz.mean(), ici.mean() if len(ici) else np.nan,
                              ici.std() / ici.mean() if len(ici) > 1 else np.nan, k.max(), np.mean(k >= K_EDGE)))
                S = np.nanmean(np.array(S, float), axis=0)
                rows.append((tu, td, ts, *S))
                print(f"{tu:6d} {td:6d} {ts:6d} | {S[0]:5.0f} {S[1]:5.0f} {S[2]:5.2f} {S[3]:6.1f} {S[4]:5.2f} "
                      f"{S[5]:6.3f} {100 * S[6]:6.2f}")
    return rows


if __name__ == "__main__":
    {"validate": validate, "scan": scan}[sys.argv[1] if len(sys.argv) > 1 else "validate"]()
