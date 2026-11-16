"""IPC probe with the capacity split into encoding and memory.

Same probe as HSP_Phase._ipc_probe (restore the snapshot, freeze HSP, drive
the 100 input Poisson sources at r0 + amp * u[k], read E spike counts per
step) and the same capacity math as hsp_capacity.capacity_report (same
targets, ridge, cross-validation, surrogate thresholds and RNG order), but
every term is kept instead of only the sums:

  ipc_encoding = linear u[k] + P2(u[k])                  (d = 0: the input
                 during the same step)
  ipc_memory   = linear u[k-d], P2(u[k-d]) for d >= 1, and every cross term
                 u[k-d1] * u[k-d2] (d1 < d2, so it always involves the past)
  ipc_total    = the old definition, unchanged: linear d >= 1 + all degree-2
                 terms (so it contains P2(u[k]) but not linear u[k])

ipc_encoding + ipc_memory = every term = ipc_total + linear_d0.

The probe's operating point is set by the caller: coarse_phase passes each
network's own drive as r0 (conditions.py: probe_r0_hz=None), so the signal
rides on the drive the network adapted to instead of replacing it.
"""
import numpy as np

from hsp_capacity import (DEFAULT_RIDGES, _cv_splits, _lag, _legendre2, _oof_capacity,
                          _select_ridge, _surrogate_threshold, linear_memory_capacity)


def capacity_terms(X, u, max_delay=40, nl_max_delay=12, washout=50, ridges=DEFAULT_RIDGES,
                   n_folds=5):
    """capacity_report() with every thresholded term kept."""
    X = np.asarray(X, float)
    u = np.asarray(u, float)
    rng = np.random.default_rng(0)   # shared by both parts, as in capacity_report
    lin = linear_memory_capacity(X, u, max_delay=max_delay, washout=washout,
                                 ridges=ridges, n_folds=n_folds, rng=rng)
    curve = lin["curve"]

    # nonlinear_capacity(), term by term
    k_idx = np.arange(washout + nl_max_delay, len(X))
    Xv = X[k_idx]
    splits = _cv_splits(len(k_idx), n_folds, embargo=nl_max_delay)
    sel = [_legendre2(_lag(u, k_idx, d)) for d in range(min(6, nl_max_delay + 1))]
    ridge = _select_ridge(Xv, sel, ridges, splits)
    thr = _surrogate_threshold(Xv, len(k_idx), ridge, splits, rng=rng)

    def cap(target):
        c = _oof_capacity(Xv, target, ridge, splits)
        return c if c >= thr else 0.0

    p2 = np.array([cap(_legendre2(_lag(u, k_idx, d))) for d in range(nl_max_delay + 1)])
    cross = np.zeros((nl_max_delay + 1, nl_max_delay + 1))
    for d1 in range(nl_max_delay + 1):
        for d2 in range(d1 + 1, nl_max_delay + 1):
            cross[d1, d2] = cap(_lag(u, k_idx, d1) * _lag(u, k_idx, d2))

    Xc = X[washout:] - X[washout:].mean(0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.size and lam.sum() > 0 else np.nan
    rank = int((lam > lam.max() * 1e-12).sum()) if lam.size else 0
    return dict(ipc_total=float(curve[1:].sum() + p2.sum() + cross.sum()),
                ipc_encoding=float(curve[0] + p2[0]),
                ipc_memory=float(curve[1:].sum() + p2[1:].sum() + cross.sum()),
                linear_d0=float(curve[0]),
                linear_mc=float(curve[1:].sum()),
                nonlinear=float(p2.sum() + cross.sum()),
                forget_curve=curve, p2_curve=p2,
                state_rank=rank, participation_ratio=pr)


def ipc_probe(net, store_name, u, r0_hz, amp_hz, step_s=0.05, max_delay=40,
              rng_seed=None, burst_rate_hz=50.0):
    """HSP_Phase._ipc_probe with capacity_terms() as the readout."""
    from brian2 import second, Hz, seed as b2seed
    from HSP_Phase import _freeze_hsp, _runkw
    net.restore(store_name)
    _freeze_hsp(net)
    if rng_seed is not None:
        b2seed(int(rng_seed))
    inp, P_E, sp_E = net["inp"], net["P_E"], net["sp_exc"]
    rk = _runkw(net)
    n_E = int(P_E.N)
    t0 = float(net.t / second)
    rates = np.clip(r0_hz + amp_hz * np.asarray(u, float), 0.0, None)
    for k in range(len(u)):
        inp.rates = rates[k] * Hz
        net.run(step_s * second, **rk)
    i_sp = np.asarray(sp_E.i)
    t_sp = np.asarray(sp_E.t / second)
    edges = t0 + np.arange(len(u) + 1) * step_s
    win = (t_sp >= edges[0]) & (t_sp < edges[-1])
    X = np.zeros((len(u), n_E))
    for nrn in range(n_E):
        ts = t_sp[win][i_sp[win] == nrn]
        if ts.size:
            X[:, nrn], _ = np.histogram(ts, bins=edges)
    step_rate = X.sum(axis=1) / (n_E * step_s)
    rep = capacity_terms(X, u, max_delay=max_delay)
    rep.update(E_rate_probe=float(X.sum() / (n_E * len(u) * step_s)),
               burst_during_probe=bool(np.any(step_rate > burst_rate_hz)))
    return rep
