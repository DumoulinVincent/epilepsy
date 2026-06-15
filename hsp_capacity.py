"""
hsp_capacity.py
===============
Task-based input-processing capacity for the HSP network, following the
reservoir-computing framework:

  - linear memory capacity (Jaeger 2002): how much of the recent input stream
    a linear readout can reconstruct from the population state, summed over
    delays. Bounded by the number of linearly independent state variables.
  - a degree-2 nonlinear capacity (in the spirit of Dambre et al. 2012,
    Sci Rep 2:514): how much NONLINEAR processing of past inputs the state
    supports. Linear + nonlinear = information processing capacity (IPC).

This is the metric that directly means "capacity to process inputs", so it
carries the capacity claim that integration / LZ could not.

Two layers, like the other modules:
  * pure-numpy capacity math  -> validated on a synthetic reservoir, no Brian2
  * a Brian2 driver that injects a fluctuating input and reads the state

Key modelling choices (state them in the methods section):
  * input drive  : rate[k] = r0 + amp * u[k], u[k] ~ iid Uniform[-1,1],
                   piecewise-constant over steps of `step_s`. r0 sets the
                   operating point (your "input frequency").
  * state        : E-population spike counts per step (K x N_E).
  * HSP frozen   : during the probe the homeostatic time constants are set to
                   ~infinity so the reservoir is stationary -> capacity of the
                   configuration the network reached at r0, not a moving target.
  * readout      : ridge regression; capacity per task = squared correlation,
                   thresholded against shuffled-input surrogates.
"""

import numpy as np


# ----------------------------------------------------------------------------
# 1. Input signal
# ----------------------------------------------------------------------------
def make_input_signal(n_steps, seed=0, kind="uniform"):
    """i.i.d. scalar input stream u[k] in [-1, 1]."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(-1.0, 1.0, size=n_steps)
    if kind == "binary":
        return rng.choice([-1.0, 1.0], size=n_steps)
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# 2. Capacity for a single target (ridge readout, squared correlation)
# ----------------------------------------------------------------------------
def _fit_capacity(X, y, ridge):
    """Squared correlation between the best ridge-linear prediction of y from X
    and y itself. X: (n, p) features, y: (n,) target. Both centered here."""
    Xc = X - X.mean(0, keepdims=True)
    yc = y - y.mean()
    p = Xc.shape[1]
    A = Xc.T @ Xc + ridge * np.eye(p)
    try:
        w = np.linalg.solve(A, Xc.T @ yc)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, Xc.T @ yc, rcond=None)[0]
    yhat = Xc @ w
    den = (yhat @ yhat) * (yc @ yc)
    if den <= 0:
        return 0.0
    return float(max((yhat @ yc) ** 2 / den, 0.0))


def _surrogate_threshold(X, n_samples, ridge, n_surrogate=20, q=99,
                         rng=None):
    """Noise floor: capacity recovered for RANDOM targets uncorrelated with the
    input. Tasks below this are set to 0 (Dambre-style thresholding)."""
    if rng is None:
        rng = np.random.default_rng(0)
    caps = [_fit_capacity(X, rng.standard_normal(n_samples), ridge)
            for _ in range(n_surrogate)]
    return float(np.percentile(caps, q))


# ----------------------------------------------------------------------------
# 3. Linear memory capacity (Jaeger)
# ----------------------------------------------------------------------------
def _lag(u, k_idx, d):
    """u[k-d] aligned to step indices k_idx."""
    return u[k_idx - d]


def linear_memory_capacity(X, u, max_delay=40, washout=50, ridge=1e-3,
                           threshold=None, rng=None):
    """
    X : (K, N) reservoir state per step.  u : (K,) input stream.
    Returns dict:
        mc_total   : sum over delays d>=1 (pure memory)
        mc_with_d0 : sum over delays d>=0 (incl. instantaneous read-out)
        delays     : array of d
        curve      : capacity at each delay (the forgetting curve)
        threshold  : noise floor used
        n_state    : N (capacity is bounded by this)
    """
    X = np.asarray(X, float)
    u = np.asarray(u, float)
    K, N = X.shape
    if rng is None:
        rng = np.random.default_rng(0)
    k_idx = np.arange(washout + max_delay, K)        # valid step indices
    Xv = X[k_idx]
    if threshold is None:
        threshold = _surrogate_threshold(Xv, len(k_idx), ridge, rng=rng)
    delays = np.arange(0, max_delay + 1)
    curve = np.zeros(len(delays))
    for i, d in enumerate(delays):
        cap = _fit_capacity(Xv, _lag(u, k_idx, d), ridge)
        curve[i] = cap if cap >= threshold else 0.0
    return dict(delays=delays, curve=curve,
                mc_with_d0=float(curve.sum()),
                mc_total=float(curve[1:].sum()),
                threshold=float(threshold), n_state=int(N))


# ----------------------------------------------------------------------------
# 4. Degree-2 nonlinear capacity (subset of full IPC)
# ----------------------------------------------------------------------------
def _legendre2(x):
    """2nd Legendre polynomial, the degree-2 orthonormal basis fn for U[-1,1]."""
    return 0.5 * (3.0 * x ** 2 - 1.0)


def nonlinear_capacity(X, u, max_delay=12, washout=50, ridge=1e-3,
                       threshold=None, rng=None):
    """
    Degree-2 capacity: quadratic self-terms P2(u[k-d]) and bilinear cross-terms
    u[k-d1]*u[k-d2]. Sum of thresholded capacities. This is the 'can it do
    NONLINEAR computation on its inputs' axis (memory-nonlinearity trade-off).
    Not the full Dambre basis (which continues to higher degree) but the
    dominant nonlinear contribution and a faithful, extensible subset.
    """
    X = np.asarray(X, float)
    u = np.asarray(u, float)
    K, N = X.shape
    if rng is None:
        rng = np.random.default_rng(1)
    k_idx = np.arange(washout + max_delay, K)
    Xv = X[k_idx]
    if threshold is None:
        threshold = _surrogate_threshold(Xv, len(k_idx), ridge, rng=rng)

    cap_self = 0.0
    for d in range(max_delay + 1):
        c = _fit_capacity(Xv, _legendre2(_lag(u, k_idx, d)), ridge)
        cap_self += c if c >= threshold else 0.0

    cap_cross = 0.0
    for d1 in range(max_delay + 1):
        for d2 in range(d1 + 1, max_delay + 1):
            tgt = _lag(u, k_idx, d1) * _lag(u, k_idx, d2)
            c = _fit_capacity(Xv, tgt, ridge)
            cap_cross += c if c >= threshold else 0.0

    return dict(cap_self=float(cap_self), cap_cross=float(cap_cross),
                nonlinear_total=float(cap_self + cap_cross),
                threshold=float(threshold))


def capacity_report(X, u, max_delay=40, nl_max_delay=12, washout=50,
                    ridge=1e-3, rng=None):
    """Linear + nonlinear in one call, plus the state rank (the capacity bound)."""
    if rng is None:
        rng = np.random.default_rng(0)
    lin = linear_memory_capacity(X, u, max_delay=max_delay, washout=washout,
                                 ridge=ridge, rng=rng)
    nl = nonlinear_capacity(X, u, max_delay=nl_max_delay, washout=washout,
                            ridge=ridge, rng=rng)
    Xc = np.asarray(X, float)[washout:]
    Xc = Xc - Xc.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    rank = int((sv > sv.max() * 1e-6).sum()) if sv.size else 0
    return dict(linear_mc=lin["mc_total"], linear_mc_with_d0=lin["mc_with_d0"],
                nonlinear=nl["nonlinear_total"],
                ipc_total=lin["mc_total"] + nl["nonlinear_total"],
                state_rank=rank, n_state=lin["n_state"],
                forget_curve=lin["curve"], delays=lin["delays"])


# ----------------------------------------------------------------------------
# 5. Brian2 driver: inject fluctuating input, read population state
# ----------------------------------------------------------------------------
def run_reservoir_probe(net, u, step_s=0.2, r0_hz=40.0, amp_hz=15.0,
                        washout_s=20.0, adapt_s=0.0, freeze_hsp=True,
                        benzo=False, Nablocker=False, na_vthr_mV=-48.0,
                        na_refrac_ms=3.0, benzo_kGABA=1.0, alpha=1.0):
    """
    Drive the network with rate[k] = r0_hz + amp_hz*u[k] (clipped >0), holding
    each value for step_s, and return the per-step E-population state.

    adapt_s    : let HSP adapt to r0 before probing (run at constant r0).
    freeze_hsp : set HSP time constants ~inf during the probe so the reservoir
                 is stationary (recommended; measures capacity AT this operating
                 point). The adaptation the network already did is preserved.

    Returns dict(u, X, step_s, r0_hz, t_drive_start, n_neurons).
    NOTE: driven step-by-step via repeated net.run(step_s) so it works with the
    existing PoissonGroup and the store/restore pattern without structural
    changes. This is the slow-but-robust path; for long probes use the
    cpp_standalone speedup discussed earlier.
    """
    from brian2 import second, Hz, mV, ms

    u = np.asarray(u, float)
    net.restore('baseline')
    inp = net['inp']; P_E = net['P_E']; PG_E = net['PG_E']; sp_E = net['sp_exc']

    PG_E.w[:] = alpha * float(PG_E.w[0])
    if Nablocker:
        P_E.V_thr_E = na_vthr_mV * mV
        P_E.tau_rp_E_var = na_refrac_ms * ms
    if benzo:
        P_E.k_GABA_E_hsp = benzo_kGABA

    global_ns = globals().get("_MODEL_NS", None)
    run_kw = dict(namespace=global_ns) if global_ns else {}

    # adapt to operating point r0 with HSP active
    inp.rates = r0_hz * Hz
    if adapt_s > 0:
        net.run(adapt_s * second, **run_kw)

    # freeze HSP for a stationary reservoir during the probe
    if freeze_hsp:
        for nm in ["tau_ins_AMPA_E", "tau_int_AMPA_E",
                   "tau_ins_GABA_E", "tau_int_GABA_E"]:
            setattr(P_E, nm, 1e99 * second)

    # washout at mean drive
    if washout_s > 0:
        net.run(washout_s * second, **run_kw)

    t_drive_start = float(net.t / second)
    rates = np.clip(r0_hz + amp_hz * u, 0.0, None)
    for k in range(len(u)):                       # one chunk per input step
        inp.rates = rates[k] * Hz
        net.run(step_s * second, **run_kw)

    # bin E spikes into the K input steps
    si = np.asarray(sp_E.i); st = np.asarray(sp_E.t / second)
    n_E = int(P_E.N)
    edges = t_drive_start + np.arange(len(u) + 1) * step_s
    win = (st >= edges[0]) & (st < edges[-1])
    si, st = si[win], st[win]
    X = np.zeros((len(u), n_E))
    for nrn in range(n_E):
        ts = st[si == nrn]
        if ts.size:
            X[:, nrn], _ = np.histogram(ts, bins=edges)
    return dict(u=u, X=X, step_s=step_s, r0_hz=r0_hz,
                t_drive_start=t_drive_start, n_neurons=n_E)


# ----------------------------------------------------------------------------
# 6. Capacity sweep across operating points / conditions
# ----------------------------------------------------------------------------
def capacity_sweep(net, r0_list_hz, condition="normal", n_steps=2500,
                   step_s=0.2, amp_hz=15.0, adapt_s=3000.0, signal_seed=0,
                   max_delay=40, **probe_kw):
    """
    Measure IPC at each operating point r0 (your 'input frequency'), so you can
    test the edge-of-chaos prediction: does capacity peak where integration /
    the seizure edge sits? Returns arrays aligned to r0_list_hz.
    """
    flags = dict(Nablocker=(condition == "nablocker"),
                 benzo=(condition == "benzo"))
    flags.update(probe_kw)
    u = make_input_signal(n_steps, seed=signal_seed)
    rng = np.random.default_rng(0)

    keys = ["linear_mc", "nonlinear", "ipc_total", "state_rank"]
    res = {k: [] for k in keys}
    for r0 in r0_list_hz:
        print(f"[{condition}] probing r0 = {r0:.1f} Hz")
        out = run_reservoir_probe(net, u, step_s=step_s, r0_hz=float(r0),
                                  amp_hz=amp_hz, adapt_s=adapt_s, **flags)
        rep = capacity_report(out["X"], out["u"], max_delay=max_delay, rng=rng)
        for k in keys:
            res[k].append(rep[k])
    res = {k: np.asarray(v, float) for k, v in res.items()}
    res.update(r0_hz=np.asarray(r0_list_hz, float), condition=condition)
    return res


# ----------------------------------------------------------------------------
# 7. FAST PATH: snapshot a settled interictal state, probe from it
# ----------------------------------------------------------------------------
def snapshot_interictal(net, input_freq_hz, condition="normal",
                        max_scan_s=8000.0, min_scan_s=2000.0, chunk_s=100.0,
                        low_factor=2.5, settle_s=400.0, target_hz=2.0,
                        store_name="probe_point", to_file=None, alpha=1.0,
                        na_vthr_mV=-48.0, na_refrac_ms=3.0, benzo_kGABA=1.0):
    """
    Run the network at `input_freq_hz` under `condition` (HSP ACTIVE) and store
    a snapshot once it has been in a settled interictal LOW phase for
    `settle_s` seconds -- i.e. the middle of an interictal interval, furthest
    from a burst. One snapshot per (condition, input frequency).

    Detection uses the slow r_est trace (neuron 45): 'low' means r_est stays
    below low_factor*target for settle_s continuously.
        - stable / low-activity regimes  -> low immediately, snapshot is quick
        - bursting regimes               -> waits for a between-burst stretch

    Returns dict(found, store_name, t_snapshot, k_AMPA_mean, k_GABA_mean,
                 status). Restore later with net.restore(store_name) (or from
                 `to_file`) and call probe_from_snapshot.
    """
    from brian2 import second, Hz, mV, ms

    net.restore('baseline')
    inp = net['inp']; P_E = net['P_E']; PG_E = net['PG_E']
    r_est_mon = net['r_est']; kA = net['k_AMPA_E_hsp']; kG = net['k_GABA_E_hsp']
    inp.rates = input_freq_hz * Hz
    PG_E.w[:] = alpha * float(PG_E.w[0])
    if condition == "nablocker":
        P_E.V_thr_E = na_vthr_mV * mV
        P_E.tau_rp_E_var = na_refrac_ms * ms
    if condition == "benzo":
        P_E.k_GABA_E_hsp = benzo_kGABA

    global_ns = globals().get("_MODEL_NS", None)
    run_kw = dict(namespace=global_ns) if global_ns else {}

    thr = low_factor * target_hz
    scanned = 0.0
    while scanned < max_scan_s:
        net.run(chunk_s * second, **run_kw)
        scanned += chunk_s
        if scanned < min_scan_s:          # warm-up: let the regime establish
            continue
        t = np.asarray(r_est_mon.t / second)
        r = np.asarray(r_est_mon.r_est[0] / Hz)
        tail = r[t >= (t[-1] - settle_s)]
        if tail.size and (t[-1] - t[0]) >= settle_s and np.all(tail < thr):
            if to_file:
                net.store(store_name, filename=to_file)
            else:
                net.store(store_name)
            return dict(found=True, store_name=store_name,
                        t_snapshot=float(t[-1]),
                        k_AMPA_mean=float(np.mean(kA.k_AMPA_E_hsp[:, -1])),
                        k_GABA_mean=float(np.mean(kG.k_GABA_E_hsp[:, -1])),
                        status="ok")
    return dict(found=False, store_name=None, t_snapshot=np.nan,
                k_AMPA_mean=np.nan, k_GABA_mean=np.nan,
                status="no settled interictal phase within max_scan_s")


def probe_from_snapshot(net, u, store_name="probe_point", to_file=None,
                        step_s=0.2, r0_hz=40.0, amp_hz=15.0, washout_s=10.0,
                        freeze_hsp=True, burst_rate_hz=50.0):
    """
    Restore a snapshot and run the IPC probe from it -- NO adaptation needed,
    because the snapshot is already the adapted interictal configuration.

    burst_rate_hz : if any probe step's population rate exceeds this, the frozen
                    config went unstable under the input (echo-state property
                    violated); the result is flagged burst_during_probe=True and
                    should be discarded / re-snapshotted earlier in the low phase.
    """
    from brian2 import second, Hz

    u = np.asarray(u, float)
    if to_file:
        net.restore(store_name, filename=to_file)
    else:
        net.restore(store_name)
    inp = net['inp']; P_E = net['P_E']; sp_E = net['sp_exc']
    r_est_mon = net['r_est']

    if freeze_hsp:
        for nm in ["tau_ins_AMPA_E", "tau_int_AMPA_E",
                   "tau_ins_GABA_E", "tau_int_GABA_E"]:
            setattr(P_E, nm, 1e99 * second)

    global_ns = globals().get("_MODEL_NS", None)
    run_kw = dict(namespace=global_ns) if global_ns else {}

    inp.rates = r0_hz * Hz
    if washout_s > 0:
        net.run(washout_s * second, **run_kw)

    t0 = float(net.t / second)
    rates = np.clip(r0_hz + amp_hz * u, 0.0, None)
    for k in range(len(u)):
        inp.rates = rates[k] * Hz
        net.run(step_s * second, **run_kw)

    si = np.asarray(sp_E.i); st = np.asarray(sp_E.t / second); n_E = int(P_E.N)
    edges = t0 + np.arange(len(u) + 1) * step_s
    win = (st >= edges[0]) & (st < edges[-1]); si, st = si[win], st[win]
    X = np.zeros((len(u), n_E))
    for nrn in range(n_E):
        ts = st[si == nrn]
        if ts.size:
            X[:, nrn], _ = np.histogram(ts, bins=edges)
    step_rate = X.sum(axis=1) / (n_E * step_s)
    burst = bool(np.any(step_rate > burst_rate_hz))

    # r_est (neuron 45) response over the probe window, for visual inspection
    rt = np.asarray(r_est_mon.t / second)
    rmask = rt >= t0
    r_est_t = rt[rmask] - t0
    r_est_val = np.asarray(r_est_mon.r_est[0] / Hz)[rmask]

    return dict(u=u, X=X, step_s=step_s, r0_hz=r0_hz, amp_hz=amp_hz,
                n_neurons=n_E, burst_during_probe=burst,
                max_step_rate_hz=float(step_rate.max()),
                step_rate_hz=step_rate, rates_hz=rates,
                r_est_t=r_est_t, r_est=r_est_val, t0=t0)


def capacity_at_point(net, input_freq_hz, condition="normal", n_steps=2000,
                      step_s=0.2, amp_hz=15.0, max_delay=40, signal_seed=0,
                      snapshot_kw=None, probe_kw=None):
    """
    One (condition, input frequency) -> snapshot the interictal low phase, probe
    from it, return the capacity report. The fast per-point routine for the
    edge-of-chaos sweep. r0 for the probe defaults to input_freq_hz.
    """
    snapshot_kw = snapshot_kw or {}
    probe_kw = probe_kw or {}
    u = make_input_signal(n_steps, seed=signal_seed)
    snap = snapshot_interictal(net, input_freq_hz, condition=condition,
                               **snapshot_kw)
    if not snap["found"]:
        return dict(found=False, status=snap["status"], r0_hz=input_freq_hz,
                    condition=condition)
    out = probe_from_snapshot(net, u, store_name=snap["store_name"],
                              r0_hz=input_freq_hz, amp_hz=amp_hz,
                              step_s=step_s, **probe_kw)
    rep = capacity_report(out["X"], out["u"], max_delay=max_delay)
    rep.update(found=True, condition=condition, r0_hz=input_freq_hz,
               burst_during_probe=out["burst_during_probe"],
               t_snapshot=snap["t_snapshot"],
               k_AMPA_mean=snap["k_AMPA_mean"], k_GABA_mean=snap["k_GABA_mean"])
    return rep


# ----------------------------------------------------------------------------
# 8. Amplitude calibration
# ----------------------------------------------------------------------------
def suggest_amp(results, sat_frac=0.95):
    """
    Pick the probe amplitude from a calibration run: among the STABLE amps
    (no burst during probe), choose the smallest whose d=0 capacity has reached
    `sat_frac` of the best stable d=0 -- i.e. the input is well-encoded but the
    swing is no larger than necessary. Returns (amp, reason).
    """
    stable = [r for r in results if not r["burst"]]
    if not stable:
        return None, "no stable amplitude found; lower the range or re-snapshot"
    d0max = max(r["d0"] for r in stable)
    if d0max <= 0:
        return None, "input not encoded at any amplitude; raise the range"
    ok = [r for r in stable if r["d0"] >= sat_frac * d0max]
    chosen = min(ok, key=lambda r: r["amp"])
    return chosen["amp"], (f"smallest stable amp reaching {sat_frac:.0%} of max "
                           f"d0 ({d0max:.2f}); IPC={chosen['ipc']:.2f}")


def calibrate_amplitude(net, input_freq_hz=40.0, condition="normal",
                        amps_hz=(2.0, 4.0, 6.0, 8.0, 12.0, 16.0),
                        n_steps=1500, step_s=0.05, max_delay=40, signal_seed=0,
                        snapshot_kw=None, probe_kw=None):
    """
    Calibrate the probe amplitude at one operating point. Snapshots ONCE, then
    probes that same frozen configuration at each amp (only amp changes), so the
    comparison is clean. For each amp returns the r_est response trace, the
    injected per-step rate, stability, d=0 capacity, linear/nonlinear/IPC.

    Use suggest_amp() on the result, eyeball the traces (smooth tracking = good,
    bursts = amp too high, input lost in noise = amp too low), pick a value, and
    FREEZE it for the whole condition sweep so cross-condition IPC is comparable.
    """
    snapshot_kw = dict(min_scan_s=2000.0, settle_s=400.0, **(snapshot_kw or {}))
    probe_kw = probe_kw or {}
    snap = snapshot_interictal(net, input_freq_hz, condition=condition,
                               **snapshot_kw)
    if not snap["found"]:
        return dict(found=False, status=snap["status"])
    u = make_input_signal(n_steps, seed=signal_seed)
    results = []
    for amp in amps_hz:
        out = probe_from_snapshot(net, u, store_name=snap["store_name"],
                                  r0_hz=input_freq_hz, amp_hz=float(amp),
                                  step_s=step_s, **probe_kw)
        rep = capacity_report(out["X"], out["u"], max_delay=max_delay)
        results.append(dict(amp=float(amp), d0=float(rep["forget_curve"][0]),
                            linear_mc=rep["linear_mc"], nonlinear=rep["nonlinear"],
                            ipc=rep["ipc_total"], burst=out["burst_during_probe"],
                            max_step_rate_hz=out["max_step_rate_hz"],
                            r_est_t=out["r_est_t"], r_est=out["r_est"],
                            t0=out["t0"], rates_hz=out["rates_hz"], step_s=step_s))
    amp_star, reason = suggest_amp(results)
    return dict(found=True, snapshot=snap, results=results, u=u,
                input_freq_hz=input_freq_hz, condition=condition,
                suggested_amp=amp_star, reason=reason)