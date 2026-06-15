"""
hsp_metrics.py
==============
Information-theoretic "computational capacity" metrics + seizure-burden metrics
for the HSP cyclic-seizure spiking model.

The story this code is built around
-----------------------------------
Both failure modes collapse network *complexity*:
    - seizure / cyclic burst  -> hypersynchrony  -> high integration, low
      differentiation -> LOW complexity
    - heavy benzo / suppression -> little activity -> low entropy -> LOW complexity
    - healthy interictal      -> structured + variable -> complexity PEAKS

So we report, per input frequency and per drug condition:
    (a) interictal complexity  (TSE neural complexity, + entropy + multi-information)
    (b) seizure burden         (from r_est, matching your spectrogram intuition)
and optimise drug dose on the (burden, complexity) plane.

Two layers:
    1. Pure-numpy metric functions  -> testable without Brian2 (validated on
       synthetic data; see test_hsp_metrics.py).
    2. A Brian2 runner + sweep that capture the FULL excitatory population
       (your current sweep only stores neuron 45, which is not enough for a
       network metric).

Author: drafted for Vincent's thesis work. Natural-log units (nats) throughout.
"""

import numpy as np

# Gaussian differential-entropy constant per dimension: 0.5*log(2*pi*e)
_HALF_LOG_2PIE = 0.5 * np.log(2.0 * np.pi * np.e)

# ---------------------------------------------------------------------------
# Model physical constants — must match the notebook's global definitions.
# Brian2 resolves equation identifiers from the *calling frame* at net.run()
# time, so constants defined only in the notebook are invisible from here.
# We define them once and pass them as an explicit namespace to net.run().
# ---------------------------------------------------------------------------
def _build_model_namespace():
    from brian2 import second, Hz, mV, ms, nF, nS
    return dict(
        tau_HSP      = 100   * second,
        target_rate  = 2.0   * Hz,
        tau_vm       = 10    * second,
        tau_active   = 10    * second,
        V_L          = -70.  * mV,
        V_thr        = -50.  * mV,
        V_reset      = -60.  * mV,
        V_E          =   0.  * mV,
        V_I          = -70.  * mV,
        C_m_E        = 0.5   * nF,
        C_m_I        = 0.1   * nF,
        g_m_E        = 10.   * nS,
        g_m_I        =  3.   * nS,
        g_AMPA_ext_E = 5.5   * nS,
        g_AMPA_rec_E = 5.5   * nS,
        g_AMPA_ext_I = 5.5   * nS,
        g_AMPA_rec_I = 5.5   * nS,
        g_GABA_E     = 1.0   * nS,
        g_GABA_I     = 1.0   * nS,
        tau_AMPA     = 8.    * ms,
        tau_GABA     = 10.   * ms,
    )

_MODEL_NS = None  # built lazily on first run


# ----------------------------------------------------------------------------
# 0. Lempel-Ziv complexity  (BOUNDED differentiation axis)
# ----------------------------------------------------------------------------
# Why this and not only the Gaussian TSE: on continuous rates, "integration"
# grows without bound as neurons synchronise, so the Gaussian estimator reads
# HIGH at the seizure end. The clinical intuition -- complexity collapses in
# BOTH seizure (stereotyped/synchronous) and encephalopathy (suppressed) and
# peaks in rich interictal activity -- is a property of BOUNDED, discrete
# complexity. LZc on binarised population activity delivers exactly that, and
# is the measure used in the anaesthesia-depth / PCI / seizure-complexity
# literature. Use LZc (differentiation) and multi-information (integration)
# together: LZc separates the two failure modes; multi-information separates
# structured activity from noise.
def lz76_complexity(seq):
    """Lempel-Ziv (1976) parse count of a binary sequence (array of 0/1).
    O(n^2) worst case in pure Python -> only use for short sequences
    (<~20k). For population data use compression_complexity() instead."""
    s = ''.join('1' if x else '0' for x in np.asarray(seq).ravel())
    n = len(s)
    if n == 0:
        return 0
    i, c, l, k, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def _packed_compress_len(bits):
    import zlib
    b = np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()
    return len(zlib.compress(b, 9))


def compression_complexity(binary_matrix, n_shuffle=3, rng=None):
    """
    Fast, bounded differentiation metric (Lempel-Ziv-style via gzip).

    binary_matrix : (n_timebins, n_neurons) of 0/1.
    Concatenate population vectors across time, bit-pack, gzip, and normalise
    by the compressed length of phase-shuffled surrogates with the same mean
    activity. Returns ~[0, ~1]:
        ~0  : silence (suppression) OR stereotyped synchrony (seizure)
        ~1  : rich, varied activity (healthy interictal / noise)
    O(n) -> fine for the full interictal window.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    B = (np.asarray(binary_matrix) > 0).astype(np.uint8)
    seq = B.ravel(order='C')
    n = seq.size
    if n < 16:
        return np.nan
    p = seq.mean()
    if p in (0.0, 1.0):
        return 0.0
    c_obs = _packed_compress_len(seq)
    surr = []
    for _ in range(n_shuffle):
        s = seq.copy()
        rng.shuffle(s)
        surr.append(_packed_compress_len(s))
    c_rand = float(np.mean(surr))
    return float(c_obs / c_rand) if c_rand > 0 else np.nan


def normalized_lzc(binary_matrix, max_len=20000, rng=None):
    """
    Exact normalized LZ76 complexity. Bounded, same inverted-U as
    compression_complexity but slower; the concatenated sequence is uniformly
    subsampled to <= max_len to stay tractable. Prefer compression_complexity
    for routine sweeps; use this to confirm a few key points.
    """
    B = (np.asarray(binary_matrix) > 0).astype(int)
    seq = B.ravel(order='C')
    n = len(seq)
    if n < 10:
        return np.nan
    if n > max_len:
        idx = np.linspace(0, n - 1, max_len).astype(int)
        seq = seq[idx]
        n = max_len
    p = seq.mean()
    if p in (0.0, 1.0):
        return 0.0
    c = lz76_complexity(seq)
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    norm = (n / np.log2(n)) * H
    return float(c / norm) if norm > 0 else np.nan


# ----------------------------------------------------------------------------
# 1. Covariance estimation (robust to N_neurons > N_timebins)
# ----------------------------------------------------------------------------
def estimate_cov(X, method="ledoit-wolf", eps=1e-6):
    """
    X : (n_timebins, n_neurons) population activity matrix.
    Returns a well-conditioned covariance matrix.

    With 100 neurons and few interictal bins the raw covariance is rank
    deficient and det -> 0, which blows up entropy. Shrinkage fixes this.
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    n, p = X.shape
    if method == "ledoit-wolf":
        try:
            from sklearn.covariance import LedoitWolf
            return LedoitWolf(assume_centered=True).fit(X).covariance_
        except Exception:
            method = "diag"  # fall through
    C = np.cov(X, rowvar=False)
    C = np.atleast_2d(C)
    if method == "diag":
        C = C + eps * np.eye(C.shape[0])
    return C


# ----------------------------------------------------------------------------
# 2. Core information measures (multivariate-Gaussian approximation)
# ----------------------------------------------------------------------------
def gaussian_entropy(C):
    """Joint differential entropy of a Gaussian with covariance C (nats)."""
    C = np.atleast_2d(C)
    N = C.shape[0]
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        # numerically singular -> regularise
        C = C + 1e-9 * np.eye(N)
        sign, logdet = np.linalg.slogdet(C)
    return N * _HALF_LOG_2PIE + 0.5 * logdet


def multi_information(C):
    """
    Total correlation / integration:  I = sum_i H(x_i) - H(X)   (nats, >= 0).
    This is EXACTLY the collaborator's "sum of entropies minus the joint
    entropy". The (2*pi*e) terms cancel, so it only depends on the
    correlation structure.
    """
    C = np.atleast_2d(C)
    d = np.clip(np.diag(C), 1e-12, None)
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        C = C + 1e-9 * np.eye(C.shape[0])
        sign, logdet = np.linalg.slogdet(C)
    val = 0.5 * (np.sum(np.log(d)) - logdet)
    return float(max(val, 0.0))


def tse_complexity(C, sizes=None, n_samples=40, rng=None):
    """
    Tononi-Sporns-Edelman neural complexity (Gaussian estimator).

        C_TSE = sum_k [ (k/N) * I(X) - <I(X_subset of size k)> ]

    High ONLY when the system is both differentiated and integrated.
    -> low for independent noise, low for global synchrony, peaks in between.
    This is the inverted-U you optimise drug dose against.

    sizes     : which subset sizes k to evaluate (default: all 1..N).
    n_samples : random subsets averaged per k (k=1 and k=N are exact).
    """
    C = np.atleast_2d(C)
    N = C.shape[0]
    if rng is None:
        rng = np.random.default_rng(0)
    if sizes is None:
        sizes = range(1, N + 1)

    I_full = multi_information(C)
    total = 0.0
    for k in sizes:
        if k <= 1:
            mean_Ik = 0.0
        elif k >= N:
            mean_Ik = I_full
        else:
            vals = []
            reps = 1 if k == 1 else n_samples
            for _ in range(reps):
                idx = rng.choice(N, size=k, replace=False)
                vals.append(multi_information(C[np.ix_(idx, idx)]))
            mean_Ik = float(np.mean(vals))
        total += (k / N) * I_full - mean_Ik
    return float(total)


def population_information(X, cov_method="ledoit-wolf",
                          tse_sizes=None, tse_samples=40, rng=None):
    """
    Convenience: all four numbers from one population matrix X (timebins x neurons).
    Returns dict with joint entropy, summed marginal entropy, multi-information,
    and TSE complexity.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[0] < 3 or X.shape[1] < 2:
        return dict(H_joint=np.nan, H_marginal_sum=np.nan,
                    multi_information=np.nan, tse_complexity=np.nan,
                    n_bins=X.shape[0])
    C = estimate_cov(X, method=cov_method)
    d = np.clip(np.diag(C), 1e-12, None)
    H_marg = float(np.sum(_HALF_LOG_2PIE + 0.5 * np.log(d)))
    return dict(
        H_joint=float(gaussian_entropy(C)),
        H_marginal_sum=H_marg,
        multi_information=float(multi_information(C)),
        tse_complexity=float(tse_complexity(C, sizes=tse_sizes,
                                            n_samples=tse_samples, rng=rng)),
        n_bins=int(X.shape[0]),
    )


# ----------------------------------------------------------------------------
# 3. Spike binning
# ----------------------------------------------------------------------------
def bin_population(spike_i, spike_t, n_neurons, t0, t1, bin_s=0.05):
    """
    spike_i, spike_t : neuron indices and spike times (seconds), already
                       restricted to the analysis window if you like.
    Returns (X, bin_centres) with X shape (n_bins, n_neurons), counts per bin.
    """
    edges = np.arange(t0, t1 + bin_s, bin_s)
    if len(edges) < 2:
        return np.zeros((0, n_neurons)), np.array([])
    X = np.zeros((len(edges) - 1, n_neurons))
    spike_i = np.asarray(spike_i)
    spike_t = np.asarray(spike_t)
    for nrn in range(n_neurons):
        ts = spike_t[spike_i == nrn]
        if ts.size:
            X[:, nrn], _ = np.histogram(ts, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return X, centres


# ----------------------------------------------------------------------------
# 4. Interictal / ictal segmentation
# ----------------------------------------------------------------------------
def detect_ictal_bins(X, bin_s, target_hz=2.0, factor=4.0,
                      smooth_bins=3, dilate_bins=2):
    """
    Classify bins as ictal (burst) vs interictal from the population rate.

    X        : (n_bins, n_neurons) counts.
    factor   : a bin is 'ictal' if population rate exceeds factor*target_hz.
    dilate_bins : also exclude this many bins on each side of a burst, so the
                  rise/decay tails don't contaminate the interictal estimate.

    Returns boolean array `ictal` (True = burst).
    """
    n_bins, n_neurons = X.shape
    pop_rate = X.sum(axis=1) / (n_neurons * bin_s)  # Hz, network mean
    if smooth_bins > 1:
        k = np.ones(smooth_bins) / smooth_bins
        pop_rate = np.convolve(pop_rate, k, mode="same")
    ictal = pop_rate > (factor * target_hz)
    # dilate
    if dilate_bins > 0 and ictal.any():
        idx = np.where(ictal)[0]
        for j in idx:
            lo, hi = max(0, j - dilate_bins), min(n_bins, j + dilate_bins + 1)
            ictal[lo:hi] = True
    return ictal


def interictal_complexity(spike_i, spike_t, n_neurons, t0, t1,
                          bin_s=0.05, target_hz=2.0, factor=4.0,
                          min_bins=40, **info_kw):
    """
    End-to-end: bin spikes -> drop ictal bins -> Gaussian complexity on the
    interictal submatrix. Returns the info dict plus interictal fraction.
    """
    X, _ = bin_population(spike_i, spike_t, n_neurons, t0, t1, bin_s=bin_s)
    if X.shape[0] == 0:
        return dict(tse_complexity=np.nan, multi_information=np.nan,
                    H_joint=np.nan, H_marginal_sum=np.nan,
                    interictal_fraction=np.nan, n_bins=0)
    ictal = detect_ictal_bins(X, bin_s, target_hz=target_hz, factor=factor)
    Xi = X[~ictal]
    frac = float((~ictal).mean())
    if Xi.shape[0] < min_bins:
        out = dict(tse_complexity=np.nan, multi_information=np.nan,
                   H_joint=np.nan, H_marginal_sum=np.nan, lzc=np.nan)
        out["interictal_fraction"] = frac
        out["n_bins"] = int(Xi.shape[0])
        out["note"] = "too few interictal bins; lengthen run or coarsen bin_s"
        return out
    out = population_information(Xi, **info_kw)
    out["lzc"] = compression_complexity(Xi > 0, rng=info_kw.get("rng"))
    # rate / repertoire axis (flags SUPPRESSION; compression flags SYNCHRONY)
    out["mean_rate_hz"] = float(Xi.sum() / (Xi.shape[0] * bin_s * Xi.shape[1]))
    out["active_frac"] = float((Xi > 0).mean())
    counts = (Xi > 0).sum(axis=1).astype(int)          # co-active neurons / bin
    hist = np.bincount(counts, minlength=Xi.shape[1] + 1).astype(float)
    pmf = hist[hist > 0] / hist.sum()
    out["count_entropy_bits"] = float(-np.sum(pmf * np.log2(pmf)))
    out["interictal_fraction"] = frac
    return out


# ----------------------------------------------------------------------------
# 5. Seizure burden (from the r_est trace you already record)
# ----------------------------------------------------------------------------
def seizure_burden(t, r_est, target_hz=2.0, factor=4.0):
    """
    Simple, interpretable burden metrics from a single r_est(t) trace.
        ictal_fraction : fraction of time above factor*target
        n_bursts       : number of distinct excursions above threshold
        burst_rate_hz  : bursts per second
        mean_overshoot : mean r_est during ictal periods (Hz)
        lf_power       : integrated power of r_est fluctuations (matches the
                         bright band in your spectrogram); see note below.
    """
    t = np.asarray(t, dtype=float)
    r = np.asarray(r_est, dtype=float)
    if t.size < 4:
        return dict(ictal_fraction=np.nan, n_bursts=0, burst_rate_hz=np.nan,
                    mean_overshoot=np.nan, lf_power=np.nan)
    thr = factor * target_hz
    ictal = r > thr
    ictal_fraction = float(ictal.mean())
    # count rising edges
    edges = np.diff(ictal.astype(int))
    n_bursts = int(np.sum(edges == 1))
    duration = t[-1] - t[0]
    burst_rate = n_bursts / duration if duration > 0 else np.nan
    mean_overshoot = float(r[ictal].mean()) if ictal.any() else 0.0
    # low-frequency fluctuation power of the (mean-removed) trace
    rr = r - r.mean()
    lf_power = float(np.mean(rr ** 2))
    return dict(ictal_fraction=ictal_fraction, n_bursts=n_bursts,
                burst_rate_hz=float(burst_rate), mean_overshoot=mean_overshoot,
                lf_power=lf_power)


# ----------------------------------------------------------------------------
# 6. Brian2 runner that CAPTURES THE FULL E POPULATION
#    (mirrors your run_full_hsp_trajectory, but also returns spikes)
# ----------------------------------------------------------------------------
def run_trajectory_capture(net, alpha=1.0, run_time=None, analysis_start_s=2000.0,
                           input_freq=None,
                           tau_ins_AMPA=100.0, tau_int_AMPA=100.0,
                           tau_ins_GABA=100.0, tau_int_GABA=100.0,
                           neuron_idx=45, benzo=False, Nablocker=False,
                           na_vthr_mV=-48.0, na_refrac_ms=3.0,
                           benzo_kGABA=1.0):
    """
    Same control flow as run_full_hsp_trajectory, but returns the full
    excitatory spike train in the analysis window so we can compute network
    metrics. Pass brian2 Quantities for run_time / input_freq (e.g. 5000*second,
    30*Hz), or plain numbers (interpreted as seconds / Hz).

    Drug knobs are now GRADED so you can sweep dose:
        Nablocker: na_vthr_mV (threshold), na_refrac_ms (refractory)
        benzo    : benzo_kGABA in [baseline .. 1.0] (GABA scaling clamp)
    """
    from brian2 import second, Hz, mV, ms

    if run_time is None:
        run_time = 5000 * second
    elif not hasattr(run_time, "dimensions"):
        run_time = run_time * second
    if input_freq is None:
        input_freq = 40 * Hz
    elif not hasattr(input_freq, "dimensions"):
        input_freq = input_freq * Hz

    net.restore('baseline')
    inp = net['inp']
    inp.rates = input_freq
    P_E = net['P_E']
    PG_E = net['PG_E']
    r_est_mon = net['r_est']
    sp_E = net['sp_exc']
    kA = net['k_AMPA_E_hsp']
    kG = net['k_GABA_E_hsp']

    baseline_w = float(PG_E.w[0])
    PG_E.w[:] = alpha * baseline_w

    P_E.tau_ins_AMPA_E = tau_ins_AMPA * second
    P_E.tau_int_AMPA_E = tau_int_AMPA * second
    P_E.tau_ins_GABA_E = tau_ins_GABA * second
    P_E.tau_int_GABA_E = tau_int_GABA * second

    if Nablocker:
        P_E.V_thr_E = na_vthr_mV * mV
        P_E.tau_rp_E_var = na_refrac_ms * ms

    if benzo:
        P_E.k_GABA_E_hsp = benzo_kGABA
        P_E.tau_ins_GABA_E = 1e99 * second
        P_E.tau_int_GABA_E = 1e99 * second

    global _MODEL_NS
    if _MODEL_NS is None:
        _MODEL_NS = _build_model_namespace()
    net.run(run_time, report='text', namespace=_MODEL_NS)

    # r_est (neuron 45) on the monitor timebase
    t_all = np.asarray(r_est_mon.t / second)
    mask = t_all >= analysis_start_s
    t_re = t_all[mask]
    r_re = np.asarray(r_est_mon.r_est[0, mask] / Hz)

    # full E spikes in the analysis window
    si = np.asarray(sp_E.i)
    st = np.asarray(sp_E.t / second)
    win = st >= analysis_start_s
    n_E = int(P_E.N)

    return dict(
        t=t_re, r_est_trace=r_re,
        spike_i=si[win], spike_t=st[win],
        n_neurons=n_E,
        t0=analysis_start_s, t1=float(analysis_start_s + run_time / second),
        kE_trace=np.asarray(kA.k_AMPA_E_hsp[neuron_idx, mask]),
        kI_trace=np.asarray(kG.k_GABA_E_hsp[neuron_idx, mask]),
    )


# ----------------------------------------------------------------------------
# 7. Frequency sweep that returns burden + interictal complexity per frequency
# ----------------------------------------------------------------------------
def run_sweep_with_metrics(net, input_freqs_hz, condition="normal",
                           run_time_s=5000.0, analysis_start_s=2000.0,
                           alpha=1.0, bin_s=0.05, target_hz=2.0, factor=4.0,
                           tse_samples=30, dose=None, seed_val=123,
                           **drug_kw):
    """
    condition in {"normal","nablocker","benzo"}.
    dose : optional scalar recorded alongside results (for Pareto plots);
           for nablocker pass e.g. na_vthr_mV via drug_kw and dose=that value,
           for benzo pass benzo_kGABA via drug_kw and dose=that value.

    Returns dict of arrays aligned to input_freqs_hz, plus per-frequency traces.
    """
    from brian2 import seed as b2seed
    b2seed(seed_val)

    flags = dict(Nablocker=(condition == "nablocker"),
                 benzo=(condition == "benzo"))
    flags.update(drug_kw)

    rng = np.random.default_rng(0)
    freqs = np.asarray(input_freqs_hz, dtype=float)
    res = {k: [] for k in ["lzc", "multi_information", "count_entropy_bits",
                           "mean_rate_hz", "tse_complexity", "H_joint",
                           "interictal_fraction", "ictal_fraction",
                           "n_bursts", "burst_rate_hz", "lf_power",
                           "mean_overshoot"]}
    traces = {}

    for f in freqs:
        print(f"[{condition}] input = {f:.2f} Hz")
        out = run_trajectory_capture(
            net, alpha=alpha, run_time=run_time_s, input_freq=f,
            analysis_start_s=analysis_start_s, **flags)

        comp = interictal_complexity(
            out["spike_i"], out["spike_t"], out["n_neurons"],
            out["t0"], out["t1"], bin_s=bin_s, target_hz=target_hz,
            factor=factor, tse_samples=tse_samples, rng=rng)

        burd = seizure_burden(out["t"], out["r_est_trace"],
                              target_hz=target_hz, factor=factor)

        res["lzc"].append(comp.get("lzc", np.nan))
        res["multi_information"].append(comp.get("multi_information", np.nan))
        res["count_entropy_bits"].append(comp.get("count_entropy_bits", np.nan))
        res["mean_rate_hz"].append(comp.get("mean_rate_hz", np.nan))
        res["tse_complexity"].append(comp.get("tse_complexity", np.nan))
        res["H_joint"].append(comp.get("H_joint", np.nan))
        res["interictal_fraction"].append(comp.get("interictal_fraction", np.nan))
        for key in ["ictal_fraction", "n_bursts", "burst_rate_hz",
                    "lf_power", "mean_overshoot"]:
            res[key].append(burd[key])

        traces[float(f)] = dict(t=out["t"], r_est=out["r_est_trace"])

    res = {k: np.asarray(v, dtype=float) for k, v in res.items()}
    res.update(input_freqs_hz=freqs, condition=condition, dose=dose,
               traces=traces)
    return res