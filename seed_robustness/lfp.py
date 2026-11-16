"""LFP proxy for the HSP network (Mazzoni et al. 2008).

Population-mean RECURRENT synaptic currents onto EXCITATORY cells (AMPA from
E, GABA from I), rectified per population, then summed:
    lfp = |I_exc| + |I_inh|
Interneurons are excluded (stellate morphology, no open field). The external
Poisson drive (I_AMPA_ext) is excluded too: it is the input, not network
activity, and between seizures it made up ~90% of a proxy that included it,
burying the recurrent signal. Spikes, membrane voltage and injected current
(I_dbs) are not part of the proxy either.

Memory: the population mean is computed inside the simulation with summed
variables into a 1-unit group, so the monitor carries 2 traces rather than
one per neuron (2 x 1.5M samples x 8 B = 24 MB for 3000 s at 2 ms).

Model-specific adaptations of the spec:
  - defaultclock.dt is 1 ms in this model, so REC_DT = 2 ms is 2 steps.
    Nothing finer than 1 ms can be recorded; the aliasing check compares 1 ms
    against 2 ms (see lfp_checks.py).
  - The currents already exist as subexpressions of P_E (I_AMPA_rec,
    I_GABA_rec), so the equations are not modified at all.
  - HSP acts on the per-neuron gains k_AMPA_E_hsp / k_GABA_E_hsp, which
    multiply the recurrent conductances; synapse `w` is static during a run.
    The "weight trace" is therefore the E-population mean of those two gains.
  - The stored baselines contain no LFP objects and Network.restore() fails on
    objects it has no state for, so the probe is attached AFTER restore
    (add_lfp_probe) instead of being built into build_network_seeded(). The
    existing pipeline and its .bri files are untouched.
"""
import sys
from pathlib import Path

import numpy as np
from scipy import signal

THIS_DIR = Path(__file__).resolve().parent
for _p in (str(THIS_DIR.parent), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from brian2 import (NeuronGroup, Synapses, StateMonitor, Hz, ms, second,  # noqa: E402
                    start_scope, defaultclock, seed as b2seed)

import seeded_network  # noqa: E402

REC_DT = 2 * ms          # 500 Hz; band of interest < ~100 Hz
WEIGHT_DT = 1 * second   # HSP gains move on tau_HSP = 100 s
DT = 1 * ms              # the model's integration step (calibrate_baseline dt_us=1000)
FIT_BANDS = ((4.0, 12.0), (40.0, 70.0))


def model_namespace():
    """Constants the model equations reference. Brian2 resolves them from the
    frame that calls net.run(), and this module doesn't define them."""
    return {k: v for k, v in vars(seeded_network).items()
            if not k.startswith("_") and not callable(v)
            and not isinstance(v, type(sys))}


def add_lfp_probe(net, rec_dt=REC_DT, weight_dt=WEIGHT_DT):
    """Attach the LFP summed-variable probe and the HSP-gain monitor to `net`.
    Call after any net.restore() of a store made without the probe."""
    ratio = float(rec_dt / net["P_E"].clock.dt)
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"rec_dt={rec_dt} must be an integer multiple of the "
                         f"integration step {net['P_E'].clock.dt}")
    lfp = NeuronGroup(1, """I_exc : amp
                           I_inh : amp""", name="LFP")
    s_lfp = Synapses(net["P_E"], lfp, """
        I_exc_post = I_AMPA_rec_pre / N_pre : amp (summed)
        I_inh_post = I_GABA_rec_pre / N_pre : amp (summed)
        """, name="S_lfp")
    s_lfp.connect()
    m_lfp = StateMonitor(lfp, ["I_exc", "I_inh"], record=0, dt=rec_dt, name="M_LFP")
    m_w = StateMonitor(net["P_E"], ["k_AMPA_E_hsp", "k_GABA_E_hsp"], record=True,
                       dt=weight_dt, name="M_W")
    net.add(lfp, s_lfp, m_lfp, m_w)
    return m_lfp, m_w


def lfp_from_monitor(m_lfp):
    """Returns (t, lfp, lfp_exc, lfp_inh), all in amps except t (s), with
    lfp = |I_exc| + |I_inh|. Rectify each population, then sum: E and I
    currents are anti-correlated in a balanced network, so a signed sum would
    cancel the shared rhythmic component.

    The first sample is dropped: the monitor records before the first
    summed-variable update, so it's always the group's initial 0."""
    t = np.asarray(m_lfp.t / second)[1:]
    lfp_exc = np.abs(np.asarray(m_lfp.I_exc[0]))[1:]
    lfp_inh = np.abs(np.asarray(m_lfp.I_inh[0]))[1:]
    return t, lfp_exc + lfp_inh, lfp_exc, lfp_inh


def weights_from_monitor(m_w):
    """Returns (t, mean k_AMPA, mean k_GABA) over E cells."""
    t = np.asarray(m_w.t / second)
    return (t, np.asarray(m_w.k_AMPA_E_hsp).mean(axis=0),
            np.asarray(m_w.k_GABA_E_hsp).mean(axis=0))


def build_restored(seed, baseline_path, input_freq_hz, drug=None, weights=None,
                   with_probe=True, rng_seed=None):
    """Fresh network for `seed`, restored from its calibrated baseline, input
    set to `input_freq_hz`, condition applied, probe optionally attached.
    Returns (net, m_lfp, m_w) (monitors None if with_probe=False)."""
    import gc
    from HSP_Phase import _apply_drugs
    from interventions import apply_weight_intervention

    # Brian2 picks auto-names ("synapses_3") by avoiding LIVE objects' names;
    # a previous network still awaiting cyclic GC would shift them and break
    # restore() against the baseline file.
    gc.collect()
    start_scope()
    defaultclock.dt = DT
    net = seeded_network.build_network_seeded(
        50 * second, 50 * second, 50 * second, 50 * second,
        input_freq=input_freq_hz, network_seed=seed)
    net.restore("baseline", filename=str(baseline_path))
    net["inp"].rates = input_freq_hz * Hz
    if weights:
        apply_weight_intervention(net, weights["kind"], scale=weights.get("scale", 1.0))
    _apply_drugs(net, drug or {})
    m_lfp = m_w = None
    if with_probe:
        m_lfp, m_w = add_lfp_probe(net)
    if rng_seed is not None:
        b2seed(int(rng_seed))
    return net, m_lfp, m_w


def run_lfp(seed, baseline_path, input_freq_hz, duration_s=3000.0, drug=None,
            weights=None, rng_seed=None, report=None):
    """Run `duration_s` from the seed's baseline under a condition, recording
    the LFP proxy. Returns (lfp, weights, spikes_E, spikes_I):
        lfp      = (t, lfp, lfp_exc, lfp_inh)
        weights  = (t, mean k_AMPA, mean k_GABA)
        spikes_E = (i, t) for E spikes during the run
        spikes_I = (i, t) for I spikes during the run
    """
    net, m_lfp, m_w = build_restored(seed, baseline_path, input_freq_hz,
                                     drug=drug, weights=weights, rng_seed=rng_seed)
    t0 = float(net.t / second)
    net.run(duration_s * second, namespace=model_namespace(), report=report)
    spikes = []
    for name in ("sp_exc", "sp_inh"):
        st = np.asarray(net[name].t / second)
        keep = st >= t0
        spikes.append((np.asarray(net[name].i)[keep], st[keep]))
    return (lfp_from_monitor(m_lfp), weights_from_monitor(m_w), *spikes)


def binned_rate(spike_t, n_cells, t0, t1, bin_s=1.0):
    """Population rate (Hz per cell) in bins of `bin_s`; returns (centers, rate)."""
    edges = np.arange(t0, t1 + 1e-9, bin_s)
    counts, _ = np.histogram(spike_t, bins=edges)
    return edges[:-1] + bin_s / 2, counts / (n_cells * bin_s)


def psd_of(x, fs, nperseg_s=4.0):
    x = signal.detrend(np.asarray(x))
    nper = int(min(len(x), nperseg_s * fs))
    return signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2)


def aperiodic_fit(f, P, fit_bands=FIT_BANDS):
    """Straight line through log10(P) vs log10(f), fitted on `fit_bands` only
    (peak-free bands, well below the shot-noise floor). Returns
    (residual, slope, intercept); residual is log10(P) minus the line, at
    every f > 0 (NaN at f = 0)."""
    f = np.asarray(f, float)
    P = np.asarray(P, float)
    fit = np.zeros(f.shape, bool)
    for lo, hi in fit_bands:
        fit |= (f >= lo) & (f <= hi)
    fit &= (f > 0) & (P > 0)
    slope, intercept = np.polyfit(np.log10(f[fit]), np.log10(P[fit]), 1)
    residual = np.full(f.shape, np.nan)
    ok = (f > 0) & (P > 0)
    residual[ok] = np.log10(P[ok]) - (intercept + slope * np.log10(f[ok]))
    return residual, slope, intercept


def band_peak(f, residual, band, fit_bands=FIT_BANDS):
    """(peak frequency, residual height in log10 units) inside `band`, found
    on the aperiodic residual -- argmax of a raw 1/f spectrum would just
    return the bottom edge of the band. Refuses a band overlapping the fit
    bands, since the fit would then absorb the very peak being measured."""
    lo, hi = band
    for flo, fhi in fit_bands:
        if lo < fhi and flo < hi:
            raise ValueError(f"band {band} overlaps aperiodic fit band {(flo, fhi)}")
    m = (f >= lo) & (f < hi)   # half-open: a band ending where a fit band starts
    k = int(np.nanargmax(residual[m]))  # must not include that shared fit bin
    return float(f[m][k]), float(residual[m][k])
