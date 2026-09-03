"""
Seeded copy of the HSP network builder used in DBS_intervention.ipynb /
IPC_only.ipynb.

The notebooks' `build_network` has no seed argument -- network realizations
(recurrent E-E/I-E/I-I weight draws, and the Poisson input/DBS streams) are
whatever `numpy`'s global RNG state happens to be at call time (in practice,
whatever `seed(123)` in the notebook cell left it at). To get independent
network realizations we need a `network_seed` threaded into the builder,
which means duplicating the builder here rather than editing the notebook
(build_network can't be imported from a .ipynb, and the whole point of this
folder is to not touch the existing notebooks/HSP_Phase.py).

Everything below -- constants, equations, connectivity, monitors -- is a
line-for-line copy of `build_network`/`run_sim` from DBS_intervention.ipynb,
with a `network_seed` parameter added and threaded through `brian2.seed()`
(which reseeds numpy's global RNG, and therefore both the weight draws and
Brian's own Poisson/threshold RNG use) right before anything random happens.
Keep this in sync by hand if the notebooks' model equations change.
"""
from pathlib import Path

from brian2 import (
    NeuronGroup, PoissonGroup, Synapses, StateMonitor, SpikeMonitor,
    Network, collect, start_scope, defaultclock, seed as b2seed,
    mV, nF, nS, ms, second, Hz, pA, us,
)
import numpy as np

# ---------------------------------------------------------------------------
# Model constants (verbatim from DBS_intervention.ipynb / IPC_only.ipynb)
# ---------------------------------------------------------------------------
V_L = -70. * mV
V_thr = -50. * mV
V_reset = -60. * mV
V_E = 0.0 * mV
V_I = -70. * mV

C_m_E = 0.5 * nF
C_m_I = 0.1 * nF

g_m_E = 10. * nS
g_m_I = 3. * nS

tau_rp_E = 2. * ms
tau_rp_I = 1. * ms

g_AMPA_ext_E = 5.5 * nS
g_AMPA_rec_E = 5.5 * nS
g_AMPA_ext_I = 5.5 * nS
g_AMPA_rec_I = 5.5 * nS
tau_AMPA = 8. * ms

g_GABA_E = 1.0 * nS
g_GABA_I = 1.0 * nS
tau_GABA = 10. * ms

tau_HSP = 100 * second
target_rate = 2.0 * Hz
record_dt = 200 * ms

tau_vm = 10 * second
tau_active = 10 * second

recurrent_radius = 2

n_inh_rows, n_inh_cols = 5, 5
_inh_positions = []
for _iy in range(n_inh_rows):
    for _ix in range(n_inh_cols):
        _x = int((_ix + 0.5) * 10 / n_inh_cols)
        _y = int((_iy + 0.5) * 10 / n_inh_rows)
        _inh_positions.append((_x, _y))
inh_positions = _inh_positions


def build_network_seeded(tau_ins_AMPA, tau_int_AMPA, tau_ins_GABA, tau_int_GABA,
                         input_freq=40, network_seed=None):
    """Same network as build_network() in DBS_intervention.ipynb, but seeded.

    `network_seed` (if not None) reseeds brian2/numpy's RNG *before* the
    recurrent weight draws and Poisson group creation, so different seeds
    give independent network realizations (connectivity topology itself is
    deterministic grid geometry, only the weights and Poisson streams vary).
    """
    if network_seed is not None:
        b2seed(int(network_seed))

    n_input_rows = 10
    n_input_columns = 10
    n_output_rows = 10
    n_output_columns = 10
    N_inp = n_input_rows * n_input_columns
    N_E = n_output_rows * n_output_columns
    n_inh_rows, n_inh_cols = 5, 5
    N_I = n_inh_rows * n_inh_cols

    eqs_E = '''
    dv / dt = (- g_m_E * (v - V_L) - I_syn + I_dbs) / C_m_E : volt (unless refractory)
    I_syn = I_AMPA_ext + I_AMPA_rec + I_GABA_rec : amp

    # Homeostatically scaled synaptic currents
    I_AMPA_ext = g_AMPA_ext_E  * (v - V_E) * s_AMPA_ext : amp
    I_AMPA_rec = g_AMPA_rec_E * k_AMPA_E_hsp  * (v - V_E) * s_AMPA : amp
    I_GABA_rec = g_GABA_E * k_GABA_E_hsp * (v - V_I) * s_GABA     : amp

    dI_dbs / dt = -I_dbs / (2*ms)  : amp

    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
    ds_AMPA / dt      = - s_AMPA / tau_AMPA     : 1
    ds_GABA / dt      = - s_GABA / tau_GABA     : 1

    dx_spike / dt = -x_spike/tau_HSP : 1
    r_est = x_spike / tau_HSP : Hz

    # HSP scaling variables (dimensionless)
    dk_AMPA_E_hsp/dt = ((1 - k_AMPA_E_hsp) * 1/(1+exp(-15*((target_rate - r_est)/target_rate-0.5))) / tau_ins_AMPA_E
                - (k_AMPA_E_hsp) * 1/(1+exp(-15*((r_est - target_rate)/target_rate-0.5))) / tau_int_AMPA_E) : 1

    dk_GABA_E_hsp/dt = (-(k_GABA_E_hsp) * 1/(1+exp(-15*((target_rate - r_est)/target_rate-0.5))) / tau_int_GABA_E
                + (1 - k_GABA_E_hsp) * 1/(1+exp(-15*((r_est - target_rate)/target_rate-0.5))) / tau_ins_GABA_E) : 1

    dvm_lp / dt = (v - vm_lp)/tau_vm : volt
    dy_act / dt = -y_act / tau_active : 1

    tau_ins_AMPA_E : second (shared)
    tau_int_AMPA_E : second (shared)
    tau_int_GABA_E : second (shared)
    tau_ins_GABA_E : second (shared)
    V_thr_E : volt
    tau_rp_E_var : second
    '''

    eqs_I = '''
    dv / dt = (- g_m_I * (v - V_L) - I_syn) / C_m_I : volt (unless refractory)
    I_syn = I_AMPA_ext + I_AMPA_rec + I_GABA_rec : amp
    I_AMPA_ext = g_AMPA_ext_I * (v - V_E) * s_AMPA_ext : amp
    I_AMPA_rec = g_AMPA_rec_I * 0.3 * (v - V_E) * 1 * s_AMPA : amp
    ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : 1
    ds_AMPA / dt = - s_AMPA / tau_AMPA : 1

    I_GABA_rec = g_GABA_I * 0.3 * (v - V_I) * s_GABA : amp
    ds_GABA / dt = - s_GABA / tau_GABA : 1

    dx_spike / dt = -x_spike/tau_HSP : 1
    r_est = x_spike / tau_HSP : Hz

    dy_act / dt = -y_act / tau_active : 1
    '''

    P_E = NeuronGroup(N_E, eqs_E, threshold='v > V_thr_E',
                      reset='v = V_reset; x_spike += 1.; y_act=1.',
                      refractory='tau_rp_E_var', method='euler', name='P_E')
    P_E.v = V_L
    P_E.k_AMPA_E_hsp = 0.35
    P_E.k_GABA_E_hsp = 0.35
    P_E.tau_ins_AMPA_E = tau_ins_AMPA
    P_E.tau_int_AMPA_E = tau_int_AMPA
    P_E.tau_ins_GABA_E = tau_ins_GABA
    P_E.tau_int_GABA_E = tau_int_GABA
    P_E.V_thr_E = V_thr
    P_E.tau_rp_E_var = tau_rp_E

    P_I = NeuronGroup(N_I, eqs_I, threshold='v > V_thr',
                      reset='v = V_reset; x_spike += 1.; y_act=1.',
                      refractory=tau_rp_I, method='euler', name='P_I')
    P_I.v = V_L

    pg_inp = PoissonGroup(N_inp, rates=input_freq * Hz, name='inp')

    C_PG_E = Synapses(pg_inp, P_E, model='w:1', on_pre='''
        s_AMPA_ext += w
        ''', method='euler', name='PG_E')
    C_PG_E.connect(j='i')

    C_E_E = Synapses(P_E, P_E, model='w:1', on_pre='''
        s_AMPA += w
        ''', method='euler', name='E_E')
    for y1 in range(n_output_rows):
        for x1 in range(n_output_columns):
            post = y1 * n_output_columns + x1
            for dy in range(-recurrent_radius, recurrent_radius + 1):
                for dx in range(-recurrent_radius, recurrent_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    x2 = x1 + dx
                    y2 = y1 + dy
                    if 0 <= x2 < n_output_columns and 0 <= y2 < n_output_rows:
                        pre = y2 * n_output_columns + x2
                        C_E_E.connect(i=pre, j=post)
    C_E_I = Synapses(P_E, P_I, model='w:1', on_pre='''
    s_AMPA += w
    ''', method='euler', name='E_I')
    C_I_E = Synapses(P_I, P_E, model='w:1', on_pre='s_GABA += w', method='euler', name='I_E')
    for inh_idx, (ix, iy) in enumerate(inh_positions):
        for ex_y in range(n_output_rows):
            for ex_x in range(n_output_columns):
                if abs(ex_x - ix) <= recurrent_radius and abs(ex_y - iy) <= recurrent_radius:
                    exc_idx = ex_y * n_output_columns + ex_x
                    C_I_E.connect(i=inh_idx, j=exc_idx)
                    C_E_I.connect(i=exc_idx, j=inh_idx)
    C_I_I = Synapses(P_I, P_I, model='w:1', on_pre='s_GABA += w', method='euler', name='I_I')
    for iy in range(n_inh_rows):
        for ix in range(n_inh_cols):
            post = iy * n_inh_cols + ix
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    x2, y2 = ix + dx, iy + dy
                    if 0 <= x2 < n_inh_cols and 0 <= y2 < n_inh_rows:
                        pre = y2 * n_inh_cols + x2
                        C_I_I.connect(i=pre, j=post)

    C_PG_E.w[:] = 1.1
    C_I_E.w[:] = np.clip(np.random.normal(loc=0.52, scale=0.05, size=len(C_I_E.w)), 0.0, 1.0)
    C_I_I.w[:] = np.clip(np.random.normal(loc=0.40, scale=0.05, size=len(C_I_I.w)), 0.0, 1.0)
    C_E_I.w[:] = 0.9
    C_E_E.w[:] = np.clip(np.random.normal(loc=0.20, scale=0.05, size=len(C_E_E.w)), 0.0, 1.0)

    # DBS: second Poisson source, CURRENT-based, one-to-one onto E cells
    pg_dbs = PoissonGroup(N_E, rates=0 * Hz, name='dbs')
    C_DBS_E = Synapses(pg_dbs, P_E, model='w_dbs : amp',
                       on_pre='I_dbs_post += w_dbs',
                       method='euler', name='DBS_E')
    C_DBS_E.connect(j='i')
    C_DBS_E.w_dbs = 0 * pA

    vm_container = NeuronGroup(1, 'vm_pop : volt', method='euler')
    C_vm = Synapses(P_E, vm_container,
                    model='vm_pop_post = vm_lp_pre / N_pre : volt (summed)',
                    method='euler')
    C_vm.connect()

    act_container = NeuronGroup(1, 'frac_act : 1', method='euler')
    C_act = Synapses(
        P_E, act_container,
        model='''scale:1 (constant)
            frac_act_post = scale * clip(y_act_pre, 0, 1) : 1 (summed)''',
        method='euler'
    )
    C_act.connect()
    C_act.scale = 1.0 / float(N_E)

    act_cont_I = NeuronGroup(1, 'frac_act : 1', method='euler')
    C_act_I = Synapses(
        P_I, act_cont_I,
        model='''scale:1 (constant)
            frac_act_post = scale * clip(y_act_pre, 0, 1) : 1 (summed)''',
        method='euler'
    )
    C_act_I.connect()
    C_act_I.scale = 1.0 / float(N_I)

    rate_container_E = NeuronGroup(1, 'r_pop_E : Hz', method='euler')
    C_rE = Synapses(P_E, rate_container_E,
                    model='r_pop_E_post = r_est_pre / N_pre : Hz (summed)',
                    method='euler')
    C_rE.connect()

    rate_container_I = NeuronGroup(1, 'r_pop_I : Hz', method='euler')
    C_rI = Synapses(P_I, rate_container_I,
                    model='r_pop_I_post = r_est_pre / N_pre : Hz (summed)',
                    method='euler')
    C_rI.connect()

    V_mon = StateMonitor(vm_container, 'vm_pop', record=True, dt=record_dt, name='V_mon')
    act_mon = StateMonitor(act_container, 'frac_act', record=True, dt=record_dt, name='act_mon')
    act_mon_I = StateMonitor(act_cont_I, 'frac_act', record=True, dt=record_dt, name='act_mon_I')

    rate_pop_mon_E = StateMonitor(rate_container_E, 'r_pop_E', record=True, dt=record_dt, name='rate_pop_mon_E')
    rate_pop_mon_I = StateMonitor(rate_container_I, 'r_pop_I', record=True, dt=record_dt, name='rate_pop_mon_I')

    gAMPA_mon_E = StateMonitor(P_E, 'k_AMPA_E_hsp', record=True, dt=record_dt, name='k_AMPA_E_hsp')
    gGABA_mon_E = StateMonitor(P_E, 'k_GABA_E_hsp', record=True, dt=record_dt, name='k_GABA_E_hsp')

    sp_E = SpikeMonitor(P_E, name='sp_exc')
    sp_I = SpikeMonitor(P_I, name='sp_inh')

    r_est_mon = StateMonitor(P_E, variables=['r_est'], record=[45], name='r_est', dt=record_dt)

    net = Network(collect())
    net.run(0 * ms, report=None)
    return net


def run_sim_seeded(tau_ins_AMPA, tau_int_AMPA, tau_ins_GABA, tau_int_GABA,
                   input_freq=40, show_time=1000 * ms, network_seed=None,
                   report=None, report_period=30 * second):
    """Seeded equivalent of run_sim() in DBS_intervention.ipynb."""
    net = build_network_seeded(tau_ins_AMPA=tau_ins_AMPA, tau_int_AMPA=tau_int_AMPA,
                               tau_ins_GABA=tau_ins_GABA, tau_int_GABA=tau_int_GABA,
                               input_freq=input_freq, network_seed=network_seed)
    net.run(show_time, report=report, report_period=report_period)
    return net


def calibrate_baseline(network_seed, out_path, calib_time_s=15000.0, dt_us=1000,
                       input_freq=40, tau_s=50.0, report="auto",
                       report_period=30 * second, trace_out_path=None):
    """Build a fresh network for `network_seed`, run it to the ~2 Hz HSP fixed
    point (same recipe as the notebooks: all four HSP time constants = 50 s,
    seed set before the build, `calib_time_s` seconds of settling), and store
    it to `out_path` under the store name 'baseline' -- a per-seed replacement
    for the notebooks' shared baseline_2hz.bri.

    `report="auto"` (the default) prints a real percentage/ETA every
    `report_period` of wall time, tagged with the seed -- calibration is a
    single net.run() of known duration (calib_time_s), so this is an honest
    computed estimate, not a guess. Pass `report=None` to silence it, or your
    own brian2 `report=` callback/string.

    If `trace_out_path` is given, also dumps the E/I population rate and
    k_AMPA/k_GABA over the *entire* settling run to that .npz path -- a
    sanity check that each seed's pre-drug baseline actually reaches (and
    stays at) the ~2 Hz fixed point rather than something unstable.

    Returns the resolved out_path (as a string, the form net.store()/restore()
    want for `filename`).
    """
    if report == "auto":
        from progress import brian_progress_reporter
        report = brian_progress_reporter(f"calib seed{network_seed}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_scope()
    defaultclock.dt = dt_us * us
    net = run_sim_seeded(
        tau_ins_AMPA=tau_s * second, tau_int_AMPA=tau_s * second,
        tau_ins_GABA=tau_s * second, tau_int_GABA=tau_s * second,
        input_freq=input_freq, show_time=calib_time_s * second,
        network_seed=network_seed, report=report, report_period=report_period,
    )
    if trace_out_path is not None:
        _save_calibration_trace(trace_out_path, net, network_seed)
    net.store('baseline', filename=str(out_path))
    return str(out_path)


def extract_calibration_trace(network_seed, baseline_path, trace_out_path,
                              tau_s=50.0, input_freq=40, dt_us=1000):
    """Pull the settling trace back out of an ALREADY-CALIBRATED baseline --
    no simulation needed. net.store()/restore() preserves each monitor's
    full recorded history, not just the final state, so everything
    calibrate_baseline() would have recorded is already sitting in
    `baseline_path` (that's why the .bri files are ~170MB, not a few KB).

    Only builds the network structure (values get overwritten by restore())
    then reads the monitors -- cheap and fast, just needs the usual one-time
    Cython compile if this process hasn't built this network before.
    """
    start_scope()
    defaultclock.dt = dt_us * us
    net = build_network_seeded(
        tau_ins_AMPA=tau_s * second, tau_int_AMPA=tau_s * second,
        tau_ins_GABA=tau_s * second, tau_int_GABA=tau_s * second,
        input_freq=input_freq, network_seed=network_seed,
    )
    net.restore('baseline', filename=str(baseline_path))
    _save_calibration_trace(trace_out_path, net, network_seed)
    return str(trace_out_path)


def _save_calibration_trace(trace_out_path, net, network_seed):
    import numpy as np
    from HSP_Phase import _trace  # same helper characterize_cycle() uses

    trace_out_path = Path(trace_out_path)
    trace_out_path.parent.mkdir(parents=True, exist_ok=True)
    kA_mon = net['k_AMPA_E_hsp']
    kG_mon = net['k_GABA_E_hsp']
    rE_mon = net['rate_pop_mon_E']
    rI_mon = net['rate_pop_mon_I']
    re_mon = net['r_est']
    np.savez_compressed(
        trace_out_path,
        seed=network_seed,
        t=np.asarray(kA_mon.t / second),
        rE=_trace(rE_mon, 'r_pop_E'), rI=_trace(rI_mon, 'r_pop_I'),
        kA=_trace(kA_mon, 'k_AMPA_E_hsp'), kG=_trace(kG_mon, 'k_GABA_E_hsp'),
        r_est=np.asarray(re_mon.r_est[0] / Hz),
    )
