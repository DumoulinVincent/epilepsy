"""Conditions and grid presets for the multi-seed replication check.

`CONDITIONS` is built by make_condition() (single place that defines what a
"condition" is, per your request): untreated, benzo (k=1.0), a DBS amplitude
sweep at the 120 Hz DBS_FREQ_HZ used throughout DBS_intervention.ipynb, and
two weight-based interventions (see interventions.py) -- AMPA block and
SV2A. Drug dicts use the same keys _apply_drugs() in HSP_Phase.py expects;
the DBS amplitude list mirrors the notebook's own `amps` sweep (its "amp=0"
point is this list's "untreated"), so per-seed results are directly
comparable to the single-seed (seed=123) numbers you already have.

AMPA block and SV2A are WEIGHT interventions, not drug dicts -- worker.py
handles them differently (see interventions.py's docstring for why: drug={}
keys are silently ignored for anything HSP_Phase._apply_drugs() doesn't
recognize, so weight changes have to happen before that point, via a
locally-modified baseline_2hz.bri).

`FINE_GRID` follows the notebook's probe length and seeds (STEP_S=0.05,
n_steps=2000, seeds=[11,22,...,111], burst/steady = 36/40 Hz) but with the
same amplitude, HSP speed and plateau rule as COARSE_GRID (see below).

`COARSE_GRID` is the cheaper grid for the cross-seed ordering check: fewer
probe seeds, same 100 s probe signal, shorter max_delay for the IPC memory-
capacity fit, and -- the dominant cost -- a shorter warmup/scan window for
cycle characterization and phase-snapshot finding. Still measured in
thousands of simulated seconds (tau_HSP=100s; the HSP cycle just needs that
long to settle), just ~2-3x smaller windows than FINE_GRID.
`char_kw.record_s` sets how much of the cycle trace characterize_cycle()
captures (and therefore how long plot_traces.py's per-condition trace plots
run for) -- 2000 s, enough for a full cycle now that HSP runs twice as fast.
"""

DBS_FREQ_HZ = 120.0
DBS_AMPS_PA = [5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

# AMPA_BLOCK_CUTS / SV2A_CUTS are REDUCTION fractions (0.1 = cut 10%), not
# multipliers -- a *0.5 multiplier (cutting the mean weight in half) silenced
# the network for both interventions, so these sweep much milder cuts
# instead. scale = 1 - cut is what actually gets passed to
# interventions.apply_weight_intervention(). 0.2 (-20%) also came back too
# strong for AMPA block -- left in the sweep as a reference/comparison point
# since it's already-computed data, not because it's the recommended dose;
# drop it from the list below if you'd rather it not show up in new plots.
AMPA_BLOCK_CUTS = [0.1, 0.15, 0.2]
SV2A_CUTS = [0.1, 0.15]


def make_condition(name, drug=None, weights=None):
    """Single place that defines what a "condition" is: a name, an
    HSP_Phase._apply_drugs()-style state-variable drug spec (na_blocker/
    benzo/dbs keys -- see HSP_Phase.py), and/or a weight intervention (see
    interventions.py). `weights`, if given, is
    {'kind': 'ampa_block'|'sv2a', 'scale': float}. A condition can carry
    both `drug` and `weights` at once (worker.py applies weights first, then
    drug, before any simulation runs) though nothing here currently does."""
    return {"name": name, "drug": drug or {}, "weights": weights}


CONDITIONS = [
    make_condition("untreated"),
    make_condition("benzo (k=1.0)", drug={"benzo": True, "benzo_kGABA": 1.0}),
] + [
    make_condition(
        f"DBS {amp:g}pA",
        drug={"dbs": True, "dbs_freq_hz": DBS_FREQ_HZ, "dbs_amp_pA": amp})
    for amp in DBS_AMPS_PA
] + [
    make_condition(
        f"AMPA block (-{cut:.0%})",
        weights={"kind": "ampa_block", "scale": 1.0 - cut})
    for cut in AMPA_BLOCK_CUTS
] + [
    make_condition(
        f"SV2A (-{cut:.0%})",
        weights={"kind": "sv2a", "scale": 1.0 - cut})
    for cut in SV2A_CUTS
]

BURST_FREQ_HZ = 36.0    # seizure input freq; every CONDITIONS entry gets its own run here
STEADY_FREQ_HZ = 40.0   # healthy fixed point; computed ONCE per seed at drug={} and
                        # shared across every condition -- see run_seed_check.py Step A.5

# HSP runs twice as fast as in the calibration: hsp_tau_s replaces the 50 s
# tau_ins/tau_int (AMPA and GABA) after every baseline restore. The baselines'
# fixed point doesn't depend on these time constants, so they stay valid.
#
# Snapshot rule, same for every condition (coarse_phase._plateau_snapshot):
# run with HSP on until the E-population mean k_AMPA has varied by less than
# plateau_tol (max - min) over the last plateau_window_s, then snapshot.
# plateau_tol = 0.06 was fixed BEFORE these runs, from the old-speed traces
# over 1000 s windows: steady state max 0.006 (40 Hz runs, calibration end);
# last 1000 s before a seizure, on the plateau, median 0.025 / max 0.054
# (14 windows; 5 that reached back onto the recovery ramp excluded).
#
# IPC probe input rate: drive + amp_hz * u, u ~ U[-1, 1]. probe_r0_hz=None
# means each network is probed around its OWN drive (36 Hz for every
# condition, 40 Hz for the reference): same drive and same signal for every
# network, as in the clinic. (Probing every network at a fixed 30 +- 8 Hz
# replaced the drive, under-drove each network by a different amount, and
# made IPC track that.) The plateau networks are close to seizure: 36 +- 8
# and 36 +- 4 Hz on the OLD snapshots (slower HSP, old rule) tipped some
# probes into a self-sustained saturated state -- amp_hz is provisional
# until the runaway test on the new snapshots (inspect_ipc_probe.py --amp).
#
# inspect_extend_s: every job also saves the IPC probe inspection figures
# (probe_plots.py) to results/ipc_probe_inspection/runs/. For the context
# figure the HSP-on run continues this long past the snapshot (the probes
# still start from the snapshot). None = no inspection figures, 0 = context
# only up to the snapshot (no extra simulation).
_PLATEAU = dict(hsp_tau_s=25.0, plateau_window_s=750.0, plateau_tol=0.06,
                plateau_chunk_s=25.0, probe_r0_hz=None, inspect_extend_s=0.0)

FINE_GRID = dict(
    probe_seeds=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111],
    amp_hz=4.0,             # provisional, see the probe input comment above
    step_s=0.05,
    n_steps=2000,
    max_delay=40,
    char_kw=dict(warmup_s=2000.0, record_s=3000.0),
    plateau_max_s=7000.0,
    **_PLATEAU,
)

COARSE_GRID = dict(
    probe_seeds=[11, 22, 33, 44],
    amp_hz=4.0,             # provisional, see the probe input comment above
    step_s=0.05,
    n_steps=2000,           # 100 s probe (was 800 = 40 s: too few spikes per step at ~0.5 Hz)
    max_delay=15,
    char_kw=dict(warmup_s=600.0, record_s=2000.0),
    plateau_max_s=5000.0,   # give up (no IPC) if k_AMPA never settles by then
    **_PLATEAU,
)
