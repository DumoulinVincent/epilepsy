"""Conditions and grid presets for the multi-seed replication check.

`CONDITIONS` is untreated, benzo (k=1.0), and a DBS amplitude sweep at the
120 Hz DBS_FREQ_HZ used throughout DBS_intervention.ipynb -- drug dicts use
the same keys _apply_drugs() in HSP_Phase.py expects, and the amplitude list
mirrors the notebook's own `amps` sweep (its "amp=0" point is this list's
"untreated"), so per-seed results are directly comparable to the single-seed
(seed=123) numbers you already have.

`FINE_GRID` mirrors the probe settings used in that notebook (AMP_STAR=8.0,
STEP_S=0.05, n_steps=2000, seeds=[11,22,...,111], burst/steady = 36/40 Hz),
including HSP_Phase.snapshot_at_kphase()'s own defaults for the phase-finding
scan (snapshot_warmup_s=2000/snapshot_chunk_s=50/snapshot_max_scan_s=8000 --
see coarse_phase.py for why those need to be spelled out explicitly).

`COARSE_GRID` is the cheaper grid for the cross-seed ordering check: fewer
probe seeds, shorter probe signal, shorter max_delay for the IPC memory-
capacity fit, and -- the dominant cost -- a shorter warmup/scan window for
cycle characterization and phase-snapshot finding. Still measured in
thousands of simulated seconds (tau_HSP=100s; the HSP cycle just needs that
long to settle), just ~2-3x smaller windows than FINE_GRID.
`char_kw.record_s=2000` sets how much of the cycle trace characterize_cycle()
captures (and therefore how long plot_traces.py's per-condition trace plots
run for) -- 2000s to comfortably show a full DBS cycle, which can run slower
than benzo's near-immediate suppression.
"""

DBS_FREQ_HZ = 120.0
DBS_AMPS_PA = [5.0, 10.0, 20.0, 25.0, 30.0, 40.0]

CONDITIONS = [
    {"name": "untreated",     "drug": {}},
    {"name": "benzo (k=1.0)", "drug": {"benzo": True, "benzo_kGABA": 1.0}},
] + [
    {"name": f"DBS {amp:g}pA",
     "drug": {"dbs": True, "dbs_freq_hz": DBS_FREQ_HZ, "dbs_amp_pA": amp}}
    for amp in DBS_AMPS_PA
]

BURST_FREQ_HZ = 36.0    # seizure input freq; every CONDITIONS entry gets its own run here
STEADY_FREQ_HZ = 40.0   # healthy fixed point; computed ONCE per seed at drug={} and
                        # shared across every condition -- see run_seed_check.py Step A.5

FINE_GRID = dict(
    probe_seeds=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111],
    amp_hz=8.0,
    step_s=0.05,
    n_steps=2000,
    max_delay=40,
    phases=("post_ictal", "mid", "pre_ictal"),
    char_kw=dict(warmup_s=2000.0, record_s=4000.0),
    snapshot_warmup_s=2000.0,
    snapshot_chunk_s=50.0,
    snapshot_max_scan_s=8000.0,
)

COARSE_GRID = dict(
    probe_seeds=[11, 22, 33, 44],
    amp_hz=8.0,
    step_s=0.05,
    n_steps=800,
    max_delay=15,
    phases=("post_ictal", "mid", "pre_ictal"),
    char_kw=dict(warmup_s=600.0, record_s=2000.0),
    snapshot_warmup_s=600.0,
    snapshot_chunk_s=25.0,
    snapshot_max_scan_s=2000.0,
)
