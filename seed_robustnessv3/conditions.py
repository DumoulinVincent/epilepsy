"""Conditions and run settings for seed_robustnessv2.

`CONDITIONS` (make_condition): untreated, benzo (k_GABA = 1), DBS amplitudes
(120 Hz Poisson current pulses onto E cells), AMPA block and SV2A weight
cuts. Each workflow then picks its subset and drive frequency:
  CYCLE_*  -- cycle-wide scalar encoding (cycle_ipc / cycle_analysis / run_cycle)
  LFP_*    -- long LFP runs for seizure prediction (lfp_cycle / run_lfp_cycle)
  IMAGE_*  -- image recovery (image_stimulus / image_cycle / image_analysis /
              run_image_cycle)
HSP time constants: hsp_tau_s replaces the calibration's 50 s after every
baseline restore (HSP twice as fast); the baselines stay valid.
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
STEADY_FREQ_HZ = 40.0   # healthy steady state: the untreated 40 Hz reference

# ---------------------------------------------------------------------------
# Cycle-wide encoding (cycle_ipc.py / cycle_analysis.py / run_cycle.py)
# ---------------------------------------------------------------------------
# No freezing, no snapshot: each (seed, condition) runs run_s with HSP ON and
# the signal ALWAYS in the drive -- rate = freq + amp_hz * u[k], u iid
# U[-1, 1] per step_s step, the same u (u_seed) for every run -- so the
# network goes through its own seizure cycles while being "tested" the whole
# time. Encoding (how well a linear readout of the E spike counts recovers
# u[k]) is then measured per window_s window and lined up against time to
# the next / since the last seizure. sim_seed fixes Brian's Poisson noise per
# network seed (the same stream for every condition of that seed).
CYCLE_GRID = dict(
    hsp_tau_s=25.0,
    run_s=8000.0,
    step_s=0.05,
    amp_hz=4.0,
    u_seed=11,
    null_seed=999,         # an iid signal never presented -> the encoding noise floor
    window_s=50.0,
    settle_s=500.0,        # leave the 40 Hz-adapted baseline state before analysing
    seizure_hz=10.0,       # E population rate (1 s bins) above this = seizure
    seizure_pad_s=(2.0, 5.0),   # steps this close before / after a seizure count as ictal
    train_gap_s=60.0,      # interictal-trained readouts use steps this far from any seizure
    frozen_train_s=2000.0, # frozen readout: trained once on [settle_s, settle_s + this), then fixed
)

CYCLE_CONDITIONS = [
    dict(c, freq_hz=BURST_FREQ_HZ) for c in CONDITIONS
    if c["name"] in ["untreated", "benzo (k=1.0)"]
    + [f"DBS {a:g}pA" for a in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)]
] + [dict(make_condition("reference 40Hz"), freq_hz=STEADY_FREQ_HZ)]


# ---------------------------------------------------------------------------
# Long LFP runs for seizure prediction (lfp_cycle.py / run_lfp_cycle.py /
# lfp_cycle_analysis.py): normal constant drive, no signal, HSP on, LFP proxy
# recorded every rec_dt_ms (1 ms = every integration step; 2 ms aliased).
# ---------------------------------------------------------------------------
LFP_GRID = dict(hsp_tau_s=25.0, run_s=8000.0, rec_dt_ms=1.0)
LFP_CONDITIONS = [dict(c, freq_hz=BURST_FREQ_HZ) for c in CONDITIONS
                  if c["name"] in ("untreated", "DBS 10pA", "DBS 20pA")]


# ---------------------------------------------------------------------------
# Image recovery (image_stimulus.py / image_cycle.py / image_analysis.py /
# run_image_cycle.py). The drive itself carries a weak binary image: during an
# image, cell i gets f + contrast_hz * (p_i - q) (q = k_white / 100), so white
# cells get f + 0.9 a, black cells f - 0.1 a, and the summed rate stays 100 f;
# during blanks every cell gets f. HSP on, DBS/drugs as in the other
# workflows. All times are integer multiples of sched_ms (10 ms) so schedule,
# spike bins and readout windows share exact tick boundaries.
# ---------------------------------------------------------------------------
IMAGE_GRID = dict(
    hsp_tau_s=25.0,
    run_s=8000.0,
    dt_ms=1,                    # integration step (asserted against the network)
    sched_ms=10,                # schedule and capture bin
    contrast_hz=4.0,            # a: white - black input-rate difference
    k_white=10,                 # white pixels per image
    image_ms=600,
    blank_ms=(200, 600),        # uniform over multiples of sched_ms, inclusive
    response_start_ms=20,
    readout_ms=(50, 100, 250, 500),
    primary_readout_ms=250,
    settle_s=500.0,
    library="random",           # "random" (balanced random masks) or "lines"
    n_library_blocks=10,        # random library: 10 blocks x 10 disjoint masks = 100 masks
    stim_seed=101,              # mask library
    order_seed=202,             # presentation order and blank durations
    seizure_hz=10.0,            # E rate in 1 s bins above this = seizure
    seizure_pad_s=(2.0, 5.0),   # response windows this close to a seizure = seizure-overlapping
    train_gap_s=60.0,           # adapted decoders train on trials this far from any seizure
    # decoding
    outer_folds=5,
    inner_folds=4,
    gap_trials=1,               # trials dropped between training and evaluation blocks
    C_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),   # inverse L2 strength (logistic)
    frozen_train_frac=0.7,      # frozen decoder: first 70% of the reference trials train it
    heldout_mask_folds=5,       # held-out-mask-ID evaluation: folds over mask IDs
    n_perm=1,                   # label-permutation null repeats (full pipeline each)
    summary_window_s=50.0,      # reporting unit for cycle plots (not a training unit)
)

IMAGE_CONDITIONS = CYCLE_CONDITIONS            # every condition is selectable
IMAGE_DEFAULT = ["reference 40Hz", "untreated", "benzo (k=1.0)", "DBS 20pA", "DBS 60pA"]


# ---------------------------------------------------------------------------
# HSP-active functional assays (assay_stimulus / assay_sim / assay_analysis /
# run_assays). An unprobed native trajectory per (seed, condition) with
# checkpoints on a uniform grid; from each checkpoint, separate branches (sham,
# encoding, noise, memory) run with HSP, its rate estimator and the treatment
# all active, then the native run continues from the checkpoint unchanged.
# Probe input: lambda_i = b_i + A * m_i, with m the zero-sum pattern of the
# current symbol (+/- the left/right mask w, or a corrupted mix); b_i = the
# condition's native drive. A is the same for every condition.
# ---------------------------------------------------------------------------
ASSAY_GRID = dict(
    hsp_tau_s=25.0,
    run_s=8000.0,
    settle_s=500.0,             # no checkpoints (and no burden) before this
    cp_spacing_s=20.0,          # uniform checkpoint grid after settle_s
    amplitude_frac=0.2,         # A = this x the smallest b among the conditions compared
    sched_ms=10,                # symbol timeline resolution
    mask="lr",                  # w: +1 left half (x < 5), -1 right half
    # encoding / noise branches (presentations)
    enc_present_ms=100, enc_count_ms=50, enc_trials=10,       # first trial = unscored lead-in
    noise_etas=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), noise_reps=2,   # per eta per branch (+1 lead-in)
    # memory branch (iid binary symbols)
    mem_bin_ms=50, mem_obs_ms=20, mem_obs_ms_alt=10, mem_bins=40, mem_max_lag=10,
    sham_s=2.0,
    # seeds
    stim_seed=4242,             # symbol sequences: identical across conditions within a network seed
    # phases and burden (the existing detector)
    seizure_hz=10.0, seizure_pad_s=(2.0, 5.0), recovery_s=400.0,
    # analysis
    outer_folds=5, inner_folds=4, gap_cps=1,
    C_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
    ridge_grid=(1.0, 10.0, 100.0, 1000.0, 10000.0),
    n_boot=1000, thin_repeats=5,
)
ASSAY_CONDITIONS = CYCLE_CONDITIONS
ASSAY_DEFAULT = ["reference 40Hz", "untreated", "benzo (k=1.0)", "DBS 20pA", "DBS 60pA", "DBS 100pA"]
