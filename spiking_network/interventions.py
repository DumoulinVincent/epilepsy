"""Weight-based interventions -- as opposed to HSP_Phase._apply_drugs()'s
state-variable ones (benzo scales k_GABA, na_blocker moves V_thr/refractory,
dbs sets a current injection rate/amplitude).

Why these can't just be new keys in a drug={} dict, HSP_Phase-style:
characterize_cycle()/snapshot_at_kphase() both start with
`net.restore('baseline', filename='baseline_2hz.bri')`, which overwrites
EVERY state variable including synapse weights, and then call
HSP_Phase._apply_drugs(net, drug) -- which only recognizes na_blocker/benzo/
dbs and silently ignores anything else. So a weight edit made before calling
into those functions is wiped out by their own restore, and there's no hook
to touch weights after it without editing HSP_Phase.py (which we don't).

Instead: worker.py restores the seed's ORIGINAL calibrated baseline, applies
a weight intervention here, and re-stores the result as THAT JOB's own local
baseline_2hz.bri. HSP_Phase's functions then restore from ours, completely
unmodified, and never need to know weight interventions exist.

Two interventions, both scale a set of Synapses objects' `w` in place by a
constant factor, preserving the heterogeneity each projection was drawn
with (only the mean moves):

  AMPA_BLOCK -- scales the two AMPA-receptor-mediated projections: 'E_E'
    (E->E recurrent) and 'E_I' (E->I). A *0.5 multiplier (halving the mean
    weight) silenced the network, and even -20% came back too strong, so
    conditions.py sweeps milder cuts (10%/15%/20%, i.e. scale=0.9/0.85/0.8).

  SV2A -- scales every recurrent weight except the external Poisson drive
    ('PG_E'): 'E_E', 'E_I', 'I_E', 'I_I'. A *0.5 multiplier also silenced the
    network here, so conditions.py sweeps 10%/15% cuts (scale=0.9/0.85)
    instead, same as AMPA block.
"""
import numpy as np

AMPA_BLOCK_SYNAPSES = ["E_E", "E_I"]
SV2A_SYNAPSES = ["E_E", "E_I", "I_E", "I_I"]  # everything except 'PG_E'

KINDS = {
    "ampa_block": AMPA_BLOCK_SYNAPSES,
    "sv2a": SV2A_SYNAPSES,
}


def scale_weights(net, synapse_names, scale):
    """Multiply Synapses[name].w by `scale` in place, for each name in
    `synapse_names`. Scaling (rather than setting a fixed value) preserves
    the per-synapse heterogeneity each projection was originally drawn with
    (e.g. C_E_E.w ~ N(0.20, 0.05) clipped) -- only the mean moves."""
    for name in synapse_names:
        syn = net[name]
        syn.w[:] = np.asarray(syn.w[:]) * scale


def apply_weight_intervention(net, kind, scale=1.0):
    """Apply weight intervention `kind` (None, 'ampa_block', or 'sv2a') to
    `net`, in place. Call AFTER net.restore('baseline', filename=<the
    seed's calibrated baseline>) and BEFORE re-storing as a job's own local
    baseline_2hz.bri -- see worker.run_condition_job.

    `scale` multiplies the relevant projections' weights (0.5 = halve the
    mean). Add new interventions by adding an entry to KINDS."""
    if kind is None:
        return
    if kind not in KINDS:
        raise ValueError(f"unknown weight intervention kind: {kind!r} "
                        f"(known: {sorted(KINDS)})")
    scale_weights(net, KINDS[kind], scale)
