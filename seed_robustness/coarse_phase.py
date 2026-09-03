"""Coarse-grid phase-resolved replication.

HSP_Phase.phase_capacity_seeded()/run_phase_replication() are convenience
wrappers around characterize_cycle() + snapshot_at_kphase() + _ipc_probe().
Good defaults for one careful run -- but snapshot_at_kphase()'s own defaults
(warmup_s=2000.0, max_scan_s=8000.0) are NOT forwarded through the wrapper,
so phase_capacity_seeded()'s exposed knobs (n_steps/max_delay/char_kw) can't
touch the dominant cost: finding each phase's snapshot is a fixed multi-
thousand-second scan regardless of how small the probe grid is.

For a multi-seed *ordering check* (as opposed to the fine single-seed
analysis) that fixed cost matters, so this module calls the same lower-level
HSP_Phase functions directly with warmup_s/chunk_s/max_scan_s also exposed.
Nothing in HSP_Phase.py is modified -- this only calls its existing public
functions (some underscore-prefixed by convention, but not actually private)
with different keyword arguments than its own wrapper passes them.
"""
import numpy as np

from HSP_Phase import _PHASE_FRAC, _ipc_probe, characterize_cycle, snapshot_at_kphase
from hsp_capacity import make_input_signal
from progress import Stage


def phase_capacity_seeded_coarse(net, input_freq_hz, seeds, drug=None,
                                 phases=("post_ictal", "mid", "pre_ictal"),
                                 n_steps=2000, amp_hz=8.0, step_s=0.05, max_delay=40,
                                 probe_r0=None, char_kw=None,
                                 snapshot_warmup_s=2000.0, snapshot_chunk_s=50.0,
                                 snapshot_max_scan_s=8000.0, progress_tag=None):
    """Same contract as HSP_Phase.phase_capacity_seeded(), plus control over
    snapshot_at_kphase()'s warmup/scan window (its defaults: 2000s/50s/8000s,
    which is what you get if you leave these three at their defaults too).

    `progress_tag`, if given, prints a start/done wall-clock marker (see
    progress.Stage) around each major step -- characterizing the cycle at
    `input_freq_hz`, finding each phase's snapshot, and probing it -- so you
    can see which step a given (seed, condition) job is on. No percentage:
    the snapshot scan runs until it finds a match or exhausts its budget, so
    its total duration isn't knowable in advance.
    """
    drug = drug or {}
    char_kw = char_kw or {}
    r0 = input_freq_hz if probe_r0 is None else probe_r0
    tag = f"{progress_tag}/{input_freq_hz:g}Hz" if progress_tag else None

    with _stage(tag, "characterizing cycle"):
        cyc = characterize_cycle(net, input_freq_hz, drug=drug, **char_kw)

    ph_list = ([("steady", 0.5)] if not cyc["is_bursting"]
              else [(p, _PHASE_FRAC[p]) for p in phases])
    by_phase = {}
    for name, frac in ph_list:
        ptag = f"{tag}/{name}" if tag else None
        k_target = cyc["k_lo"] + frac * (cyc["k_hi"] - cyc["k_lo"])
        with _stage(ptag, f"scanning for snapshot (budget {snapshot_max_scan_s:.0f}s sim)"):
            snap = snapshot_at_kphase(net, input_freq_hz, k_target, drug=drug,
                                      warmup_s=snapshot_warmup_s, chunk_s=snapshot_chunk_s,
                                      max_scan_s=snapshot_max_scan_s,
                                      store_name=f"phase_{name}")
        if not snap["found"]:
            by_phase[name] = dict(found=False, status=snap.get("status"))
            continue
        acc = {k: [] for k in ("ipc_total", "linear_mc", "nonlinear", "E_rate_probe",
                               "state_rank", "participation_ratio", "burst_during_probe")}
        with _stage(ptag, f"probing {len(seeds)} seed(s)"):
            for s in seeds:
                u = make_input_signal(n_steps, seed=int(s))
                r = _ipc_probe(net, snap["store_name"], u, r0_hz=r0, amp_hz=amp_hz,
                               step_s=step_s, max_delay=max_delay, rng_seed=int(s))
                for k in acc:
                    acc[k].append(r[k])
        acc = {k: np.array(v) for k, v in acc.items()}
        acc.update(found=True, seeds=list(seeds),
                  k_AMPA=snap["k_AMPA"], k_GABA=snap["k_GABA"])
        by_phase[name] = acc
    return dict(cycle=cyc, by_phase=by_phase, input_freq_hz=input_freq_hz, drug=drug)


def _stage(tag, label):
    if tag is None:
        return _NullStage()
    return Stage(tag, label)


class _NullStage:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
