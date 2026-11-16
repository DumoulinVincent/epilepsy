"""Probe inputs for the HSP-active functional assays (conditions.ASSAY_GRID).

Input channels are the network's existing external Poisson channels, one per
E cell (one-to-one). During a probe, channel i has rate

    lambda_i = b_i + A * m_i,   m = q * w           (clean symbol q = +/-1)
                                m = (1-eta) q w + eta v   (corrupted)

w: fixed zero-sum spatial mask (+1 left half x < 5, -1 right half), v: a fresh
balanced +/-1 mask per corrupted trial, independent of the label. Every
pattern row sums to zero, so the expected summed drive is 100 b for every
symbol and corruption level; |m_i| <= 1 and A <= 0.2 b keep lambda >= 0.8 b.
Rates are never clipped.

All branches of a run are laid out back to back on one symbol timeline
(sched_ms resolution); a branch reads its own slot through an offset, so the
whole branch is a single net.run.
"""
import numpy as np

N = 100


def mask_w(kind="lr"):
    x = np.arange(N) % 10
    if kind != "lr":
        raise ValueError(kind)
    return np.where(x < 5, 1.0, -1.0)


def balanced_random_mask(rng):
    v = np.r_[np.ones(N // 2), -np.ones(N // 2)]
    return rng.permutation(v)


def checkpoint_times(g):
    longest = max(branch_durations(g).values())
    t = np.arange(g["settle_s"], g["run_s"] - longest + 1e-9, g["cp_spacing_s"])
    return t


def branch_durations(g):
    return dict(sham=g["sham_s"],
                enc=g["enc_trials"] * g["enc_present_ms"] / 1000,
                noise=(1 + len(g["noise_etas"]) * g["noise_reps"]) * g["enc_present_ms"] / 1000,
                mem=g["mem_bins"] * g["mem_bin_ms"] / 1000)


def build_schedule(g, n_cp):
    """Symbols, pattern rows and timeline slots for n_cp checkpoints.
    Deterministic in g['stim_seed'] (identical across conditions)."""
    sb = g["sched_ms"]
    for v in (g["enc_present_ms"], g["enc_count_ms"], g["mem_bin_ms"], g["mem_obs_ms"], g["mem_obs_ms_alt"]):
        if v % sb or (v % g.get("dt_ms", 1)):
            raise ValueError(f"{v} ms is not a multiple of sched_ms")
    rng = np.random.default_rng(g["stim_seed"])
    w = mask_w(g["mask"])
    rows = [np.zeros(N), w.copy(), -w.copy()]          # 0: none, 1: q = +1, 2: q = -1
    timeline = [0]                                    # bin 0 unused (offset 0 = no modulation)
    rowq = {1: 1, -1: 2}
    enc_q = np.zeros((n_cp, g["enc_trials"]), int)
    noise_q = np.zeros((n_cp, 1 + len(g["noise_etas"]) * g["noise_reps"]), int)
    noise_eta = np.zeros(noise_q.shape)
    mem_q = np.zeros((n_cp, g["mem_bins"]), int)
    off = np.zeros((n_cp, 3))                          # enc, noise, mem slot start (s)
    pres = g["enc_present_ms"] // sb
    mbin = g["mem_bin_ms"] // sb
    for j in range(n_cp):
        # encoding: iid balanced symbols (prior stimulus therefore balanced too)
        off[j, 0] = len(timeline) * sb / 1000
        q = rng.choice([-1, 1], g["enc_trials"])
        enc_q[j] = q
        for qq in q:
            timeline += [rowq[qq]] * pres
        # noise: a clean lead-in, then each eta noise_reps times in random order
        off[j, 1] = len(timeline) * sb / 1000
        etas = np.r_[0.0, rng.permutation(np.repeat(g["noise_etas"], g["noise_reps"]))]
        q = rng.choice([-1, 1], len(etas))
        noise_q[j], noise_eta[j] = q, etas
        for qq, eta in zip(q, etas):
            if eta == 0:
                timeline += [rowq[qq]] * pres
            else:
                rows.append((1 - eta) * qq * w + eta * balanced_random_mask(rng))
                timeline += [len(rows) - 1] * pres
        # memory: iid Bernoulli(0.5) symbols, one per bin
        off[j, 2] = len(timeline) * sb / 1000
        q = 2 * rng.integers(0, 2, g["mem_bins"]) - 1
        mem_q[j] = q
        for qq in q:
            timeline += [rowq[qq]] * mbin
    return dict(rows=np.array(rows), timeline=np.array(timeline, np.int32), offsets=off,
                enc_q=enc_q, noise_q=noise_q, noise_eta=noise_eta, mem_q=mem_q)


def check_schedule(sch, b_values, A):
    """Zero-sum rows (identical expected total drive for every symbol and
    corruption level), non-negative rates, bounded modulation. Raises."""
    rows = sch["rows"]
    if not np.allclose(rows.sum(1), 0):
        raise ValueError("every pattern row must sum to zero")
    if np.abs(rows).max() > 1 + 1e-12:
        raise ValueError("|m_i| must be <= 1")
    if min(b_values) - A * np.abs(rows).max() < 0:
        raise ValueError("negative input rate (no clipping allowed)")
    return True


def add_probe_drive(net, b_hz, weight, sch, sched_ms, name="drive"):
    """The drive group replacing `inp`: one Bernoulli-per-tick (Poisson)
    train per E cell, rate b + A * rows[timeline[(t - tb + off)]]. A = 0
    gives the native drive. Shared variables A, tb, off select the branch."""
    from brian2 import NeuronGroup, Synapses, TimedArray, Hz, ms
    sym = TimedArray(sch["timeline"].astype(float), dt=sched_ms * ms)
    pat = TimedArray(sch["rows"], dt=1 * ms)
    G = NeuronGroup(N, """b : Hz (constant)
A : Hz (shared)
tb : second (shared)
off : second (shared)
rate = b + A * pat((sym(t - tb + off) + 0.5) * ms, i) : Hz""",
                    threshold="rand() < rate * dt", name=name, namespace=dict(sym=sym, pat=pat))
    G.b = np.broadcast_to(np.asarray(b_hz, float), N) * Hz
    G.A = 0 * Hz
    S = Synapses(G, net["P_E"], model="w:1", on_pre="s_AMPA_ext += w", name=f"{name.upper()}_E")
    S.connect(j="i")
    S.w[:] = float(weight)
    net.add(G, S)
    return G, S
