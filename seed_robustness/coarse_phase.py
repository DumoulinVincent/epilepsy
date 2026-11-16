"""Which network state each IPC probe is run on.

characterize_cycle() and _ipc_probe() come from HSP_Phase.py unmodified; the
snapshot selection does not. One rule for every condition, bursting or not
(_plateau_snapshot): run with HSP on until the E-population mean k_AMPA has
varied by less than `plateau_tol` (max - min) over the last
`plateau_window_s`, then snapshot that state. A seizure crashes k_AMPA, so a
window containing one can't pass -- no seizure detection is needed. The range
(not end minus start) is used so a seizure plus recovery inside the window
can't sneak through.

Probes run through ipc_probe.ipc_probe (HSP_Phase._ipc_probe plus the
encoding/memory split). HSP is frozen during every probe (it sets all four HSP time constants
to 1e99 s after restoring the snapshot); drugs, DBS and weight interventions
are part of the stored state and stay on throughout.
"""
import numpy as np

import probe_plots
from HSP_Phase import _apply_drugs, _runkw, characterize_cycle
from ipc_probe import ipc_probe
from hsp_capacity import make_input_signal
from progress import Stage


def _restore_and_drive(net, input_freq_hz, drug):
    from brian2 import Hz
    net.restore("baseline", filename="baseline_2hz.bri")
    net["inp"].rates = input_freq_hz * Hz
    _apply_drugs(net, drug)
    return _runkw(net)


def _plateau_snapshot(net, input_freq_hz, drug, window_s, tol, chunk_s, max_s,
                      store_name):
    """Snapshot once mean k_AMPA's range over the last window_s is < tol.
    k is read straight from the group every chunk_s (no monitor copies)."""
    from brian2 import second
    rk = _restore_and_drive(net, input_freq_hz, drug)
    P = net["P_E"]
    t0 = float(net.t / second)
    ts, ks = [t0], [float(np.mean(P.k_AMPA_E_hsp[:]))]
    while ts[-1] - t0 < max_s:
        net.run(chunk_s * second, **rk)
        ts.append(float(net.t / second))
        ks.append(float(np.mean(P.k_AMPA_E_hsp[:])))
        if ts[-1] - t0 < window_s:
            continue
        w = np.asarray(ks)[np.asarray(ts) >= ts[-1] - window_s - 1e-9]
        if w.max() - w.min() < tol:
            net.store(store_name)
            return dict(found=True, store_name=store_name, settle_s=ts[-1] - t0,
                        converged=True, k_range=float(w.max() - w.min()),
                        k_AMPA=ks[-1], k_GABA=float(np.mean(P.k_GABA_E_hsp[:])))
    return dict(found=False, status=f"k_AMPA never stayed within {tol:g} over "
                                    f"{window_s:g} s in {max_s:g} s")


def phase_capacity_seeded_coarse(net, input_freq_hz, seeds, drug=None,
                                 n_steps=2000, amp_hz=4.0, step_s=0.05, max_delay=40,
                                 probe_r0=None, char_kw=None,
                                 plateau_window_s=750.0, plateau_tol=0.06,
                                 plateau_chunk_s=25.0, plateau_max_s=5000.0,
                                 inspect_extend_s=None, progress_tag=None):
    """Characterize the cycle (for the cycle stats and traces), snapshot the
    plateau, probe it with every input seed. Returns the same structure as
    HSP_Phase.phase_capacity_seeded(), with a single 'plateau' phase.

    `inspect_extend_s`, if not None, also collects the probe inspection data
    (probe_plots.py) into the result's 'inspect' key: after the snapshot the
    run continues inspect_extend_s with HSP on (context figure only; every
    probe restores the snapshot), then every probe's binned E/I rates and k.

    `progress_tag`, if given, prints start/done wall-clock markers around each
    step so you can see which step a job is on."""
    drug = drug or {}
    char_kw = char_kw or {}
    r0 = input_freq_hz if probe_r0 is None else probe_r0
    tag = f"{progress_tag}/{input_freq_hz:g}Hz" if progress_tag else None

    with _stage(tag, "characterizing cycle"):
        cyc = characterize_cycle(net, input_freq_hz, drug=drug, **char_kw)

    by_phase = {}
    ptag = f"{tag}/plateau" if tag else None
    with _stage(ptag, f"plateau snapshot: k_AMPA range < {plateau_tol:g} over "
                      f"{plateau_window_s:g} s (cap {plateau_max_s:g} s)"):
        snap = _plateau_snapshot(net, input_freq_hz, drug, plateau_window_s, plateau_tol,
                                 plateau_chunk_s, plateau_max_s, store_name="phase_plateau")
    if not snap["found"]:
        by_phase["plateau"] = dict(found=False, status=snap["status"])
        return dict(cycle=cyc, by_phase=by_phase, input_freq_hz=input_freq_hz, drug=drug)

    inspect = None
    if inspect_extend_s is not None:
        from brian2 import second
        t_snap = float(net.t / second)
        if inspect_extend_s > 0:
            with _stage(ptag, f"inspection: {inspect_extend_s:g} s past the snapshot, HSP on"):
                net.run(inspect_extend_s * second, **_runkw(net))
        inspect = probe_plots.context_data(net, t_snap - snap["settle_s"], t_snap,
                                           plateau_window_s, snap)
        inspect["meta"].update(plateau_tol=plateau_tol, probe_r0_hz=r0, amp_hz=amp_hz,
                               step_s=step_s, n_steps=n_steps, input_freq_hz=input_freq_hz)

    acc = {k: [] for k in ("ipc_total", "ipc_encoding", "ipc_memory", "linear_d0",
                           "linear_mc", "nonlinear", "forget_curve", "E_rate_probe",
                           "state_rank", "participation_ratio", "burst_during_probe")}
    with _stage(ptag, f"probing {len(seeds)} seed(s)"):
        for s in seeds:
            u = make_input_signal(n_steps, seed=int(s))
            r = ipc_probe(net, snap["store_name"], u, r0_hz=r0, amp_hz=amp_hz,
                          step_s=step_s, max_delay=max_delay, rng_seed=int(s))
            if inspect is not None:
                probe_plots.add_probe(inspect, net, t_snap, n_steps, step_s,
                                      np.clip(r0 + amp_hz * np.asarray(u, float), 0.0, None),
                                      s, r)
            for k in acc:
                acc[k].append(r[k])
    acc = {k: np.array(v) for k, v in acc.items()}
    acc.update(found=True, seeds=list(seeds), k_AMPA=snap["k_AMPA"], k_GABA=snap["k_GABA"],
               k_range=snap["k_range"], settle_s=snap["settle_s"], converged=True)
    by_phase["plateau"] = acc
    return dict(cycle=cyc, by_phase=by_phase, input_freq_hz=input_freq_hz, drug=drug,
                inspect=inspect)


def _stage(tag, label):
    if tag is None:
        return _NullStage()
    return Stage(tag, label)


class _NullStage:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
