"""Image-recovery run: one (seed, condition) job (conditions.IMAGE_GRID).

The seed's baseline is restored, HSP gets the grid's time constants, the
condition is applied exactly as in cycle_ipc.py (weights, drug/benzo, DBS as
the existing separate 120 Hz Poisson current), and the ordinary drive `inp`
is switched off: the image source (image_stimulus.add_image_input) carries
the whole drive, f during blanks and f + a (p - q) during images. HSP stays
on; the network runs run_s in ONE net.run through its own seizure cycles.

Saved (one .npz per run, everything needed to reanalyse without simulating):
  E_counts  (n_bins, 100) uint8/uint16 -- E spikes per sched_ms bin
  I_counts  (n_bins,)     uint16       -- all I spikes per bin
  masks, mask_names, onset, offset, blank, mask_id (schedule bins)
  t_k, k_AMPA, k_GABA (200 ms monitors), seizures (onset, offset in s, same detector as
  the analysis), meta (resolved configuration, code hash, completion status)
"""
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

import worker  # puts the epilepsy dir on sys.path

THIS_DIR = Path(__file__).resolve().parent


def _code_version():
    h = hashlib.sha1()
    for f in ("image_cycle.py", "image_stimulus.py", "seeded_network.py", "conditions.py"):
        h.update((THIS_DIR / f).read_bytes())
    return h.hexdigest()[:12]


def _reporter(tag, every=0.1):
    nxt = [0.0]
    t_wall = time.time()

    def _cb(elapsed, completed, start, duration):
        if completed + 1e-9 >= nxt[0]:
            nxt[0] += every
            el = time.time() - t_wall
            eta = el * (1 - completed) / completed if completed > 0 else float("nan")
            print(f"[{tag}] {100 * completed:5.1f}%  {el:6.0f}s wall, ETA {eta:6.0f}s", flush=True)
    return _cb


def detect_seizures(E_counts, bin_ms, n_E, thr_hz):
    """(onset_s, offset_s) of runs of 1 s bins whose E rate exceeds thr_hz."""
    nb = 1000 // bin_ms
    T = len(E_counts) // nb
    r1 = E_counts[:T * nb].sum(1).reshape(T, nb).sum(1) / n_E
    above = np.r_[False, r1 > thr_hz, False]
    d = np.diff(above.astype(int))
    return np.array([(float(a), float(b)) for a, b in
                     zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]).reshape(-1, 2)


def run_image_job(seed, condition, baseline_path, grid, work_dir, cache_dir, out_path,
                  template_cache_dir=None):
    name = condition["name"]
    tag = f"image seed{seed}/{name}"
    try:
        worker._isolate(work_dir, cache_dir, template_cache_dir=template_cache_dir)
        from brian2 import second, ms, Hz, seed as b2seed
        from seeded_network import build_network_seeded
        from HSP_Phase import _apply_drugs, _runkw
        from interventions import apply_weight_intervention
        import image_stimulus as ist

        g = dict(grid)
        freq, a = float(condition["freq_hz"]), float(g["contrast_hz"])
        net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second,
                                   input_freq=freq, network_seed=seed)
        net.restore("baseline", filename=str(baseline_path))
        P_E = net["P_E"]
        dt_ms = float(P_E.clock.dt / ms)
        if abs(dt_ms - g["dt_ms"]) > 1e-9 or g["sched_ms"] % g["dt_ms"]:
            raise ValueError(f"integration step {dt_ms} ms does not match the grid ({g['dt_ms']} ms)")
        for nm in worker.HSP_TAUS:
            setattr(P_E, nm, g["hsp_tau_s"] * second)
        if condition.get("weights"):
            w = condition["weights"]
            apply_weight_intervention(net, w["kind"], scale=w.get("scale", 1.0))
        _apply_drugs(net, condition.get("drug") or {})
        w_drive = float(np.mean(net["PG_E"].w[:]))
        net["inp"].rates = 0 * Hz                        # the image source carries the drive

        masks, names = ist.make_library(g)
        sch = ist.make_schedule(len(masks), g["run_s"], g)
        table = ist.rate_table(masks, freq, a, g["k_white"])
        ist.add_image_input(net, sch["img_series"], table, g["sched_ms"], w_drive)
        b2seed(3000 + int(seed))                         # same Poisson streams for every condition of a seed
        t0 = float(net.t / second)
        print(f"[{tag}] start: {g['run_s']:g} s, {len(sch['onset'])} trials, library "
              f"{g['library']} ({len(masks)} masks), f {freq:g} Hz, a {a:g} Hz", flush=True)
        net.run(g["run_s"] * second, report=_reporter(tag), report_period=10 * second, **_runkw(net))

        n_bins = sch["n_bins"]
        tpb = int(round(g["sched_ms"] / g["dt_ms"]))
        n_E, n_I = int(P_E.N), int(net["P_I"].N)
        st, si = np.asarray(net["sp_exc"].t / second), np.asarray(net["sp_exc"].i)
        keep = st >= t0 - 1e-9                           # post-restore spikes only
        k = ist.tick_bins(st[keep], t0, g["dt_ms"] / 1000, tpb)
        ok = (k >= 0) & (k < n_bins)
        E = np.bincount(k[ok] * n_E + si[keep][ok], minlength=n_bins * n_E).reshape(n_bins, n_E)
        stI = np.asarray(net["sp_inh"].t / second)
        kI = ist.tick_bins(stI[stI >= t0 - 1e-9], t0, g["dt_ms"] / 1000, tpb)
        I = np.bincount(kI[(kI >= 0) & (kI < n_bins)], minlength=n_bins)
        mA, mG = net["k_AMPA_E_hsp"], net["k_GABA_E_hsp"]
        kt = np.asarray(mA.t / second)
        sel = kt >= t0 - 1e-9
        kt_rel = kt[sel] - t0
        kA = np.asarray(mA.k_AMPA_E_hsp)[:, sel].mean(0)
        kG = np.asarray(mG.k_GABA_E_hsp)[:, sel].mean(0)
        sz = detect_seizures(E, g["sched_ms"], n_E, g["seizure_hz"])

        meta = dict(seed=int(seed), condition=name, freq_hz=freq, contrast_hz=a,
                    drug=condition.get("drug"), weights=condition.get("weights"),
                    drive_weight=w_drive, dt_ms=dt_ms, t0_s=t0, n_E=n_E, n_I=n_I,
                    baseline=str(baseline_path), sim_seed=3000 + int(seed),
                    n_trials=int(len(sch["onset"])), code_version=_code_version(),
                    grid={k_: (list(v) if isinstance(v, tuple) else v) for k_, v in g.items()},
                    completed=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        dtype = np.uint8 if E.max() < 256 else np.uint16
        np.savez_compressed(out_path, E_counts=E.astype(dtype), I_counts=I.astype(np.uint16),
                            masks=masks, mask_names=np.array(names), onset=sch["onset"],
                            offset=sch["offset"], blank=sch["blank"], mask_id=sch["mask_id"],
                            t_k=kt_rel, k_AMPA=kA, k_GABA=kG,
                            seizures=sz, meta=json.dumps(meta))
        print(f"[{tag}] done: E {E.sum() / (n_E * g['run_s']):.2f} Hz, {len(sz)} seizures -> {out_path}",
              flush=True)
        return dict(seed=seed, condition=name, ok=True, out_path=str(out_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] FAILED: {exc}", flush=True)
        return dict(seed=seed, condition=name, ok=False, error=str(exc),
                    traceback=traceback.format_exc())
