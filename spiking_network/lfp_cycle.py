"""Long LFP run for seizure prediction: one (seed, condition) job.

The seed's baseline is restored, HSP gets the grid's time constants, the
condition is applied, and the network runs run_s at its normal constant drive
(no signal) with HSP on, going through its own seizure cycles. The LFP proxy
(lfp.py: |I_exc| + |I_inh|, population-mean recurrent currents onto E cells)
is recorded every rec_dt.

Saves (results/lfp_cycle/raw/<seed>_<condition>.npz):
  lfp_exc, lfp_inh (float32, pA, every rec_dt), fs
  rate_t, rate_E, rate_I   -- population rates, 1 s bins
  t_k, k_AMPA, k_GABA      -- E-mean HSP gains, 1 s
"""
import json
import time
import traceback
from pathlib import Path

import numpy as np

import worker  # puts the epilepsy dir on sys.path


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


def run_lfp_job(seed, condition, baseline_path, grid, work_dir, cache_dir, out_path,
                template_cache_dir=None):
    name = condition["name"]
    tag = f"lfp seed{seed}/{name}"
    try:
        worker._isolate(work_dir, cache_dir, template_cache_dir=template_cache_dir)
        from brian2 import second, ms, Hz, seed as b2seed
        from seeded_network import build_network_seeded
        from HSP_Phase import _apply_drugs, _runkw
        from interventions import apply_weight_intervention
        import lfp

        freq = float(condition["freq_hz"])
        run_s = grid["run_s"]
        net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second,
                                   input_freq=freq, network_seed=seed)
        net.restore("baseline", filename=str(baseline_path))
        P_E = net["P_E"]
        for nm in worker.HSP_TAUS:
            setattr(P_E, nm, grid["hsp_tau_s"] * second)
        if condition.get("weights"):
            w = condition["weights"]
            apply_weight_intervention(net, w["kind"], scale=w.get("scale", 1.0))
        _apply_drugs(net, condition.get("drug") or {})
        net["inp"].rates = freq * Hz
        m_lfp, m_w = lfp.add_lfp_probe(net, rec_dt=grid["rec_dt_ms"] * ms, weight_dt=1 * second)
        b2seed(2000 + int(seed))
        t0 = float(net.t / second)
        print(f"[{tag}] start: {run_s:g} s at {freq:g} Hz, LFP every {grid['rec_dt_ms']:g} ms",
              flush=True)
        net.run(run_s * second, report=_reporter(tag), report_period=10 * second, **_runkw(net))

        _, _, exc, inh = lfp.lfp_from_monitor(m_lfp)
        n_E, n_I = int(P_E.N), int(net["P_I"].N)
        edges = t0 + np.arange(int(run_s) + 1)
        rates = []
        for mon, n in (("sp_exc", n_E), ("sp_inh", n_I)):
            st = np.asarray(net[mon].t / second)
            c, _ = np.histogram(st[st >= t0], bins=edges)
            rates.append(c / n)
        tw, kA, kG = lfp.weights_from_monitor(m_w)
        meta = dict(seed=int(seed), condition=name, freq_hz=freq, drug=condition.get("drug"),
                    t0=t0, fs=1000.0 / grid["rec_dt_ms"], **grid)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, lfp_exc=(exc * 1e12).astype(np.float32),
                            lfp_inh=(inh * 1e12).astype(np.float32),
                            rate_t=(edges[:-1] - t0 + 0.5), rate_E=rates[0], rate_I=rates[1],
                            t_k=tw - t0, k_AMPA=kA, k_GABA=kG, meta=json.dumps(meta))
        print(f"[{tag}] done -> {out_path}", flush=True)
        return dict(seed=seed, condition=name, ok=True, out_path=str(out_path))
    except Exception as exc_:  # noqa: BLE001
        print(f"[{tag}] FAILED: {exc_}", flush=True)
        return dict(seed=seed, condition=name, ok=False, error=str(exc_),
                    traceback=traceback.format_exc())
