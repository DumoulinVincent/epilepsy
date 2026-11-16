"""Frozen-HSP rate probe for HSP_Epileptor_trajectory.ipynb.

One probe = one short Brian2 run of the spiking network (../spiking_network) (the model of
the LFP / image-recovery results) with the HSP gains held fixed:

  restore the seed's baseline -> freeze HSP (all four time constants = 1e99 s)
  -> set every E cell's k_AMPA / k_GABA to the requested values -> drive at
  `freq_hz` -> reseed the noise (`noise_seed`; same value = same Poisson input
  streams, so +/- perturbation runs share their input realisation)
  -> run `equil_s` (discarded) + `measure_s` (mean E rate measured).

Each probe runs in its own process with a private working directory and Brian2
cache (spiking_network/worker._isolate), like the other job modules.
"""
import sys
import traceback
from pathlib import Path

import numpy as np

EPI = Path(__file__).resolve().parent
V2 = EPI.parent / "spiking_network"
for _p in (str(EPI), str(V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEIZURE_HZ = 10.0     # project seizure detector: E rate > 10 Hz in 1 s bins


def run_probe(seed, kA, kG, freq_hz, equil_s, measure_s, noise_seed, work_root):
    tag = f"s{seed}_kA{kA:.5f}_kG{kG:.5f}_n{noise_seed}"
    try:
        import worker
        work_root = Path(work_root)
        worker._isolate(work_root / "work" / tag, work_root / "cache" / tag,
                        template_cache_dir=V2 / "results" / "cython_cache" / "_template")
        from brian2 import second, Hz, seed as b2seed
        from seeded_network import build_network_seeded
        from HSP_Phase import _freeze_hsp, _runkw

        net = build_network_seeded(50 * second, 50 * second, 50 * second, 50 * second,
                                   input_freq=freq_hz, network_seed=seed)
        net.restore("baseline", filename=str(V2 / "results" / "baselines" / f"seed{seed}.bri"))
        _freeze_hsp(net)
        P_E = net["P_E"]
        P_E.k_AMPA_E_hsp = float(kA)
        P_E.k_GABA_E_hsp = float(kG)
        net["inp"].rates = freq_hz * Hz
        b2seed(int(noise_seed))
        t0 = float(net.t / second)
        net.run((equil_s + measure_s) * second, **_runkw(net))
        st = np.asarray(net["sp_exc"].t / second)
        edges = t0 + equil_s + np.arange(int(measure_s) + 1)
        per_s = np.histogram(st, bins=edges)[0] / int(P_E.N)
        eq = np.histogram(st, bins=t0 + np.arange(int(equil_s) + 1))[0] / int(P_E.N)
        return dict(ok=True, seed=seed, kA=kA, kG=kG, noise_seed=noise_seed, rate=float(per_s.mean()),
                    rate_sd_1s=float(per_s.std()), max_1s=float(max(per_s.max(), eq.max(initial=0))),
                    runaway=bool(max(per_s.max(), eq.max(initial=0)) > SEIZURE_HZ))
    except Exception as exc:  # noqa: BLE001
        return dict(ok=False, seed=seed, kA=kA, kG=kG, noise_seed=noise_seed, error=str(exc),
                    traceback=traceback.format_exc())
