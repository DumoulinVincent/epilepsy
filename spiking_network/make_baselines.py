"""Step 0 of every simulation: one calibrated network state per seed.

Each network seed (1-5) is built and run for 15000 s at the healthy 40 Hz drive with all four HSP time constants at
50 s, so that HSP settles at its ~2 Hz fixed point. The state is stored as results/baselines/seed<s>.bri; every
other run (LFP, image task, slow-HSP cycles) restores it, then sets its own time constants and condition.

    python make_baselines.py                 # seeds 1-5, skips seeds that already have a baseline
    python make_baselines.py --seeds 1 --force

Each seed takes about 1 h. A .bri file is ~165 MB (it holds the full monitor history), so baselines are not in git.
The calibration trace (E/I rates and HSP gains over the settling run) is saved next to it as seed<s>_trace.npz.
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import worker  # noqa: E402  (puts ../model on sys.path)

RESULTS = THIS_DIR / "results"


def calibrate_job(seed, out_dir, calib_time_s):
    try:
        worker._isolate(RESULTS / "work" / f"calib_seed{seed}", RESULTS / "cython_cache" / f"calib_seed{seed}",
                        template_cache_dir=RESULTS / "cython_cache" / "_template")
        from seeded_network import calibrate_baseline
        path = calibrate_baseline(seed, out_dir / f"seed{seed}.bri", calib_time_s=calib_time_s,
                                  trace_out_path=out_dir / f"seed{seed}_trace.npz")
        return dict(seed=seed, ok=True, path=path)
    except Exception as e:  # noqa: BLE001
        return dict(seed=seed, ok=False, error=f"{e}\n{traceback.format_exc()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--calib-time-s", type=float, default=15000.0)
    ap.add_argument("--n-jobs", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="recalibrate seeds that already have a baseline")
    args = ap.parse_args()
    out_dir = RESULTS / "baselines"
    seeds = [int(s) for s in args.seeds.split(",")]
    todo = [s for s in seeds if args.force or not (out_dir / f"seed{s}.bri").exists()]
    if not todo:
        print("all baselines present:", ", ".join(str(out_dir / f"seed{s}.bri") for s in seeds))
        return
    template = RESULTS / "cython_cache" / "_template"
    if not template.exists():
        print("[prewarm] compiling the network once into", template, flush=True)
        r = worker.prewarm_cache_job(template, RESULTS / "work" / "_prewarm")
        if not r["ok"]:
            raise SystemExit(r["traceback"])
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=min(args.n_jobs, len(todo)), backend="loky")(
        delayed(calibrate_job)(s, out_dir, args.calib_time_s) for s in todo)
    for r in res:
        print(f"seed {r['seed']}: " + (r["path"] if r["ok"] else "FAILED\n" + r["error"]))


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
