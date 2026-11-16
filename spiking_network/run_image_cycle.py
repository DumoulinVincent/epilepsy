"""Image-recovery runs (image_cycle.py) in parallel, then the analysis
(image_analysis.py). Nothing runs unless invoked; --dry-run prints the plan.

    # smoke test: 1 seed, 2 conditions, 600 s (a few minutes)
    python -u run_image_cycle.py --smoke
    # pilot: 1 seed, the default conditions, the full run length
    python -u run_image_cycle.py --seeds 1
    # full: 5 seeds x default conditions
    python -u run_image_cycle.py
    # options
    python -u run_image_cycle.py --library lines --conditions "untreated,DBS 60pA" --run-s 3000
    python -u run_image_cycle.py --analyse-only [--fast]
    python -u run_image_cycle.py --dry-run

Default conditions: conditions.IMAGE_DEFAULT (40 Hz reference, untreated,
benzo, DBS 20 and 60 pA); every condition in conditions.IMAGE_CONDITIONS is
selectable. A job is skipped if its result exists (--force reruns).
Results: results/image_<library>/raw/*.npz, then .../analysis/.
"""
import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import worker  # noqa: E402
import conditions as cond_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--conditions", default=None,
                    help=f"comma-separated names (default: {', '.join(cond_mod.IMAGE_DEFAULT)})")
    ap.add_argument("--library", default=None, choices=("random", "lines"))
    ap.add_argument("--run-s", type=float, default=None)
    ap.add_argument("--contrast", type=float, default=None, help="image contrast a (Hz)")
    ap.add_argument("--n-jobs", type=int, default=None, help="default: all cores minus 2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="seed 1, reference + untreated, 600 s (settling shortened to 100 s)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--fast", action="store_true", help="analysis: local/count decoders, adapted + frozen only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    g = dict(cond_mod.IMAGE_GRID)
    if args.library:
        g["library"] = args.library
    if args.contrast is not None:
        g["contrast_hz"] = args.contrast
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    names = [c.strip() for c in args.conditions.split(",")] if args.conditions else list(cond_mod.IMAGE_DEFAULT)
    if args.smoke:
        seeds, names = [1], ["reference 40Hz", "untreated"]
        g.update(run_s=600.0, settle_s=100.0)
    if args.run_s:
        g["run_s"] = args.run_s
    conds = [c for c in cond_mod.IMAGE_CONDITIONS if c["name"] in names]
    # any other DBS amplitude, e.g. "DBS 300pA": built like the listed ones (36 Hz drive)
    import re
    for nm in names:
        m = re.fullmatch(r"DBS (\d+(?:\.\d+)?)pA", nm)
        if m and nm not in {c["name"] for c in conds}:
            conds.append(dict(cond_mod.make_condition(nm, drug={"dbs": True, "dbs_freq_hz": cond_mod.DBS_FREQ_HZ,
                                                                "dbs_amp_pA": float(m.group(1))}),
                              freq_hz=cond_mod.BURST_FREQ_HZ))
    missing = set(names) - {c["name"] for c in conds}
    if missing:
        raise SystemExit(f"unknown condition(s): {sorted(missing)}")
    if args.contrast is not None and args.contrast != cond_mod.IMAGE_GRID["contrast_hz"] and not args.out:
        raise SystemExit("a non-default contrast must go to its own --out directory")
    results = THIS_DIR / "results"
    out_dir = Path(args.out).resolve() if args.out else results / (
        f"image_{g['library']}" + ("_smoke" if args.smoke else ""))
    raw = out_dir / "raw"

    jobs = []
    for s in seeds:
        for c in conds:
            slug = worker._slug(f"seed{s}_{c['name']}")
            out = raw / f"{slug}.npz"
            if out.exists() and not args.force:
                print(f"[run] {slug}: using cached {out}")
                continue
            jobs.append((s, c, slug, out))
    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 4) - 2)
    mb = g["run_s"] * 100 / g["sched_ms"] * 1000 / 1e6
    print(f"[plan] {len(jobs)} job(s) -> {out_dir}; {g['run_s']:g} s each, library {g['library']}, "
          f"contrast {g['contrast_hz']:g} Hz, {min(n_jobs, max(len(jobs), 1))} at a time\n"
          f"[plan] data: E counts {mb:.0f} MB per run uncompressed (10 ms bins x 100 cells, uint8), "
          f"mostly zeros -> roughly {0.12 * mb:.0f}-{0.3 * mb:.0f} MB compressed per run "
          f"({len(jobs) * 0.3 * mb / 1000:.1f} GB at most for these jobs); analysis files a few MB per run",
          flush=True)
    for s, c, slug, _ in jobs:
        print(f"[plan]   seed {s}: {c['name']} ({c['freq_hz']:g} Hz)")
    if args.dry_run:
        return
    if jobs and not args.analyse_only:
        from joblib import Parallel, delayed
        import image_cycle
        res = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(image_cycle.run_image_job)(
                seed=s, condition=c, baseline_path=str(results / "baselines" / f"seed{s}.bri"),
                grid=g, work_dir=str(results / "work" / f"{out_dir.name}_{slug}"),
                cache_dir=str(results / "cython_cache" / f"{out_dir.name}_{slug}"), out_path=str(out),
                template_cache_dir=str(results / "cython_cache" / "_template"))
            for s, c, slug, out in jobs)
        for r in res:
            if not r["ok"]:
                print(f"[run] seed {r['seed']} / {r['condition']} FAILED:\n{r['traceback']}")
    import image_analysis       # reads the protocol saved in the run files
    image_analysis.main(out_dir, n_jobs=args.n_jobs, fast=args.fast)


if __name__ == "__main__":
    main()
