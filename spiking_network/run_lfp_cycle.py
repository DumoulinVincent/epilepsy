"""Long LFP runs (lfp_cycle.py) for every (seed, condition) in parallel, then
the pre-seizure vs plateau analysis (lfp_cycle_analysis.py).

    python -u run_lfp_cycle.py --n-jobs 15
    python -u run_lfp_cycle.py --analyse-only

A job is skipped if its result exists (--force reruns).
Results: results/lfp_cycle/raw/*.npz, then results/lfp_cycle/.
"""
import argparse
import os
import re
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
                    help="comma-separated names (default: conditions.LFP_CONDITIONS; any "
                         "conditions.CYCLE_CONDITIONS name, e.g. 'reference 40Hz', is accepted)")
    ap.add_argument("--n-jobs", type=int, default=None, help="default: all cores minus 2")
    ap.add_argument("--run-s", type=float, default=None, help="override LFP_GRID run_s")
    ap.add_argument("--out", default=None, help="default: results/lfp_cycle")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--hsp-tau", type=float, default=None, help="override LFP_GRID hsp_tau_s (s)")
    ap.add_argument("--skip-analysis", action="store_true",
                    help="no pre-seizure vs plateau analysis after the runs")
    args = ap.parse_args()

    results = THIS_DIR / "results"
    out_dir = Path(args.out).resolve() if args.out else results / "lfp_cycle"
    raw = out_dir / "raw"
    if not args.analyse_only:
        from joblib import Parallel, delayed
        import lfp_cycle
        g = dict(cond_mod.LFP_GRID)
        if args.run_s:
            g["run_s"] = args.run_s
        if args.hsp_tau:
            g["hsp_tau_s"] = args.hsp_tau
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        conds = cond_mod.LFP_CONDITIONS
        if args.conditions:
            want = {c.strip() for c in args.conditions.split(",")}
            conds = [c for c in cond_mod.CYCLE_CONDITIONS if c["name"] in want]
            # any other DBS amplitude, e.g. "DBS 300pA": built like the listed ones (36 Hz drive)
            for nm in sorted(want - {c["name"] for c in conds}):
                m = re.fullmatch(r"DBS (\d+(?:\.\d+)?)pA", nm)
                if m:
                    conds.append(dict(cond_mod.make_condition(nm, drug={
                        "dbs": True, "dbs_freq_hz": cond_mod.DBS_FREQ_HZ, "dbs_amp_pA": float(m.group(1))}),
                        freq_hz=cond_mod.BURST_FREQ_HZ))
            if want - {c["name"] for c in conds}:
                raise SystemExit(f"unknown condition(s): {sorted(want - {c['name'] for c in conds})}")
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
        print(f"[run] {len(jobs)} LFP job(s), {min(n_jobs, max(len(jobs), 1))} at a time; "
              f"{g['run_s']:g} s each, HSP tau {g['hsp_tau_s']:g} s", flush=True)
        if jobs:
            res = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                delayed(lfp_cycle.run_lfp_job)(
                    seed=s, condition=c,
                    baseline_path=str(results / "baselines" / f"seed{s}.bri"), grid=g,
                    work_dir=str(results / "work" / f"{out_dir.name}_{slug}"),
                    cache_dir=str(results / "cython_cache" / f"{out_dir.name}_{slug}"),
                    out_path=str(out),
                    template_cache_dir=str(results / "cython_cache" / "_template"))
                for s, c, slug, out in jobs)
            for r in res:
                if not r["ok"]:
                    print(f"[run] seed {r['seed']} / {r['condition']} FAILED:\n{r['traceback']}")
    if not args.skip_analysis:
        import lfp_cycle_analysis
        lfp_cycle_analysis.main(out_dir)


if __name__ == "__main__":
    main()
