"""HSP-active functional assays: run (seed, condition) jobs, then analyse.

    python -u run_assays.py --smoke              # seed 1, reference + untreated, 700 s
    python -u run_assays.py --seeds 1            # pilot: default conditions, 8000 s
    python -u run_assays.py                      # 5 seeds x default conditions
    python -u run_assays.py --analyse-only
    python -u run_assays.py --dry-run

Default conditions: conditions.ASSAY_DEFAULT; any condition of
conditions.ASSAY_CONDITIONS or any "DBS <amp>pA" is selectable. The probe
amplitude is A = amplitude_frac x the smallest native drive among ALL assay
conditions (36 Hz -> 7.2 Hz), identical for every condition and every
subset that is run; --amplitude-hz overrides it (use its own --out).
"""
import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import worker  # noqa: E402
import conditions as C  # noqa: E402


def resolve_conditions(names):
    conds = [c for c in C.ASSAY_CONDITIONS if c["name"] in names]
    for nm in names:
        m = re.fullmatch(r"DBS (\d+(?:\.\d+)?)pA", nm)
        if m and nm not in {c["name"] for c in conds}:
            conds.append(dict(C.make_condition(nm, drug={"dbs": True, "dbs_freq_hz": C.DBS_FREQ_HZ,
                                                         "dbs_amp_pA": float(m.group(1))}),
                              freq_hz=C.BURST_FREQ_HZ))
    missing = set(names) - {c["name"] for c in conds}
    if missing:
        raise SystemExit(f"unknown condition(s): {sorted(missing)}")
    return conds


def _run_spec(spec, log, retries=2):
    """One job in its own Python process (a native crash cannot take the
    batch down); retried on failure."""
    import json
    import subprocess
    spec_path = Path(spec["out_path"]).with_suffix(".job.json")
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec))
    for attempt in range(1 + retries):
        with open(log, "a") as fh:
            r = subprocess.run([sys.executable, "-u", str(THIS_DIR / "assay_sim.py"), str(spec_path)],
                               stdout=fh, stderr=subprocess.STDOUT, cwd=THIS_DIR,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        if r.returncode == 0 and Path(spec["out_path"]).exists():
            spec_path.unlink()
            return True, attempt
        print(f"[run] {Path(spec['out_path']).stem}: attempt {attempt + 1} failed (exit {r.returncode})",
              flush=True)
    return False, retries


def run_jobs(jobs, g, A, results, out_dir, n_jobs):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    template = results / "cython_cache" / "_template"
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    gj = {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()}
    if not (template / ".assay_prewarmed").exists():
        print("[run] compiling the assay code once into the template cache ...", flush=True)
        warm = Path(tempfile.mkdtemp())
        s, c, _, _ = jobs[0]
        ok, _ = _run_spec(dict(seed=1, condition=c, baseline_path=str(results / "baselines" / "seed1.bri"),
                               grid=dict(gj, run_s=g["settle_s"] + 30.0), A_hz=A, work_dir=str(warm / "w"),
                               cache_dir=str(template), out_path=str(warm / "warm.npz"), n_checkpoints=1),
                          logs / "prewarm.log", retries=3)
        if not ok:
            raise SystemExit(f"prewarm failed, see {logs / 'prewarm.log'}")
        (template / ".assay_prewarmed").touch()
    specs = [dict(seed=s, condition=c, baseline_path=str(results / "baselines" / f"seed{s}.bri"), grid=gj,
                  A_hz=A, work_dir=str(results / "work" / f"{out_dir.name}_{slug}"),
                  cache_dir=str(results / "cython_cache" / f"{out_dir.name}_{slug}"), out_path=str(out),
                  template_cache_dir=str(template)) for s, c, slug, out in jobs]
    print(f"[run] {len(specs)} job(s), {n_jobs} at a time; per-job logs in {logs}", flush=True)
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_run_spec, sp, logs / f"{Path(sp['out_path']).stem}.log"): sp for sp in specs}
        for i, fu in enumerate(as_completed(futs), 1):
            ok, att = fu.result()
            retry = f" (after {att} retries)" if att else ""
            print(f"[run] {i}/{len(specs)} {Path(futs[fu]['out_path']).stem}: "
                  f"{'ok' if ok else 'FAILED'}{retry}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--run-s", type=float, default=None)
    ap.add_argument("--amplitude-hz", type=float, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    g = dict(C.ASSAY_GRID)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    names = [c.strip() for c in args.conditions.split(",")] if args.conditions else list(C.ASSAY_DEFAULT)
    if args.smoke:
        seeds, names = [1], ["reference 40Hz", "untreated"]
        g.update(run_s=700.0, settle_s=100.0)
    if args.run_s:
        g["run_s"] = args.run_s
    conds = resolve_conditions(names)
    A = args.amplitude_hz if args.amplitude_hz is not None else \
        g["amplitude_frac"] * min(c["freq_hz"] for c in C.ASSAY_CONDITIONS)
    if args.amplitude_hz is not None and not args.out:
        raise SystemExit("a non-default amplitude must go to its own --out directory")
    results = THIS_DIR / "results"
    out_dir = Path(args.out).resolve() if args.out else results / ("assays_smoke" if args.smoke else "assays")
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
    import assay_stimulus as ast
    n_cp = len(ast.checkpoint_times(g))
    print(f"[plan] {len(jobs)} job(s) -> {out_dir}; {g['run_s']:g} s native + {n_cp} checkpoints x "
          f"{sum(ast.branch_durations(g).values()):.1f} s of branches; A = {A:g} Hz; "
          f"{min(n_jobs, max(len(jobs), 1))} at a time", flush=True)
    for s, c, slug, _ in jobs:
        print(f"[plan]   seed {s}: {c['name']} (b = {c['freq_hz']:g} Hz)")
    if args.dry_run:
        return
    if jobs and not args.analyse_only:
        run_jobs(jobs, g, A, results, out_dir, n_jobs)
    import assay_analysis
    assay_analysis.main(out_dir, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
