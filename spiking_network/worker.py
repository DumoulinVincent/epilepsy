"""Per-process helpers shared by the job modules (cycle_ipc, lfp_cycle,
image_cycle): each job runs in its own OS process (joblib/loky) with its own
working directory and Brian2 Cython cache, seeded by hardlinks from a
compiled template cache so identical code isn't recompiled per job.
"""
import shutil
import sys
import traceback
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR.parent / "model"      # HSP_Phase, hsp_capacity, hsp_metrics
for _p in (str(MODEL_DIR), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# the four HSP time constants of the E population
HSP_TAUS = ("tau_ins_AMPA_E", "tau_int_AMPA_E", "tau_ins_GABA_E", "tau_int_GABA_E")


def _slug(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")


def _seed_cache_from_template(cache_dir, template_cache_dir):
    """Populate a fresh per-process cache dir from an already-compiled
    template cache, via hardlinks (instant, no recompilation) rather than a
    real copy. Every (seed, condition) job builds the SAME network structure
    -- only runtime weight values and drug-set state variables differ, never
    the generated Cython/C++ source -- so the compiled .so's from one warm-up
    build are valid for every job. Safe to hardlink because brian2 never
    mutates a cached file in place: a structural change produces a new
    content hash and therefore a new filename."""
    import os
    template_cache_dir = Path(template_cache_dir)
    if not template_cache_dir.exists():
        return
    # Only the compiled extensions matter for a cache hit; the cache dir also
    # accumulates a "tmp/<full source path>/" tree of intermediate .cpp/.o
    # build files (a pyximport quirk when the cache dir lives under /tmp) and
    # 0-byte .lock marker files -- skip those, we don't need them and some
    # are directories os.link()/copy2() can't handle anyway.
    for src in template_cache_dir.glob("*.so"):
        dst = Path(cache_dir) / src.name
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _isolate(work_dir, cache_dir, template_cache_dir=None):
    """chdir into `work_dir` and point brian2's cython cache at `cache_dir`,
    both private to this process. Must run before any brian2 code generation
    (i.e. before the first net.run()). If `template_cache_dir` is given (see
    `prewarm_cache_job`), hardlink its already-compiled extensions in first
    so this process doesn't pay to recompile identical code from scratch."""
    import os
    from brian2 import prefs

    work_dir = Path(work_dir)
    cache_dir = Path(cache_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if template_cache_dir is not None:
        _seed_cache_from_template(cache_dir, template_cache_dir)
    os.chdir(work_dir)
    prefs.codegen.runtime.cython.cache_dir = str(cache_dir)
    prefs.codegen.runtime.cython.multiprocess_safe = False


def prewarm_cache_job(template_cache_dir, work_dir, network_seed=0, tau_s=50.0):
    """Compile every code object the network needs exactly once, into
    `template_cache_dir`. Run this single-process, before fanning out the
    (seed, condition) grid, so each of those parallel processes can hardlink
    the results instead of independently recompiling ~dozens of Cython
    extensions (which is what made the very first build in this environment
    take the better part of an hour)."""
    try:
        _isolate(work_dir, template_cache_dir)
        from brian2 import second
        from seeded_network import build_network_seeded
        build_network_seeded(
            tau_ins_AMPA=tau_s * second, tau_int_AMPA=tau_s * second,
            tau_ins_GABA=tau_s * second, tau_int_GABA=tau_s * second,
            network_seed=network_seed,
        )
        return dict(ok=True, template_cache_dir=str(template_cache_dir))
    except Exception as exc:  # noqa: BLE001 - report, don't kill the batch
        return dict(ok=False, error=str(exc), traceback=traceback.format_exc())
