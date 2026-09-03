"""IPC-across-conditions summary plot: every probe value, every seed.

Complements plot_traces.py's per-condition trace plots (raw dynamics, one
figure per condition) with the actual outcome metric side by side across
ALL conditions in one figure: x = condition (dose order untreated -> benzo
-> DBS amplitudes, from conditions.CONDITIONS), y = IPC at the reference
phase (same phase worker.run_condition_job uses for ipc_ref: 'mid' if that
condition bursts at 36Hz, else the shared 'steady' reference). Every
individual probe-seed IPC value is scattered (jittered, colored by network
seed) and each network seed also gets a mean line connecting conditions, so
you can see both the spread within a seed and whether the pattern holds
across seeds -- exactly the "does the ordering replicate" question this
whole pipeline exists to answer, in one plot.

Run standalone to (re)generate from already-saved results:
    python plot_ipc_summary.py [results_dir]
`run_seed_check.py` also calls plot_ipc_across_conditions() automatically.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import conditions as cond_mod
import orderings


def _reference_ipc_values(result):
    """Raw per-probe-seed IPC array at the same reference phase
    worker.run_condition_job uses for ipc_ref ('mid' if bursting at 36Hz,
    else the shared 'steady' reference)."""
    by_phase = result.get("by_phase_raw", {})
    d = by_phase.get("mid") or by_phase.get("steady")
    return np.asarray(d["ipc_total"], float) if d else None


def default_condition_order(results):
    names_present = {r["condition"] for r in results if r.get("ok")}
    ordered = [c["name"] for c in cond_mod.CONDITIONS if c["name"] in names_present]
    extra = sorted(names_present - set(ordered))  # anything not in CONDITIONS, unlikely
    return ordered + extra


def plot_ipc_across_conditions(results, out_path, condition_order=None):
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return None

    if condition_order is None:
        condition_order = default_condition_order(ok)

    seeds = sorted({r["seed"] for r in ok})
    colors = plt.cm.tab10.colors
    seed_color = {s: colors[i % len(colors)] for i, s in enumerate(seeds)}

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(condition_order)), 6))

    means_by_seed = {s: [np.nan] * len(condition_order) for s in seeds}
    for xi, cond in enumerate(condition_order):
        for r in ok:
            if r["condition"] != cond:
                continue
            vals = _reference_ipc_values(r)
            if vals is None or len(vals) == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), xi) + jitter, vals,
                      color=seed_color[r["seed"]], s=18, alpha=0.75,
                      edgecolor="none", zorder=3)
            means_by_seed[r["seed"]][xi] = float(np.mean(vals))

    xs = range(len(condition_order))
    for s in seeds:
        ax.plot(xs, means_by_seed[s], color=seed_color[s], lw=1.4,
               marker="o", ms=5, label=f"seed {s}", zorder=4)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(condition_order, rotation=30, ha="right")
    ax.set_ylabel("IPC (ipc_total)")
    ax.set_title("IPC across conditions -- every probe value, every seed\n"
                "(dots = individual probes, lines = per-seed mean)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    results = orderings.load_results(results_dir / "raw")
    out_path = plot_ipc_across_conditions(results, results_dir / "plots" / "ipc_summary.png")
    print(f"wrote {out_path}" if out_path else "no results found")
