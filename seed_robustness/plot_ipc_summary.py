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
from worker import CONDITION_PHASES


PROBE_SEIZURE_HZ = 10.0   # 5x the 2 Hz target; seizing probes run at 74-220 Hz


METRIC_LABELS = {
    "ipc_total": "IPC (ipc_total: linear d>=1 + degree 2)",
    "ipc_encoding": "IPC encoding (d = 0: input in the same step)",
    "ipc_memory": "IPC memory (d >= 1 and cross terms)",
}


def probe_ipc(result, metric="ipc_total"):
    """(ipc, seized) per probe seed for the condition's OWN 36 Hz network in
    its steady state: 'plateau' (between seizures) if it bursts, its converged
    'steady' if the treatment suppressed bursting -- never the shared 40 Hz
    reference (stored separately as 'reference'). `seized` flags probes
    during which the network went into a seizure; their IPC (~0) measures a
    seizing network, not the treated state."""
    by_phase = result.get("by_phase_raw", {})
    d = next((by_phase[p] for p in CONDITION_PHASES if p in by_phase), None)
    if not d or metric not in d:
        return None, None
    ipc = np.asarray(d[metric], float)
    return ipc, np.asarray(d["E_rate_probe"], float) > PROBE_SEIZURE_HZ


def condition_ipc_values(result, metric="ipc_total"):
    """Per-probe IPC with seizing probes removed (see probe_ipc)."""
    ipc, seized = probe_ipc(result, metric)
    return None if ipc is None else ipc[~seized]


def default_condition_order(results):
    names_present = {r["condition"] for r in results if r.get("ok")}
    ordered = [c["name"] for c in cond_mod.CONDITIONS if c["name"] in names_present]
    extra = sorted(names_present - set(ordered))  # anything not in CONDITIONS, unlikely
    return ordered + extra


V3_CONDITIONS = (["untreated", "benzo (k=1.0)"]
                 + [f"DBS {a:g}pA" for a in (5, 10, 20, 25, 30, 40, 50)]
                 + ["SV2A (-10%)", "SV2A (-15%)", "AMPA block (-10%)", "AMPA block (-15%)"])


V4_CONDITIONS = ["untreated", "benzo (k=1.0)"] + [f"DBS {a:g}pA" for a in (10, 20, 30, 40)]


def plot_ipc_across_conditions(results, out_path, condition_order=None, title=None,
                               metric="ipc_total"):
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return None

    if condition_order is None:
        condition_order = default_condition_order(ok)

    seeds = sorted({r["seed"] for r in ok})
    colors = plt.cm.tab10.colors
    seed_color = {s: colors[i % len(colors)] for i, s in enumerate(seeds)}

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(condition_order) + 3), 6))

    # untreated 40 Hz reference: one value per seed, identical in every
    # condition's JSON, so take it from any of them
    ref_by_seed = {}
    for r in ok:
        d = r.get("by_phase_raw", {}).get("reference")
        if d and metric in d and r["seed"] not in ref_by_seed:
            ref_by_seed[r["seed"]] = float(np.mean(d[metric]))
    if ref_by_seed:
        refs = np.array(list(ref_by_seed.values()))
        ax.axhspan(refs.mean() - refs.std(), refs.mean() + refs.std(), color="0.5",
                   alpha=0.2, zorder=1,
                   label=f"untreated 40 Hz reference (mean ± SD, {len(refs)} seeds)")

    means_by_seed = {s: [np.nan] * len(condition_order) for s in seeds}
    bursting_by_seed = {s: [False] * len(condition_order) for s in seeds}
    n_excluded = 0
    for xi, cond in enumerate(condition_order):
        for r in ok:
            if r["condition"] != cond:
                continue
            ipc, seized = probe_ipc(r, metric)
            if ipc is None:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(ipc))
            x = np.full(len(ipc), xi) + jitter
            ax.scatter(x[~seized], ipc[~seized], color=seed_color[r["seed"]], s=18,
                      alpha=0.75, edgecolor="none", zorder=3)
            ax.scatter(x[seized], ipc[seized], color="0.5", marker="x", s=22, zorder=3)
            n_excluded += int(seized.sum())
            if (~seized).any():
                means_by_seed[r["seed"]][xi] = float(np.mean(ipc[~seized]))
            bursting_by_seed[r["seed"]][xi] = bool(r["cycle"]["is_bursting"])
    if n_excluded:
        ax.scatter([], [], color="0.5", marker="x", s=22,
                   label=f"probe seized (E > {PROBE_SEIZURE_HZ:g} Hz), excluded ({n_excluded})")

    xs = np.arange(len(condition_order))
    for s in seeds:
        m = np.array(means_by_seed[s])
        b = np.array(bursting_by_seed[s])
        ax.plot(xs, m, color=seed_color[s], lw=1.4, label=f"seed {s}", zorder=4)
        ax.plot(xs[~b], m[~b], "o", color=seed_color[s], ms=6, zorder=5)
        ax.plot(xs[b], m[b], "o", mfc="white", mec=seed_color[s], ms=6, mew=1.5, zorder=5)
    ax.plot([], [], "o", color="0.3", label="seizure suppressed at 36 Hz")
    ax.plot([], [], "o", mfc="white", mec="0.3", mew=1.5, label="still bursting at 36 Hz")

    ax.set_xticks(xs)
    ax.set_xticklabels(condition_order, rotation=30, ha="right")
    ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)}\nof the 36 Hz network")
    ax.set_title(title or "IPC across conditions -- every probe value, every seed\n"
                          "(dots = individual probes, lines = per-seed mean)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_ipc_v3(results, out_path):
    """V3_CONDITIONS only, in that order: the IPC of each condition's steady
    state (plateau between seizures, or converged non-bursting state)."""
    present = {r["condition"] for r in results if r.get("ok")}
    return plot_ipc_across_conditions(
        results, out_path, condition_order=[c for c in V3_CONDITIONS if c in present],
        title="IPC of the steady state (between-seizure plateau, or converged if no "
              "seizures) -- every probe, every seed\n(dots = individual probes, "
              "lines = per-seed mean of non-seizing probes)")


def plot_ipc_v4(results, out_path):
    """V4_CONDITIONS only, in that order; same layout as v3 (100 s probes).
    One figure per metric: out_path for ipc_total, then *_encoding.png and
    *_memory.png when the results have the split (ipc_probe.py). Returns the
    paths written."""
    present = {r["condition"] for r in results if r.get("ok")}
    order = [c for c in V4_CONDITIONS if c in present]
    out_path = Path(out_path)
    written = []
    for metric, path in (("ipc_total", out_path),
                         ("ipc_encoding", out_path.with_name(out_path.stem + "_encoding.png")),
                         ("ipc_memory", out_path.with_name(out_path.stem + "_memory.png"))):
        if not any(probe_ipc(r, metric)[0] is not None for r in results if r.get("ok")):
            continue
        p = plot_ipc_across_conditions(
            results, path, condition_order=order, metric=metric,
            title=f"{METRIC_LABELS[metric]} of the steady state (between-seizure plateau, "
                  f"or converged if no seizures), HSP frozen, 100 s probes\n(dots = "
                  f"individual probes, lines = per-seed mean of non-seizing probes)")
        if p:
            written.append(p)
    return written


if __name__ == "__main__":
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    results = orderings.load_results(results_dir / "raw")
    for p in (plot_ipc_across_conditions(results, results_dir / "plots" / "ipc_summary.png"),
              plot_ipc_v3(results, results_dir / "plots" / "ipc_summary_v3.png"),
              *plot_ipc_v4(results, results_dir / "plots" / "ipc_summary_v4.png")):
        print(f"wrote {p}" if p else "no results found")
