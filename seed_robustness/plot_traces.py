"""Auto-plot the raw cycle traces worker.run_condition_job() saves (when given
`trace_out_path`) to results/traces/*.npz -- one PNG per condition, all seeds
overlaid, so you can eyeball "does this look like the same dynamics" rather
than only reading summary statistics.

Run standalone to (re)generate plots from already-saved traces without
rerunning any simulation:
    python plot_traces.py [results_dir]
`run_seed_check.py` also calls `plot_all()` automatically at the end of a run.
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_traces(traces_dir):
    traces = []
    for p in sorted(Path(traces_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        traces.append({k: d[k] for k in d.files} | {"_path": p})
    return traces


def _slug(name):
    return "".join(c if c.isalnum() else "_" for c in str(name)).strip("_")


def group_by_condition(traces):
    by_cond = defaultdict(list)
    for tr in traces:
        by_cond[str(tr["condition"])].append(tr)
    for cond in by_cond:
        by_cond[cond].sort(key=lambda tr: int(tr["seed"]))
    return dict(by_cond)


def load_calibration_traces(traces_dir):
    """Pre-drug baseline settling traces, saved separately under
    traces/calibration/ (not picked up by load_traces()'s condition glob)."""
    traces = []
    for p in sorted(Path(traces_dir / "calibration").glob("seed*.npz")):
        d = np.load(p, allow_pickle=False)
        traces.append({k: d[k] for k in d.files})
    traces.sort(key=lambda tr: int(tr["seed"]))
    return traces


_ROWS = (("rE", "E rate (Hz)"), ("rI", "I rate (Hz)"),
        ("kA", "k_AMPA"), ("kG", "k_GABA"))


def plot_calibration(traces, out_path):
    """All seeds' baseline (pre-drug) settling runs overlaid -- the sanity
    check that each network reaches (and holds) the ~2 Hz HSP fixed point
    before any drug condition is applied on top of it. One row per signal
    (E rate / I rate / k_AMPA / k_GABA) so seeds are the only thing distinguished
    by color -- no solid-vs-dashed to squint at."""
    if not traces:
        return
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for row, (key, ylabel) in enumerate(_ROWS):
        ax = axes[row]
        for j, tr in enumerate(traces):
            seed = int(tr["seed"])
            color = colors[j % len(colors)]
            ax.plot(tr["t"], tr[key], color=color, lw=0.9,
                   label=f"seed {seed}" if row == 0 else None)
        if key == "rE":
            ax.axhline(2.0, color="0.4", lw=0.9, ls=":", label="target (2 Hz)")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Baseline (pre-drug) settling, all seeds")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_condition(traces_for_condition, condition_name, out_path):
    """One row per signal (E rate / I rate / k_AMPA / k_GABA), one column per
    frequency (burst_freq_hz, then the shared reference frequency) -- seeds
    distinguished by color only, no solid-vs-dashed pairing to tell apart."""
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex="col")

    for col, freq_tag in enumerate(("burst", "steady")):
        freq_hz = float(traces_for_condition[0][f"{freq_tag}_freq_hz"])
        for row, (key, ylabel) in enumerate(_ROWS):
            ax = axes[row, col]
            for j, tr in enumerate(traces_for_condition):
                seed = int(tr["seed"])
                color = colors[j % len(colors)]
                ax.plot(tr[f"t_{freq_tag}"], tr[f"{key}_{freq_tag}"],
                       color=color, lw=1.0,
                       label=f"seed {seed}" if (row == 0 and col == 0) else None)
            if row == 0:
                ax.set_title(f"{freq_hz:g} Hz")
            ax.set_ylabel(ylabel)

    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"Cycle traces, all seeds -- {condition_name}")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_all(out_dir):
    out_dir = Path(out_dir)
    written = []

    calib_traces = load_calibration_traces(out_dir / "traces")
    if calib_traces:
        out_path = out_dir / "plots" / "calibration_traces.png"
        plot_calibration(calib_traces, out_path)
        written.append(out_path)

    traces = load_traces(out_dir / "traces")
    by_cond = group_by_condition(traces)
    for cond, trs in by_cond.items():
        out_path = out_dir / "plots" / f"{_slug(cond)}_traces.png"
        plot_condition(trs, cond, out_path)
        written.append(out_path)
    return written


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    paths = plot_all(results_dir)
    if not paths:
        print(f"no traces found under {Path(results_dir) / 'traces'}")
    for p in paths:
        print(f"wrote {p}")
