import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_traces import _slug, group_by_condition, load_traces

WINDOW_PRE_S = 600.0    # how far before onset to show
WINDOW_POST_S = 100.0   # how far after onset to show (covers a typical
                        # ~150-200s burst + ~195s recovery_s, with margin)

# One row per signal: (trace-array prefix, y-axis label, fixed y-limits).
# E rate is clipped to 0-5Hz so the sub-burst ramp is visible (the burst
# peak itself, tens of Hz, is intentionally cut off); k_AMPA/k_GABA are
# fixed to their natural 0-1 range (bounded by construction in the HSP
# equations) so conditions are directly comparable to each other.
_ONSET_ROWS = (
    ("rE", "E rate (Hz)", (0.0, 5.0)),
    ("kA", "k_AMPA", (0.0, 1.0)),
    ("kG", "k_GABA", (0.0, 1.0)),
)


def plot_onset_alignment(traces_for_condition, condition_name, out_path,
                         freq_tag="burst", window_pre_s=WINDOW_PRE_S,
                         window_post_s=WINDOW_POST_S, rows=_ONSET_ROWS):
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 3.2 * len(rows)), sharex=True)

    n_onsets_total = 0
    for j, tr in enumerate(traces_for_condition):
        seed = int(tr["seed"])
        color = colors[j % len(colors)]
        onsets = np.atleast_1d(tr[f"burst_onsets_{freq_tag}"])
        t = tr[f"t_{freq_tag}"]
        for k, onset_t in enumerate(onsets):
            rel_t = t - onset_t
            mask = (rel_t >= -window_pre_s) & (rel_t <= window_post_s)
            if not mask.any():
                continue
            for row, (key, _, _) in enumerate(rows):
                axes[row].plot(rel_t[mask], tr[f"{key}_{freq_tag}"][mask],
                              color=color, lw=1.1, alpha=0.85,
                              label=f"seed {seed}" if (k == 0 and row == 0) else None)
            n_onsets_total += 1

    if n_onsets_total == 0:
        plt.close(fig)
        return None

    for row, (_, ylabel, ylim) in enumerate(rows):
        ax = axes[row]
        ax.axvline(0, ls=":", color="0.3", lw=1.2, label="onset" if row == 0 else None)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xlim(-window_pre_s, window_post_s)
    axes[-1].set_xlabel("time relative to seizure onset (s)")
    fig.suptitle(f"Onset-aligned approach to bifurcation -- {condition_name}\n"
                f"({n_onsets_total} onset(s) across {len(traces_for_condition)} seed(s))")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_all(out_dir):
    out_dir = Path(out_dir)
    traces = load_traces(out_dir / "traces")
    by_cond = group_by_condition(traces)
    written = []
    for cond, trs in by_cond.items():
        out_path = out_dir / "plots" / f"{_slug(cond)}_onset.png"
        p = plot_onset_alignment(trs, cond, out_path)
        if p:
            written.append(p)
    return written


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    paths = plot_all(results_dir)
    if not paths:
        print(f"no onsets found under {Path(results_dir) / 'traces'}")
    for p in paths:
        print(f"wrote {p}")
