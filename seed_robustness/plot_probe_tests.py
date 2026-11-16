"""Summary of the run_probe_tests.sh runaway / IPC test (seed 3, probe around
each network's own drive): encoding IPC, memory IPC and mean E rate per probe,
one column of dots per test. Probes that ran away (E > 10 Hz) are red crosses
and are left out of the means.

    python plot_probe_tests.py
-> results/ipc_probe_inspection/probe_tests_summary.png
"""
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import probe_plots  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = THIS_DIR / "results" / "ipc_probe_inspection"
TESTS = [  # (npz stem, x label)
    ("seed3_reference_40Hz_probe40pm4", "40 Hz reference\n40 ± 4 Hz"),
    ("seed3_untreated_probe36pm4", "untreated\n36 ± 4 Hz"),
    ("seed3_untreated_probe36pm2", "untreated\n36 ± 2 Hz"),
    ("seed3_DBS_40pA_probe36pm4", "DBS 40 pA\n36 ± 4 Hz"),
]
STABLE = "#2a78d6"
RUNAWAY = "#d03b3b"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3de"


def main():
    data = []
    for stem, label in TESTS:
        m = probe_plots.load_npz(OUT / f"{stem}.npz")["meta"]
        E = np.array(m["E_rate"])
        data.append(dict(label=label, seeds=m["probe_seeds"], E=E, ran=E > probe_plots.RUNAWAY_HZ,
                         enc=np.array(m["ipc_encoding"]), mem=np.array(m["ipc_memory"])))

    panels = [("enc", "IPC encoding (d = 0)", False),
              ("mem", "IPC memory (d ≥ 1)", False),
              ("E", "mean E rate during probe (Hz)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    xs = np.arange(len(data))
    offs = np.linspace(-0.18, 0.18, 4)
    for ax, (key, ylabel, log) in zip(axes, panels):
        for x, d in zip(xs, data):
            v = d[key]
            ok = ~d["ran"]
            ax.scatter(x + offs[ok], v[ok], s=60, color=STABLE, edgecolor="white", lw=1.5, zorder=3)
            ax.scatter(x + offs[~ok], v[~ok], s=70, marker="X", color=RUNAWAY, edgecolor="white",
                       lw=1, zorder=3)
            if ok.any():
                ax.hlines(v[ok].mean(), x - 0.3, x + 0.3, color=INK, lw=2, zorder=2)
        if log:
            ax.set_yscale("log")
            ax.axhline(probe_plots.RUNAWAY_HZ, color=MUTED, ls="--", lw=1)
            ax.text(xs[-1] + 0.45, probe_plots.RUNAWAY_HZ * 1.15, "runaway threshold 10 Hz",
                    ha="right", va="bottom", fontsize=8, color=MUTED)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{d['label']}\n{int(d['ran'].sum())}/{len(d['ran'])} ran away"
                            for d in data], fontsize=9, color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_xlim(-0.6, len(data) - 0.4)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    axes[0].scatter([], [], s=60, color=STABLE, label="probe, stable")
    axes[0].scatter([], [], s=70, marker="X", color=RUNAWAY, label="probe, ran away (E > 10 Hz)")
    axes[0].plot([], [], color=INK, lw=2, label="mean of stable probes")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9,
               frameon=False)
    fig.suptitle("IPC probe test, seed 3: each network probed around its own drive, HSP frozen, "
                 "4 probes × 100 s", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = OUT / "probe_tests_summary.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
