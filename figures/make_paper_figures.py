"""Every figure of the paper, from the simulation results (run reproduce_all.sh first, or the stages it needs).

    python figures/make_paper_figures.py          # -> figures/out/ (file names as used by the manuscript)

  network, cycle_slow_fast, treatments_slow, epileptor    overview_figures.py
  fig1_cycle ... fig7_examples                            spiking_figures.py
  input_sweep                                             spiking_network/input_sweep_figure.py
  mechanism, cycle_gains                                  spiking_network/frozen_gain_figures.py
  clamp, clamp_deaff                                      spiking_network/seizure_clamp_figures.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SN = HERE.parent / "spiking_network"
OUT = HERE / "out"


def run(script, cwd, *args):
    print(f"--- {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, script, *args], cwd=cwd, check=True)


def main():
    OUT.mkdir(exist_ok=True)
    run("overview_figures.py", HERE, "network", "story", "conditions", "epi")
    run("spiking_figures.py", HERE)
    run("input_sweep_figure.py", SN)
    run("frozen_gain_figures.py", SN)
    run("seizure_clamp_figures.py", SN)
    copies = {SN / "results/input_sweep/input_sweep.png": "input_sweep.png",
              SN / "results/frozen_gain/figures/frozen_gain.pdf": "mechanism.pdf",
              SN / "results/frozen_gain/figures/cycle_gains.pdf": "cycle_gains.pdf",
              SN / "results/seizure_clamp/figures/clamp_inzone.png": "clamp.png",
              SN / "results/seizure_clamp/figures/clamp_deaff.png": "clamp_deaff.png"}
    for src, name in copies.items():
        if src.exists():
            shutil.copy(src, OUT / name)
        else:
            print("missing (stage not run?):", src)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
