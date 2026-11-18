# Homeostatic plasticity and cyclic seizure risk

Spiking-network model (Brian2) in which **homeostatic synaptic plasticity (HSP)** turns a loss of input
(deafferentation) into recurring **high- and low-risk periods** for seizures. The repository holds everything
behind a paper in preparation and an AES 2026 poster:

1. **The cycle**: 100 excitatory + 25 inhibitory neurons; HSP scales each excitatory cell's AMPA (`k_E`) and GABA
   (`k_I`) gains to keep its rate near 2 Hz. A 10 % drop of input (40 → 36 Hz) makes HSP drift until the network
   enters a high-rate (high-risk) state, then reverse: a slow cycle.
2. **Treatments**: benzodiazepine (inhibition held at its maximum, `k_I = 1`) and DBS (120 Hz current pulses into
   every excitatory cell, 20–300 pA).
3. **Image recovery**: how well a weak binary image carried by the input can be read out of the spikes, with an
   upper bound (decoder re-trained on each condition) and a lower bound (decoder trained on the healthy network).
4. **LFP**: power below 30 Hz tracks the hidden HSP state and flags high-risk periods.
5. **Outlook**: the HSP state mapped onto the excitability `x0` of the Epileptor (the seizure model of The Virtual
   Brain), as an opening towards whole-brain models.

Information-processing capacity (IPC) and participation-ratio analyses from earlier stages are **not** part of the
paper and are not in this tree; they are in the git tag `archive-2026-11` (see "History").

## Repository map

| folder | what it is | start with |
|---|---|---|
| `model/` | the original network and HSP code (`HSP_Phase.py`, `hsp_capacity.py`, `hsp_metrics.py`). Kept unchanged; other code imports it. | `HSP_Phase.py` (`_apply_drugs`, `_runkw`) |
| `spiking_network/` | the seeded network, the conditions, every simulation and analysis used in the paper | `seeded_network.py`, `conditions.py` |
| `figures/` | every paper figure from the results: `make_paper_figures.py` writes them to `figures/out/` | `make_paper_figures.py` |
| `spiking_network/results/` | simulation outputs; only small tables are in git (see `results/README.md`) | |
| `epileptor/` | HSP → Epileptor `x0` mapping: notebook, probe script, exported trajectories, theory report | `report/report.pdf` |
| `docs/` | background textbook on dynamical systems and bifurcations (phase planes, folds, Hopf, slow–fast systems) | `dynamics_textbook/dynamics_textbook.pdf` |
| `exploration/` | five early notebooks kept for reference (see `exploration/README.md`) | |

Suggested reading order for a newcomer: `spiking_network/report/lfp_image_results.pdf` (short results report),
then `docs/dynamics_textbook` and `epileptor/report/report.pdf` before
touching the Epileptor part.

## Setup and running everything

On Linux or macOS (Intel or Apple Silicon), from the repository root:

```bash
./setup.sh            # once: checks Python 3.12 and a C compiler, creates ./venv with every package, tests Brian2
./reproduce_all.sh    # every simulation, analysis and figure, in order
```

`setup.sh` tells you what to install if something is missing (macOS: `brew install python@3.12`,
`xcode-select --install`; Ubuntu/Debian: `sudo apt install python3.12 python3.12-venv python3.12-dev
build-essential`). A LaTeX distribution (`pdflatex`) is optional: without it the PDF reports are skipped, the figures
are still made.

`reproduce_all.sh` uses the venv automatically, detects the machine's cores and memory, and runs that many jobs in
parallel (all cores but one, limited by memory: ~1.3 GB per LFP/image job, ~0.6 GB otherwise); override with
`N_JOBS=8 ./reproduce_all.sh` (and `N_HEAVY` for the LFP/image jobs). Stages can be run alone:
`./reproduce_all.sh frozen clamp`. Every stage skips runs that already exist, and the long seizure-clamp runs write a
full checkpoint every 10 000 s of model time, so **after an interruption (reboot, update), run the same command again**:
finished runs are skipped and unfinished ones continue from their last checkpoint.

**Runtime: about 2.5 days on a 16-core machine** (baselines ~1 h, LFP ~7 h, image task ~12 h, slow-HSP runs ~4 h,
input sweep and 38 Hz runs ~2 h, frozen-gain analysis ~5 h, seizure clamp ~35 h, of which ~25 h for the runs from
deafferentation). It scales roughly with the number of cores.

The first Brian2 run compiles the network with Cython (a few minutes); compiled code is cached in
`spiking_network/results/cython_cache/` and shared between parallel jobs.

## Data

There is no experimental data: every result comes from simulations of this model, and all of them can be
regenerated from the code here. Raw outputs are therefore **not in git**; only small tables and the slow-HSP runs
are (see `spiking_network/results/README.md`). Every run is seeded (network weights and Poisson input), so
regenerating should give the same series with the same package versions (`requirements.txt`); other hardware or
versions may introduce small numerical differences, which change individual noise realisations but not the
statistics.

## Pipeline

`reproduce_all.sh` runs these stages; to run one by hand, start from `spiking_network/` (replace `--n-jobs`). Times
are for a 16-core machine.

| stage | command(s) | output | time |
|---|---|---|---|
| baselines | `python make_baselines.py` | `results/baselines/seed{1..5}.bri`: each network settled at 40 Hz, 15000 s, all HSP τ = 50 s | ~1 h |
| lfp | `python -u run_lfp_cycle.py --conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 200pA,DBS 250pA,DBS 300pA" --hsp-tau 250 --run-s 15000 --skip-analysis --out results/lfp_tau250_15k`; the same with 4 conditions, `--run-s 20000 --out results/lfp_tau250_20k`; then `lfp_treatment.py`, `lfp_ews.py`, `lfp_entropy.py` | `results/lfp_tau250_{15k,20k}/` | ~7 h |
| image | `python -u run_image_cycle.py --contrast 16 --out results/image_random_c16 --conditions "..."` (10 conditions) | `results/image_random_c16/{raw,analysis}/` | ~12 h |
| slow | `python -u long_tau_cycle.py --tau 250 --tau-int-ampa 5000 --tau-ins-gaba 5000 --tau-hsp 1000 --run-s 20000 --seeds 1,2,3,4,5 --conditions "..."`, plus two comparison runs | `results/long_tau/raw/` (in git) | ~4 h |
| sweep | `python -u long_tau_cycle.py --tau 50 --run-s 5000 --seeds 1 --conditions "untreated 32Hz,...,untreated 42Hz" --out results/input_sweep` | input-rate sweep (Fig. 2) | ~0.5 h |
| long38 | `python -u long_tau_cycle.py --tau 50 --run-s 30000 --seeds 1,2,3,4,5 --conditions "untreated 38Hz"` | `results/long_tau/raw/` | ~1.5 h |
| frozen | `python -u frozen_gain.py pilot`, `full`, `checks`, `cycle_gains` | gains frozen: persistence of the high-rate state, regime map and hysteresis sweeps, escape risk, 0.5 ms check, per-cell gain snapshots and their risk (`results/frozen_gain/`) | ~5 h |
| clamp | `python -u seizure_clamp.py first`, `inzone`, `deaff` | seizures ended after D s with HSP running: from per-cell gains below and inside the risk zone, and from deafferentation with checkpoints (`results/seizure_clamp/`) | ~35 h |
| epileptor | run `epileptor/HSP_Epileptor_trajectory.ipynb` top to bottom | `epileptor/results/trajectories.npz`, `probes.json` | ~15 min |
| outputs | `python figures/make_paper_figures.py`; `python spiking_network/event_model_figures.py`; the results report and the Epileptor report | every paper figure in `figures/out/`, PDFs | minutes |

`figures/make_paper_figures.py` writes every figure of the paper under the file names the manuscript uses
(`figures/overview_figures.py`, `figures/spiking_figures.py`, and the figure scripts of each analysis in
`spiking_network/`). `spiking_network/event_model.py` is a reduced model calibrated on the frozen-gain results, used to
choose the settings of the long clamp runs (`python event_model.py validate` compares it with the spiking runs).

Quick checks: `python test_image_task.py` (image stimulus and decoders, synthetic data);
`python -u run_image_cycle.py --smoke` (one seed, 600 s); `python -u long_tau_cycle.py --tau 250 --run-s 30 --out /tmp/smoke`
(needs the baselines).

## Conventions used everywhere

- **Conditions** (`spiking_network/conditions.py`): healthy reference = 40 Hz input; untreated (deafferented) =
  36 Hz; benzodiazepine = `k_GABA` clamped at 1; DBS *A* pA = 120 Hz Poisson current pulses of amplitude *A* onto
  every E cell. Five network seeds (1–5) draw independent weights and noise.
- **HSP time constants** are set per experiment after restoring the baseline: τ = 250 s for the LFP runs, 25 s for
  the image runs; the slow-HSP runs set the four constants and the rate-sensor time constant separately. In the
  brain these would be days; in the model they are compressed so that many cycles fit in a run.
- **High-rate (high-risk) state**: E population rate above 10 Hz in 1 s bins. The first 500 s of every run are
  discarded (settling after the switch from the 40 Hz baseline).
- **LFP proxy**: |mean recurrent AMPA current| + |mean recurrent GABA current| onto E cells, sampled every 1 ms.
- Every job runs in its own process with its own working directory and Cython cache (`worker._isolate`), and
  restores `results/baselines/seed<s>.bri`.

## History

The repository was reorganised in November 2026. Everything from before (the early receptive-field, alcohol,
two-layer and HH notebooks, the IPC/participation-ratio pipelines `seed_robustness/` and `seed_robustnessv3/`, and
their results) is preserved in the tag **`archive-2026-11`**:

```bash
git checkout archive-2026-11        # browse the old tree (read-only); `git checkout main` to come back
```
