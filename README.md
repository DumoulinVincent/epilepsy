# Homeostatic plasticity and cyclic seizure risk

Spiking-network model (Brian2) in which **homeostatic synaptic plasticity (HSP)** turns a loss of input
(deafferentation) into recurring **high- and low-risk periods** for seizures. The repository holds everything
behind the paper in preparation and the AES 2026 poster (`poster/poster_aes_final.pdf`):

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
| `spiking_network/results/` | simulation outputs; only small tables are in git (see `results/README.md`) | |
| `epileptor/` | HSP → Epileptor `x0` mapping: notebook, probe script, exported trajectories, theory report | `report/report.pdf` |
| `paper/` | builds the current paper draft from the results | `build_paper.py`, `output/paper.pdf` |
| `poster/` | AES 2026 poster (source, figure script, presented PDF) | `poster_aes_final.pdf` |
| `docs/` | background textbook on dynamical systems and bifurcations (phase planes, folds, Hopf, slow–fast systems) | `dynamics_textbook/dynamics_textbook.pdf` |
| `exploration/` | five early notebooks kept for reference (see `exploration/README.md`) | |

Suggested reading order for a newcomer: the poster, `spiking_network/report/lfp_image_results.pdf` (short results
report), `paper/output/paper.pdf` (draft), then `docs/dynamics_textbook` and `epileptor/report/report.pdf` before
touching the Epileptor part.

## Setup

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
The first Brian2 run compiles the network with Cython (a few minutes); compiled code is cached in
`spiking_network/results/cython_cache/` and shared between parallel jobs.

## Data

Raw simulation outputs are **not in git** (about 8 GB). They are kept at: **[shared drive location — to fill in]**.
Copy them into `spiking_network/results/` keeping the folder names (`baselines/`, `lfp_tau250_15k/raw/`, ...),
or regenerate them with the commands below. What is in git is enough to rebuild the poster; the paper figures and
the Epileptor notebook also need the raw LFP and image runs.

## Pipeline

All simulation commands run from `spiking_network/`. Runtimes are for a 16-core machine; LFP jobs need about
1.2 GB of RAM each, so do not run the LFP and image batches at the same time.

| step | command | output | time |
|---|---|---|---|
| 0. baselines | `python make_baselines.py` | `results/baselines/seed{1..5}.bri`: each seed settled at 40 Hz, 15000 s, all HSP τ = 50 s | ~1 h |
| 1a. LFP runs | `python -u run_lfp_cycle.py --conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 200pA,DBS 250pA,DBS 300pA" --hsp-tau 250 --run-s 15000 --n-jobs 10 --skip-analysis --out results/lfp_tau250_15k` | `results/lfp_tau250_15k/raw/` (10 conditions × 5 seeds) | ~5 h |
| 1b. long LFP runs | same with `--conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA" --run-s 20000 --out results/lfp_tau250_20k` (benzo and DBS 20 pA are slow to seize, so the paper takes them from here) | `results/lfp_tau250_20k/raw/` | ~2 h |
| 1c. LFP analyses | `python lfp_treatment.py results/lfp_tau250_15k` (and `.../lfp_tau250_20k`); `python lfp_ews.py`; `python lfp_entropy.py` | power < 30 Hz, alarm threshold, early-warning trends, entropy | minutes |
| 2. image task | `python -u run_image_cycle.py --contrast 16 --out results/image_random_c16 --conditions "reference 40Hz,untreated,benzo (k=1.0),DBS 20pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 200pA,DBS 250pA,DBS 300pA"` | `results/image_random_c16/{raw,analysis}/` | ~12 h (analysis included) |
| 3. slow-HSP cycles (poster) | `python -u long_tau_cycle.py --tau 250 --tau-int-ampa 5000 --tau-ins-gaba 5000 --tau-hsp 1000 --run-s 20000 --seeds 1,2,3,4,5 --conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA,DBS 40pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 300pA"`; fast-HSP comparison: `--tau 250 --run-s 20000 --seeds 1` | `results/long_tau/raw/` (in git) | ~3 h |
| 4. Epileptor mapping | run `epileptor/HSP_Epileptor_trajectory.ipynb` top to bottom (from `epileptor/`) | `epileptor/results/trajectories.npz`, `probes.json` | ~15 min with cached probes |
| 5. outputs | `python paper/build_paper.py`; `python poster/poster_figures.py && pdflatex poster_aes.tex` (in `poster/`); `python spiking_network/report/build_lfp_image_report.py`; in `epileptor/report/`: `python theory_figures.py`, `python phase_plane_figures.py`, `python single_gain.py`, `pdflatex report.tex` | PDFs | minutes |

Quick checks: `python test_image_task.py` (image stimulus and decoders, synthetic data);
`python -u run_image_cycle.py --smoke` (one seed, 600 s); `python -u long_tau_cycle.py --tau 250 --run-s 30 --out /tmp/smoke`.

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
