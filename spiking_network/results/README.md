# spiking_network/results/

Simulation outputs. Raw runs are not in git (see the top-level README, "Data"); the small tables that the figures
read are.

| folder | content | in git |
|---|---|---|
| `baselines/` | `seed{1..5}.bri`: calibrated 40 Hz network state per seed (`make_baselines.py`) | no (~165 MB each) |
| `cython_cache/` | compiled Brian2 code (`_template/` + one copy per job) | no (machine-specific) |
| `lfp_tau250_15k/raw/` | LFP runs, 10 conditions × 5 seeds, 15000 s, HSP τ = 250 s: `lfp_exc`, `lfp_inh` (pA, 1 ms), E/I rates and HSP gains (1 s) | no (~5 GB) |
| `lfp_tau250_15k/treatment/` | `p30_windows.csv` (power < 30 Hz per 50 s window), `threshold.csv` (alarm threshold and its test) | yes |
| `lfp_tau250_15k/ews/` | early-warning trends in the last 10 min before onset (`ews_windows.csv`, `ews_kendall.csv`) | yes |
| `lfp_tau250_15k/entropy/` | spectral / permutation entropy per window and their trends | yes |
| `lfp_tau250_20k/` | benzodiazepine, DBS 20 pA (+ untreated and reference) for 20000 s; `treatment/` as above | raw no, tables yes |
| `image_random_c16/` | image task, contrast 16 Hz, 10 conditions × 5 seeds, 8000 s, HSP τ = 25 s: `raw/` runs, `analysis/summary.csv` (balanced accuracy per condition, seed, readout window, decoder) | summary only |
| `image_c16_dbs300/` | the same at DBS 300 pA, seed 1 | summary only |
| `long_tau/raw/` | slow-HSP runs for the poster (`long_tau_cycle.py`): 1 s rates and 10 s gains, 20000 s | yes |
