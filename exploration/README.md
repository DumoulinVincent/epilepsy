# exploration/

Five early notebooks, kept unchanged for reference. Each defines its own copy of the network inline.
`DBS_intervention.ipynb` also imports `HSP_Phase`, `hsp_capacity` and `hsp_metrics` (add `../model` to `sys.path`)
and restores the shared `baseline_2hz.bri` (regenerate it with the notebook's own calibration cell, or use a
`spiking_network/results/baselines/seed<s>.bri`). The original layout is in the tag `archive-2026-11`.

| notebook | content |
|---|---|
| `Phase_plane.ipynb` | phase-plane analysis of the HSP gains (`k_AMPA`, `k_GABA`) and the population rate |
| `2D_PP_simple.ipynb` | small standalone Brian2 model for the HSP phase plane |
| `sanity_check_HSP.ipynb` | single-neuron check of the HSP rate estimate and gain dynamics |
| `benzo_simu.ipynb` | first benzodiazepine simulations (GABA gain and I→E weight increased) |
| `DBS_intervention.ipynb` | first DBS simulations (Poisson current source on E cells; capacity vs seizure trade-off); `seeded_network.run_sim_seeded` reproduces its `run_sim()` |
