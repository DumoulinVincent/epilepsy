# model/

The original network and HSP code, moved here unchanged from the project root. Do not edit these files: the
simulations in `../spiking_network/` import them (`spiking_network/worker.py` puts this folder on `sys.path`).

- `HSP_Phase.py`: drug application (`_apply_drugs`: benzodiazepine, DBS), HSP freezing (`_freeze_hsp`), and the
  Brian2 run namespace (`_runkw`). Its phase-analysis functions belong to earlier work and are not used in the paper.
- `hsp_capacity.py`, `hsp_metrics.py`: imported by `HSP_Phase.py` (they also provide the model constants used by
  `_runkw`). Their capacity/information measures are not used in the paper.

The network itself (seeded construction, connectivity, HSP equations) is in `../spiking_network/seeded_network.py`.
