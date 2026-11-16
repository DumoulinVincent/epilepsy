#!/bin/bash
# Regenerate every simulation behind the results, then the reports.
#
#   ./reproduce_all.sh                    # all stages, in order (about 1 day on 16 cores)
#   ./reproduce_all.sh baselines lfp      # only some stages
#
# Stages (each skips runs whose output already exists, so an interrupted run can simply be restarted):
#   baselines   5 calibrated network states, 40 Hz, 15000 s each                       ~1 h
#   lfp         LFP runs: 10 conditions x 5 seeds x 15000 s, + 4 conditions x 20000 s   ~7 h
#               and their analyses (power < 30 Hz, alarm threshold, early warning, entropy)
#   image       image-recovery runs: 10 conditions x 5 seeds x 8000 s, + analysis       ~12 h
#   slow        slow-HSP cycles (already in git; rerun to check)                         ~4 h
#   epileptor   HSP -> Epileptor x0 mapping notebook (needs lfp)                        ~15 min
#   outputs     short results report and Epileptor report                                minutes
#
# Memory: an LFP job needs ~1.2 GB; set N_JOBS (default 10) to fit your machine. Do not run two stages at once.
# Outputs: spiking_network/results/, epileptor/results/, spiking_network/report/, epileptor/report/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SN="$ROOT/spiking_network"
N_JOBS="${N_JOBS:-10}"
export OPENBLAS_NUM_THREADS=1
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(baselines lfp image slow epileptor outputs)
say() { echo; echo "==== $(date '+%F %T')  $*"; }

stage_baselines() {
    cd "$SN" && python make_baselines.py --n-jobs 5
}

stage_lfp() {
    cd "$SN"
    python -u run_lfp_cycle.py --hsp-tau 250 --run-s 15000 --n-jobs "$N_JOBS" --skip-analysis \
        --conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 200pA,DBS 250pA,DBS 300pA" \
        --out results/lfp_tau250_15k
    python -u run_lfp_cycle.py --hsp-tau 250 --run-s 20000 --n-jobs "$N_JOBS" --skip-analysis \
        --conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA" --out results/lfp_tau250_20k
    python lfp_treatment.py results/lfp_tau250_15k
    python lfp_treatment.py results/lfp_tau250_20k
    python lfp_ews.py results/lfp_tau250_15k
    python lfp_entropy.py
}

stage_image() {
    cd "$SN"
    python -u run_image_cycle.py --contrast 16 --out results/image_random_c16 --n-jobs "$N_JOBS" \
        --conditions "reference 40Hz,untreated,benzo (k=1.0),DBS 20pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 200pA,DBS 250pA,DBS 300pA"
}

stage_slow() {
    cd "$SN"
    python -u long_tau_cycle.py --tau 250 --tau-int-ampa 5000 --tau-ins-gaba 5000 --tau-hsp 1000 --run-s 20000 \
        --seeds 1,2,3,4,5 --n-jobs "$N_JOBS" \
        --conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA,DBS 40pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 300pA"
    python -u long_tau_cycle.py --tau 250 --run-s 20000 --seeds 1                  # fast HSP, for comparison
    python -u long_tau_cycle.py --tau 250 --tau-hsp 1000 --run-s 20000 --seeds 1   # slower rate sensor only
}

stage_epileptor() {
    cd "$ROOT/epileptor"
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 HSP_Epileptor_trajectory.ipynb
}

stage_outputs() {
    cd "$SN/report" && python build_lfp_image_report.py
    cd "$ROOT/epileptor/report" && python theory_figures.py && python phase_plane_figures.py && python single_gain.py \
        && pdflatex -interaction=nonstopmode report.tex > /dev/null && pdflatex -interaction=nonstopmode report.tex > /dev/null
}

for s in "${STAGES[@]}"; do
    declare -F "stage_$s" > /dev/null || { echo "unknown stage: $s"; exit 1; }
    say "stage $s"
    "stage_$s"
done
say "done"
