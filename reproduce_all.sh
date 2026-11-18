#!/bin/bash
# Regenerate every simulation behind the results, then the figures and reports. Run ./setup.sh once first.
#
#   ./reproduce_all.sh                    # all stages, in order
#   ./reproduce_all.sh frozen clamp       # only some stages
#   N_JOBS=8 ./reproduce_all.sh           # override the number of parallel jobs
#
# Stages, with runtimes on a 16-core machine (each skips runs whose output already exists, and the long seizure-clamp
# runs keep checkpoints, so an interrupted run can simply be restarted with the same command):
#   baselines   5 calibrated network states, 40 Hz, 15000 s each                              ~1 h
#   lfp         LFP runs: 10 conditions x 5 seeds x 15000 s, + 4 conditions x 20000 s          ~7 h
#               and their analyses (power < 30 Hz, alarm threshold, early warning, entropy)
#   image       image-recovery runs: 10 conditions x 5 seeds x 8000 s, + analysis              ~12 h
#   slow        slow-HSP cycles (already in git; rerun to check)                                ~4 h
#   sweep       input-rate sweep, 32-42 Hz, HSP tau 50 s, 5000 s each (Fig. 2)                  ~0.5 h
#   long38      38 Hz with HSP running, tau 50 s, 5 seeds x 30000 s                              ~1.5 h
#   frozen      frozen-gain analysis: pilot, regime map + risk (full), checks, per-cell gains  ~5 h
#   clamp       seizure clamp: below / inside the risk zone, and from deafferentation          ~35 h
#   epileptor   HSP -> Epileptor x0 mapping notebook (needs lfp)                               ~15 min
#   outputs     every paper figure (figures/out/), results report and Epileptor report         minutes
#
# Parallel jobs: by default, all cores but one, limited by memory (LFP and image jobs ~1.3 GB each, others ~0.6 GB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SN="$ROOT/spiking_network"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"
if [ "$(uname)" = Darwin ]; then
    CORES=$(sysctl -n hw.ncpu); MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
else
    CORES=$(nproc); MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
fi
by_mem() { local n=$(( MEM_GB * 10 / $1 )); local c=$(( CORES > 1 ? CORES - 1 : 1 )); n=$(( n < c ? n : c )); echo $(( n > 0 ? n : 1 )); }
N_JOBS="${N_JOBS:-$(by_mem 6)}"            # light jobs, ~0.6 GB
N_HEAVY="${N_HEAVY:-$(by_mem 14)}"         # LFP / image jobs, ~1.3 GB (+ margin)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per job (after detection)
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(baselines lfp image slow sweep long38 frozen clamp epileptor outputs)
say() { echo; echo "==== $(date '+%F %T')  $*"; }
say "machine: $CORES cores, $MEM_GB GB RAM -> $N_JOBS parallel jobs ($N_HEAVY for LFP/image)"

stage_baselines() {
    cd "$SN" && python make_baselines.py --n-jobs "$(( N_HEAVY < 5 ? N_HEAVY : 5 ))"
}

stage_lfp() {
    cd "$SN"
    python -u run_lfp_cycle.py --hsp-tau 250 --run-s 15000 --n-jobs "$N_HEAVY" --skip-analysis \
        --conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 200pA,DBS 250pA,DBS 300pA" \
        --out results/lfp_tau250_15k
    python -u run_lfp_cycle.py --hsp-tau 250 --run-s 20000 --n-jobs "$N_HEAVY" --skip-analysis \
        --conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA" --out results/lfp_tau250_20k
    python lfp_treatment.py results/lfp_tau250_15k
    python lfp_treatment.py results/lfp_tau250_20k
    python lfp_ews.py results/lfp_tau250_15k
    python lfp_entropy.py
}

stage_image() {
    cd "$SN"
    python -u run_image_cycle.py --contrast 16 --out results/image_random_c16 --n-jobs "$N_HEAVY" \
        --conditions "reference 40Hz,untreated,benzo (k=1.0),DBS 20pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 200pA,DBS 250pA,DBS 300pA"
}

stage_slow() {
    cd "$SN"
    python -u long_tau_cycle.py --tau 250 --tau-int-ampa 5000 --tau-ins-gaba 5000 --tau-hsp 1000 --run-s 20000 \
        --seeds 1,2,3,4,5 --n-jobs "$N_JOBS" \
        --conditions "untreated,reference 40Hz,benzo (k=1.0),DBS 20pA,DBS 40pA,DBS 60pA,DBS 100pA,DBS 150pA,DBS 300pA"
    python -u long_tau_cycle.py --tau 250 --run-s 20000 --seeds 1 --n-jobs "$N_JOBS"                  # fast HSP
    python -u long_tau_cycle.py --tau 250 --tau-hsp 1000 --run-s 20000 --seeds 1 --n-jobs "$N_JOBS"   # slower sensor
}

stage_sweep() {
    cd "$SN"
    local c="" f
    for f in 32 32.5 33 33.5 34 34.5 35 35.5 36 36.5 37 37.5 38 38.5 39 39.5 40 40.5 41 41.5 42; do
        c+="${c:+,}untreated ${f}Hz"
    done
    python -u long_tau_cycle.py --tau 50 --run-s 5000 --seeds 1 --n-jobs "$N_JOBS" --conditions "$c" \
        --out results/input_sweep
}

stage_long38() {
    cd "$SN" && python -u long_tau_cycle.py --tau 50 --run-s 30000 --seeds 1,2,3,4,5 --n-jobs "$N_JOBS" \
        --conditions "untreated 38Hz"
}

stage_frozen() {
    cd "$SN"
    python -u frozen_gain.py pilot --n-jobs "$N_JOBS"
    python -u frozen_gain.py full --n-jobs "$N_JOBS"
    python -u frozen_gain.py checks --n-jobs "$N_JOBS"
    python -u frozen_gain.py cycle_gains --n-jobs "$N_JOBS"
}

stage_clamp() {
    cd "$SN"
    python -u seizure_clamp.py first --n-jobs "$N_JOBS"
    python -u seizure_clamp.py inzone --n-jobs "$N_JOBS"
    python -u seizure_clamp.py deaff --n-jobs "$N_JOBS"       # resumable: rerun to continue from checkpoints
}

stage_epileptor() {
    cd "$ROOT/epileptor"
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 HSP_Epileptor_trajectory.ipynb
}

stage_outputs() {
    cd "$ROOT" && python figures/make_paper_figures.py
    cd "$SN" && python event_model_figures.py
    cd "$SN/report" && python build_lfp_image_report.py
    cd "$ROOT/epileptor/report" && python theory_figures.py && python phase_plane_figures.py && python single_gain.py
    if command -v pdflatex > /dev/null; then
        pdflatex -interaction=nonstopmode report.tex > /dev/null && pdflatex -interaction=nonstopmode report.tex > /dev/null
    else
        echo "pdflatex not found: Epileptor report PDF skipped"
    fi
}

for s in "${STAGES[@]}"; do
    declare -F "stage_$s" > /dev/null || { echo "unknown stage: $s"; exit 1; }
    say "stage $s"
    "stage_$s"
done
say "done"
