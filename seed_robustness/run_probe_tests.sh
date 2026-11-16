#!/usr/bin/env bash
# Runaway / IPC probe test on seed 3, probing each network around its own
# drive: 40 Hz reference (40 +- 4), untreated (36 +- 4 and 36 +- 2), DBS 40 pA
# (36 +- 4). All four in parallel, one core each (inspect_ipc_probe.py uses a
# single BLAS thread), cycle characterization skipped. Logs and figures go to
# results/ipc_probe_inspection/; a summary is printed when all are done.
#
#   bash run_probe_tests.sh
set -u
cd "$(dirname "$0")"
source /home/vincent/bmnn/bin/activate
out=results/ipc_probe_inspection
mkdir -p "$out"

run() {  # name, then inspect_ipc_probe.py args
    local name=$1; shift
    python -u inspect_ipc_probe.py --seed 3 --skip-cycle "$@" > "$out/test_$name.log" 2>&1 &
    echo "started $name (log: $out/test_$name.log)"
}
run reference_pm4  --condition reference  --amp 4
run untreated_pm4  --condition untreated  --amp 4
run untreated_pm2  --condition untreated  --amp 2
run dbs40_pm4      --condition "DBS 40pA" --amp 4

echo "progress: tail -f $out/test_*.log   (Ctrl+C stops tail, not the runs)"
wait
echo
for f in "$out"/test_reference_pm4.log "$out"/test_untreated_pm4.log \
         "$out"/test_untreated_pm2.log "$out"/test_dbs40_pm4.log; do
    echo "== $(basename "$f" .log)"
    grep -E "snapshot at|no snapshot|probe [0-9]+:|ran away|runaway|Traceback|Error" "$f"
done
