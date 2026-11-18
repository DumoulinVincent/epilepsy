#!/bin/bash
# One-time setup: checks Python 3.12 and a C compiler, creates ./venv with every package, and tests that Brian2 can
# compile. Works on Linux and macOS (Intel or Apple Silicon).
#
#   ./setup.sh                         # then: ./reproduce_all.sh
#   PYTHON=/path/to/python3.12 ./setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
OS="$(uname)"

# Ubuntu/Debian (also WSL on Windows): install the system packages if something is missing (asks for your password)
if [ "$OS" = Linux ] && command -v apt-get > /dev/null; then
    need=0
    command -v python3.12 > /dev/null || need=1
    command -v gcc > /dev/null || need=1
    command -v git > /dev/null || need=1
    command -v python3.12 > /dev/null && ! python3.12 -c "import ensurepip" 2> /dev/null && need=1
    if [ $need = 1 ]; then
        echo "Installing system packages: python3.12, venv, headers, compiler, git (sudo)..."
        sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential git
    fi
fi

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for c in python3.12 python3; do
        if command -v "$c" > /dev/null && "$c" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then PY="$c"; break; fi
    done
fi
if [ -z "$PY" ]; then
    echo "Python 3.12 not found."
    [ "$OS" = Darwin ] && echo "  macOS: brew install python@3.12   (then rerun ./setup.sh)" \
                       || echo "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi
echo "Python: $($PY --version) ($PY)"

if ! command -v cc > /dev/null && ! command -v gcc > /dev/null && ! command -v clang > /dev/null; then
    echo "No C compiler found (Brian2 compiles the network)."
    [ "$OS" = Darwin ] && echo "  macOS: xcode-select --install" || echo "  Ubuntu/Debian: sudo apt install build-essential"
    exit 1
fi

[ -d venv ] || "$PY" -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip > /dev/null
python -m pip install -r requirements.txt

echo "Testing Brian2 + Cython compilation..."
python - <<'PY'
from brian2 import NeuronGroup, run, ms, prefs
prefs.codegen.target = "cython"
G = NeuronGroup(10, "dv/dt = -v / (10*ms) : 1")
run(10 * ms)
import numba, sklearn, scipy, pandas, joblib  # noqa: F401
print("Brian2/Cython OK; all packages import.")
PY

command -v pdflatex > /dev/null || echo "Note: pdflatex not found; the PDF reports will be skipped (figures still made)."
command -v jupyter > /dev/null || true
echo
echo "Setup done. Next:  ./reproduce_all.sh      (see README for stages and runtimes)"
