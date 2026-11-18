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

# System tools: a C compiler (Brian2 compiles the network), git and curl. On Ubuntu/Debian (also WSL on Windows) they
# are installed if missing (asks for your password); Python itself is NOT taken from apt (see below).
if [ "$OS" = Linux ] && command -v apt-get > /dev/null; then
    if ! command -v gcc > /dev/null || ! command -v git > /dev/null || ! command -v curl > /dev/null; then
        echo "Installing system packages: compiler, git, curl (sudo)..."
        sudo apt-get update && sudo apt-get install -y build-essential git curl
    fi
fi
if ! command -v cc > /dev/null && ! command -v gcc > /dev/null && ! command -v clang > /dev/null; then
    echo "No C compiler found (Brian2 compiles the network)."
    [ "$OS" = Darwin ] && echo "  macOS: xcode-select --install" || echo "  install gcc (e.g. build-essential)"
    exit 1
fi

# Python 3.12 (the pinned packages need it). Use the system's python3.12 if it has venv support; otherwise (e.g.
# Ubuntu 26.04, which only ships newer Pythons) uv downloads a standalone Python 3.12 into the user's home, no root.
PY="${PYTHON:-}"
if [ -z "$PY" ] && [ "${USE_UV:-0}" != 1 ]; then
    for c in python3.12 python3; do
        if command -v "$c" > /dev/null && "$c" -c 'import sys, ensurepip; sys.exit(sys.version_info[:2] != (3, 12))' 2> /dev/null; then
            PY="$c"; break
        fi
    done
fi

if [ -n "$PY" ]; then
    echo "Python: $($PY --version) ($PY)"
    [ -d venv ] || "$PY" -m venv venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    python -m pip install --upgrade pip > /dev/null
    python -m pip install -r requirements.txt
else
    if ! command -v uv > /dev/null; then
        echo "No Python 3.12 on this system: installing uv to fetch one (https://astral.sh/uv)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${UV_INSTALL_DIR:-$HOME/.local/bin}:$PATH"
    fi
    [ -d venv ] || uv venv --python 3.12 --python-preference only-managed venv
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "Python: $(python --version) (uv-managed)"
    uv pip install -r requirements.txt
fi

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
