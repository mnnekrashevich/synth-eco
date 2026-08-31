#!/usr/bin/env bash
# run_all.sh — Full pipeline: generate -> evaluate -> figures
#
# Prerequisites:
#   1. export OPENROUTER_API_KEY="sk-or-v1-..."
#   2. pip install numpy pandas scipy matplotlib requests
#
# Usage:
#   bash run_all.sh                    # full pipeline
#   bash run_all.sh --skip-generate    # skip generation, re-evaluate only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Check prerequisites ---
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY is not set."
    echo "Run:  export OPENROUTER_API_KEY='sk-or-v1-...'"
    exit 1
fi

python3 -c "import numpy, pandas, scipy, matplotlib" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install --break-system-packages numpy pandas scipy matplotlib requests
}

SKIP_GEN=false
for arg in "$@"; do
    case "$arg" in
        --skip-generate) SKIP_GEN=true ;;
    esac
done

# --- Step 1: Build reference (idempotent) ---
echo ""
echo "============================================"
echo "  Step 1/4: Build reference statistics"
echo "============================================"
python3 scripts/build_reference.py

# --- Step 2: Generate synthetic data ---
if [ "$SKIP_GEN" = false ]; then
    echo ""
    echo "============================================"
    echo "  Step 2/4: Generate synthetic data"
    echo "  (3 new cheap models x 3 conditions x 2 seeds, parallel)"
    echo "  Est. time: 20-35 min, cost: ~\$0.04"
    echo "============================================"
    python3 scripts/generate_data.py
else
    echo ""
    echo "  [SKIP] Step 2: Generation skipped (--skip-generate)"
fi

# --- Step 3: Evaluate ---
echo ""
echo "============================================"
echo "  Step 3/5: Evaluate + bootstrap CIs"
echo "============================================"
python3 scripts/evaluate.py

# --- Step 3b: Extended analysis (seed stability, gender gap, JSD, Wasserstein) ---
echo ""
echo "============================================"
echo "  Step 3b/5: Extended analysis"
echo "============================================"
python3 scripts/extended_analysis.py

# --- Step 3c: Calibration demonstration (post-stratification recipe) ---
echo ""
echo "============================================"
echo "  Step 3c/5: Calibration demonstration"
echo "============================================"
python3 scripts/calibration_demo.py

# --- Step 4: Generate figures ---
echo ""
echo "============================================"
echo "  Step 4/5: Generate figures"
echo "============================================"
python3 scripts/make_figures.py

echo ""
echo "============================================"
echo "  DONE"
echo "  Results:  results/evaluation.json"
echo "            results/extended_analysis.json"
echo "            results/calibration_demo.json"
echo "  Figures:  paper/figures/*.png"
echo "============================================"
