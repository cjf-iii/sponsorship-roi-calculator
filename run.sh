#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# run.sh — End-to-end pipeline: generate data → train model → launch app
# ═══════════════════════════════════════════════════════════════════════
# Usage:
#   bash run.sh          # Full pipeline (generate + train + run)
#   bash run.sh --app    # Skip data/training, just launch the app
#
# Prerequisites:
#   pip install -r requirements.txt
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# Resolve the project root directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo " Sports Sponsorship ROI Calculator"
echo "═══════════════════════════════════════════════════════════════"

# ──────────────────────────────────────────────────────────────────────
# Skip data generation and training if --app flag is passed
# This is useful when iterating on the Streamlit UI without waiting
# for the pipeline to regenerate data every time
# ──────────────────────────────────────────────────────────────────────
if [[ "${1:-}" != "--app" ]]; then

    echo ""
    echo "Step 1/3: Generating synthetic sponsorship deal data..."
    echo "───────────────────────────────────────────────────────────────"
    python data/generate_data.py

    echo ""
    echo "Step 2/3: Training XGBoost models..."
    echo "───────────────────────────────────────────────────────────────"
    python model/train.py

fi

echo ""
echo "Step 3/3: Launching Streamlit app..."
echo "───────────────────────────────────────────────────────────────"
echo "Open http://localhost:8501 in your browser"
echo ""
streamlit run app.py
