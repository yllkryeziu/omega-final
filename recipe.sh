#!/usr/bin/env bash
# ============================================================================
#  recipe.sh — Full pipeline: data → labels → training → prediction → submission
#
#  Usage:
#    bash recipe.sh          # Run everything sequentially
#    bash recipe.sh train    # Run only the training step
#    bash recipe.sh predict  # Run only prediction + ensemble
#
#  Prerequisites:
#    - Python 3.10+ with venv activated (source .venv/bin/activate)
#    - Data downloaded (make download_data_from_s3)
#    - PyTorch with ROCm, XGBoost, scipy installed
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

STEP="${1:-all}"

# ---------- colors ----------
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
step() { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
         echo -e "${CYAN}  $*${NC}"; \
         echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ============================================================================
# STEP 1: FUSE LABELS
# ============================================================================
run_labels() {
    step "Step 1 / 5 — Fuse weak labels into consensus training targets"
    log "Fusing RADD + GLAD-L + GLAD-S2 via majority vote..."
    python3 -u build_dataset.py
    log "Labels saved to data/fused-labels/"
}

# ============================================================================
# STEP 2: TRAIN MODELS
# ============================================================================
run_train() {
    step "Step 2 / 5 — Train XGBoost + U-TAE (4 folds)"

    log "Training XGBoost baseline..."
    python3 -u train_xgb.py
    log "XGBoost model saved to models/xgb_baseline.json"

    log "Training U-TAE fold 0..."
    python3 -u train_utae.py --fold 0 --epochs 50 --batch_size 8

    log "Training U-TAE fold 1..."
    python3 -u train_utae.py --fold 1 --epochs 50 --batch_size 8

    log "Training U-TAE fold 2..."
    python3 -u train_utae.py --fold 2 --epochs 50 --batch_size 8

    log "Training U-TAE fold 3..."
    python3 -u train_utae.py --fold 3 --epochs 50 --batch_size 8

    log "All models saved to models/"
}

# ============================================================================
# STEP 3: GENERATE PREDICTIONS
# ============================================================================
run_predict() {
    step "Step 3 / 5 — Generate test predictions"

    log "Predicting with XGBoost → submission_xgb/prob_*.tif"
    python3 -u predict_xgb.py

    log "Predicting with U-TAE (4-fold ensemble) → submission/prob_*.tif"
    python3 -u predict_utae.py --threshold 0.15

    log "Probability maps ready for ensemble"
}

# ============================================================================
# STEP 4: ENSEMBLE + SUBMISSION
# ============================================================================
run_ensemble() {
    step "Step 4 / 5 — Ensemble fusion + GeoJSON submission"

    log "Running gated union ensemble (U-TAE + XGBoost + NDVI + NBR + S1)..."
    python3 -u ensemble_v3.py

    log "Submission saved to submission/submission.geojson"
}

# ============================================================================
# STEP 5: VISUALIZE (optional)
# ============================================================================
run_visualize() {
    step "Step 5 / 5 — Generate visualizations"

    mkdir -p figures
    for tile in 18NVJ_1_6 18NYH_2_1 33NTE_5_1 47QMA_6_2 48PWA_0_6; do
        log "Visualizing ${tile}..."
        python3 -u visualize.py --tile "$tile" --split test || true
    done
    log "Figures saved to figures/"
}

# ============================================================================
# DISPATCH
# ============================================================================
case "$STEP" in
    all)
        run_labels
        run_train
        run_predict
        run_ensemble
        run_visualize
        ;;
    labels)    run_labels ;;
    train)     run_train ;;
    predict)   run_predict ;;
    ensemble)  run_ensemble ;;
    visualize) run_visualize ;;
    *)
        echo "Usage: bash recipe.sh [all|labels|train|predict|ensemble|visualize]"
        exit 1
        ;;
esac

echo ""
step "Done"
log "Upload submission/submission.geojson to the leaderboard"
