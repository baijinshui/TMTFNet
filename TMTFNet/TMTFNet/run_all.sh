#!/bin/bash
# ============================================================
# TMTFNet - Complete Experiment Pipeline
# ============================================================
# This script runs ALL experiments for the paper.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh              # synthetic data (fast, ~15 min)
#   ./run_all.sh --real       # real data (needs download first)
#
# Environment: boluo_huo (Anaconda)
# GPU: RTX 5090 recommended
# ============================================================

set -e

# Activate conda environment
echo "Activating conda environment: boluo_huo"
eval "$(conda shell.bash hook)"
conda activate boluo_huo

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q torch numpy pandas scikit-learn matplotlib seaborn pyyaml tqdm 2>/dev/null || true

# Parse arguments
USE_REAL=""
if [ "$1" == "--real" ]; then
    USE_REAL="--use_real_data"
    echo "Mode: Real datasets"
    
    # Download data if needed
    if [ ! -d "./data/UCI HAR Dataset" ] || [ ! -f "./data/ETTh1.csv" ]; then
        echo "Downloading datasets..."
        bash download_data.sh
    fi
else
    echo "Mode: Synthetic data (fast)"
fi

echo ""
echo "============================================================"
echo "Starting TMTFNet Experiment Pipeline"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU mode')"
echo "============================================================"
echo ""

# Run all experiments
python run_experiments.py \
    $USE_REAL \
    --d_model 64 \
    --n_heads 4 \
    --n_layers 2 \
    --batch_size 128 \
    --max_epochs 50 \
    --patience 10 \
    --lr 0.001 \
    --seed 42 \
    --num_workers 4 \
    --exp all

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "Results:  ./results/"
echo "Figures:  ./figures/"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Check results JSON files in ./results/"
echo "  2. Check generated figures in ./figures/"
echo "  3. Share results with Claude for paper writing"
