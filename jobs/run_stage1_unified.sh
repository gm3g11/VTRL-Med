#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Stage1_Unified_Pipeline

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

# ==============================================================================
#  Unified Stage 1: MedMNIST + Kvasir + ISIC2019
# ==============================================================================

set -e

MAIN_OUTPUT_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice"
SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"

# MedMNIST datasets (no data_dir needed)
medmnist_datasets=(
    "dermamnist"
    "pathmnist"
    "octmnist"
    "bloodmnist"
    "pneumoniamnist"
    "retinamnist"
    "breastmnist"
    "tissuemnist"
    "organamnist"
    "organcmnist"
    "organsmnist"
)

# Custom datasets (require data_dir)
declare -A custom_datasets
custom_datasets=(
    ["kvasir"]="$VTRL_MED_ROOT/data/kvasir-dataset"
    ["isic2019"]="$VTRL_MED_ROOT/data/isic2019"
)

# ==============================================================================
#  Train MedMNIST Datasets
# ==============================================================================

echo "===================================================================="
echo "Training MedMNIST Datasets"
echo "===================================================================="

for dataset in "${medmnist_datasets[@]}"; do
    echo ""
    echo "🚀 Starting Stage 1 for: $dataset"
    
    stage1_dir="$MAIN_OUTPUT_DIR/stage1/$dataset"
    mkdir -p "$stage1_dir"
    
    # Check if already trained
    checkpoint="$stage1_dir/stage1_sam_simclr_${dataset}_224.pth"
    if [ -f "$checkpoint" ]; then
        echo "✓ Checkpoint exists: $checkpoint"
        echo "  Skipping..."
        continue
    fi
    
    # Train Stage 1
    python scripts/stage1_unified.py \
        --dataset "$dataset" \
        --sam_checkpoint "$SAM_CHECKPOINT" \
        --lr 5e-4 \
        --batch_size 512 \
        --epochs 200 \
        --unfrozen_blocks 4 \
        --use_amp \
        --num_workers 16 \
        --output_dir "$stage1_dir"
    
    echo "✅ Stage 1 completed for $dataset"
done

# ==============================================================================
#  Train Custom Datasets (Kvasir, ISIC2019)
# ==============================================================================

echo ""
echo "===================================================================="
echo "Training Custom Datasets"
echo "===================================================================="

for dataset in "${!custom_datasets[@]}"; do
    data_dir="${custom_datasets[$dataset]}"
    
    echo ""
    echo "🚀 Starting Stage 1 for: $dataset"
    echo "   Data dir: $data_dir"
    
    stage1_dir="$MAIN_OUTPUT_DIR/stage1/$dataset"
    mkdir -p "$stage1_dir"
    
    # Check if already trained
    checkpoint="$stage1_dir/stage1_sam_simclr_${dataset}_224.pth"
    if [ -f "$checkpoint" ]; then
        echo "✓ Checkpoint exists: $checkpoint"
        echo "  Skipping..."
        continue
    fi
    
    # Verify data directory exists
    if [ ! -d "$data_dir" ]; then
        echo "❌ ERROR: Data directory not found: $data_dir"
        echo "  Skipping $dataset..."
        continue
    fi
    
    # Train Stage 1
    python scripts/stage1_unified.py \
        --dataset "$dataset" \
        --data_dir "$data_dir" \
        --sam_checkpoint "$SAM_CHECKPOINT" \
        --lr 5e-4 \
        --batch_size 512 \
        --epochs 200 \
        --unfrozen_blocks 4 \
        --use_amp \
        --num_workers 16 \
        --output_dir "$stage1_dir"
    
    echo "✅ Stage 1 completed for $dataset"
done

echo ""
echo "===================================================================="
echo "✨ All Stage 1 training completed! ✨"
echo "===================================================================="
