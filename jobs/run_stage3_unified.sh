#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Stage3_Unified

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

# ==============================================================================
#  Stage 3: Joint Fine-Tuning (Topology version)
# ==============================================================================

set -e

MAIN_OUTPUT_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice"
SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"

datasets=(
    "dermamnist"
    "pathmnist"
    "octmnist"
    "bloodmnist"
)

for dataset in "${datasets[@]}"; do
    echo "===================================================================="
    echo "Starting Stage 3 for: $dataset"
    echo "===================================================================="

    stage1_dir="$MAIN_OUTPUT_DIR/stage1/$dataset"
    stage2_dir="$MAIN_OUTPUT_DIR/stage2/$dataset"
    stage3_dir="$MAIN_OUTPUT_DIR/stage3/$dataset"
    mkdir -p "$stage3_dir"

    python scripts/stage3_unified.py \
        --dataset "$dataset" \
        --batch_size 512 \
        --epochs 50 \
        --lr_backbone 5e-6 \
        --lr_head 5e-5 \
        --num_workers 16 \
        --eval_frequency 2 \
        --sam_checkpoint "$SAM_CHECKPOINT" \
        --checkpoint_dir "$stage1_dir" \
        --stage2_dir "$stage2_dir" \
        --output_dir "$stage3_dir"

    echo "Stage 3 completed for $dataset"
done

echo "===================================================================="
echo "All Stage 3 training completed!"
echo "===================================================================="
