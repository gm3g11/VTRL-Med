#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Stage1_PathMNIST

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

# ==============================================================================
#  Stage 1: PathMNIST (SAM + SimCLR)
# ==============================================================================

SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"
OUTPUT_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice/stage1/pathmnist"

mkdir -p "$OUTPUT_DIR"

# Check if already trained
CHECKPOINT="$OUTPUT_DIR/stage1_sam_simclr_pathmnist_224.pth"
if [ -f "$CHECKPOINT" ]; then
    echo "Checkpoint exists: $CHECKPOINT"
    echo "Skipping training."
    exit 0
fi

echo "===================================================================="
echo "Training Stage 1 SimCLR for PathMNIST"
echo "===================================================================="

python scripts/stage1_unified.py \
    --dataset pathmnist \
    --sam_model_type vit_h \
    --sam_checkpoint "$SAM_CHECKPOINT" \
    --epochs 200 \
    --batch_size 512 \
    --lr 5e-4 \
    --unfrozen_blocks 4 \
    --use_amp \
    --num_workers 16 \
    --output_dir "$OUTPUT_DIR"

echo "===================================================================="
echo "Stage 1 PathMNIST training complete!"
echo "Checkpoint: $CHECKPOINT"
echo "===================================================================="

# ==============================================================================
#  K-Means Evaluation: PathMNIST SAM Adapted (run immediately after training)
# ==============================================================================

echo ""
echo "===================================================================="
echo "Running K-Means evaluation: PathMNIST SAM Adapted"
echo "===================================================================="

python evaluation/kmeans_backbone_comparison.py \
    --project_root "$VTRL_MED_ROOT" \
    --batch_size 256 \
    --datasets pathmnist \
    --backbones sam_adapted \
    --output results/kmeans_backbone_pathmnist_adapted.json

echo "===================================================================="
echo "PathMNIST SAM Adapted evaluation complete!"
echo "===================================================================="
