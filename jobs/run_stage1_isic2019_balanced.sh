#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Stage1_ISIC2019_Balanced

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

# ==============================================================================
#  Stage 1: ISIC2019 with balanced sampling (SAM + SimCLR)
# ==============================================================================

SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"
OUTPUT_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice/stage1/isic2019_balanced"

mkdir -p "$OUTPUT_DIR"

CHECKPOINT="$OUTPUT_DIR/stage1_sam_simclr_isic2019_224.pth"
if [ -f "$CHECKPOINT" ]; then
    echo "Checkpoint exists: $CHECKPOINT"
    echo "Skipping training."
else
    echo "===================================================================="
    echo "Training Stage 1 SimCLR for ISIC2019 (balanced sampling)"
    echo "===================================================================="

    python scripts/stage1_unified.py \
        --dataset isic2019 \
        --data_dir "$VTRL_MED_ROOT/data/isic2019" \
        --sam_model_type vit_h \
        --sam_checkpoint "$SAM_CHECKPOINT" \
        --epochs 200 \
        --batch_size 512 \
        --lr 5e-4 \
        --unfrozen_blocks 4 \
        --use_amp \
        --balanced_sampling \
        --num_workers 16 \
        --output_dir "$OUTPUT_DIR"

    echo "===================================================================="
    echo "Stage 1 ISIC2019 (balanced) training complete!"
    echo "Checkpoint: $CHECKPOINT"
    echo "===================================================================="
fi

# ==============================================================================
#  K-Means Evaluation: ISIC2019 SAM Adapted (balanced)
# ==============================================================================

echo ""
echo "===================================================================="
echo "Running K-Means evaluation: ISIC2019 SAM Adapted (balanced)"
echo "===================================================================="

python evaluation/kmeans_backbone_comparison.py \
    --project_root "$VTRL_MED_ROOT" \
    --batch_size 256 \
    --datasets isic2019 \
    --backbones sam_adapted \
    --stage1_override "isic2019:checkpoints/sam_spice/stage1/isic2019_balanced/stage1_sam_simclr_isic2019_224.pth" \
    --output results/kmeans_backbone_isic2019_balanced.json

echo "===================================================================="
echo "ISIC2019 balanced evaluation complete!"
echo "===================================================================="
