#!/bin/bash
#$ -q gpu@@coba-a40
#$ -l gpu_card=1
#$ -pe smp 4
#$ -m abe
#$ -N KMeans_Backbone_Comparison

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

# ==============================================================================
#  K-Means Backbone Comparison
#  3 datasets x 4 backbones = 12 evaluations
#  Uses GPU k-means via faiss-gpu
# ==============================================================================

echo "===================================================================="
echo "K-Means Backbone Comparison Experiment"
echo "Datasets: PathMNIST, Kvasir, ISIC2019"
echo "Backbones: Raw Pixels, ResNet-50, SAM Frozen, SAM Adapted"
echo "===================================================================="

# Verify Stage 1 checkpoints exist
echo ""
echo "Checking Stage 1 checkpoints..."
for ckpt in \
    "checkpoints/sam_spice/stage1/pathmnist/stage1_sam_simclr_pathmnist_224.pth" \
    "checkpoints/sam_spice_kvasir_isic/stage1/kvasir/stage1_sam_simclr_kvasir_224.pth" \
    "checkpoints/sam_spice_kvasir_isic/stage1/isic2019/stage1_sam_simclr_isic2019_224.pth"; do
    if [ -f "$ckpt" ]; then
        echo "  OK: $ckpt"
    else
        echo "  MISSING: $ckpt"
    fi
done

echo ""
echo "Running evaluation..."

python evaluation/kmeans_backbone_comparison.py \
    --project_root "$VTRL_MED_ROOT" \
    --batch_size 256 \
    --output results/kmeans_backbone_comparison.json

echo ""
echo "===================================================================="
echo "Evaluation complete!"
echo "Results: results/kmeans_backbone_comparison.json"
echo "===================================================================="
