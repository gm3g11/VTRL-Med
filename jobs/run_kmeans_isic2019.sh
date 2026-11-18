#!/bin/bash
#$ -q gpu@@coba-a40
#$ -l gpu_card=1
#$ -pe smp 4
#$ -m abe
#$ -N KMeans_ISIC2019

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

echo "K-Means Backbone Comparison: ISIC2019 (all 4 backbones)"

python evaluation/kmeans_backbone_comparison.py \
    --project_root "$VTRL_MED_ROOT" \
    --batch_size 256 \
    --datasets isic2019 \
    --backbones raw_pixels resnet50 sam_frozen sam_adapted \
    --output results/kmeans_backbone_isic2019.json

echo "Done: ISIC2019 (all backbones)"
