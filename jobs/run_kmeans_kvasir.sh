#!/bin/bash
#$ -q gpu@@coba-a40
#$ -l gpu_card=1
#$ -pe smp 4
#$ -m abe
#$ -N KMeans_Kvasir

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

echo "K-Means Backbone Comparison: Kvasir (all 4 backbones)"

python evaluation/kmeans_backbone_comparison.py \
    --project_root "$VTRL_MED_ROOT" \
    --batch_size 256 \
    --datasets kvasir \
    --backbones raw_pixels resnet50 sam_frozen sam_adapted \
    --output results/kmeans_backbone_kvasir.json

echo "Done: Kvasir (all backbones)"
