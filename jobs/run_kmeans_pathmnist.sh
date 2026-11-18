#!/bin/bash
#$ -q gpu@@coba-a40
#$ -l gpu_card=1
#$ -pe smp 4
#$ -m abe
#$ -N KMeans_PathMNIST

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

echo "K-Means Backbone Comparison: PathMNIST (3 backbones, skip sam_adapted)"

python evaluation/kmeans_backbone_comparison.py \
    --project_root "$VTRL_MED_ROOT" \
    --batch_size 256 \
    --datasets pathmnist \
    --backbones raw_pixels resnet50 sam_frozen \
    --output results/kmeans_backbone_pathmnist.json

echo "Done: PathMNIST (raw_pixels, resnet50, sam_frozen)"
