#!/bin/bash
#$ -q gpu@@coba-a40
#$ -l gpu_card=1
#$ -pe smp 4
#$ -m abe
#$ -N KMeans_PathMNIST_Adapted

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

# This job should be submitted after Stage 1 PathMNIST training completes
CHECKPOINT="checkpoints/sam_spice/stage1/pathmnist/stage1_sam_simclr_pathmnist_224.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: PathMNIST Stage 1 checkpoint not found: $CHECKPOINT"
    echo "Submit this job after run_stage1_pathmnist.sh completes."
    exit 1
fi

echo "K-Means Backbone Comparison: PathMNIST SAM Adapted (final 1/12)"

python evaluation/kmeans_backbone_comparison.py \
    --project_root "$VTRL_MED_ROOT" \
    --batch_size 256 \
    --datasets pathmnist \
    --backbones sam_adapted \
    --output results/kmeans_backbone_pathmnist_adapted.json

echo "Done: PathMNIST sam_adapted (12/12 complete)"
