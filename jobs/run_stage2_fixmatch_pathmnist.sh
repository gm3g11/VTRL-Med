#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N S2_fix_path

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

python scripts/stage2_unified.py \
    --dataset pathmnist \
    --pseudo_label_method fixmatch \
    --checkpoint_dir checkpoints/sam_spice/stage1/pathmnist \
    --output_dir checkpoints/stage2_fixmatch_flexmatch/fixmatch/pathmnist \
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \
    --sam_model_type vit_h \
    --img_size 224 \
    --epochs 15 \
    --batch_size 1024 \
    --lr 6e-4 \
    --num_heads 10 \
    --eval_frequency 5 \
    --num_workers 16 \
    --fixmatch_threshold 0.95 \
    --seed 42
