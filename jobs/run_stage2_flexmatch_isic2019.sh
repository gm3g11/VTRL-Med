#!/bin/bash
#$ -q gpu@@coba-a40
#$ -l gpu_card=1
#$ -pe smp 8
#$ -m abe
#$ -N S2_flex_isic

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

python scripts/stage2_unified.py \
    --dataset isic2019 \
    --pseudo_label_method flexmatch \
    --checkpoint_dir checkpoints/sam_spice_kvasir_isic/stage1/isic2019 \
    --output_dir checkpoints/stage2_fixmatch_flexmatch/flexmatch/isic2019 \
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \
    --sam_model_type vit_h \
    --img_size 224 \
    --epochs 15 \
    --batch_size 512 \
    --lr 6e-4 \
    --num_heads 10 \
    --eval_frequency 5 \
    --num_workers 8 \
    --fixmatch_threshold 0.95 \
    --seed 42 \
    --data_dir data/isic2019
