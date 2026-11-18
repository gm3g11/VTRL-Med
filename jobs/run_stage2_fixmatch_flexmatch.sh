#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Stage2_FixFlex

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

# ==============================================================================
#  Stage 2: FixMatch & FlexMatch Experiment
# ==============================================================================
# 2 methods x 3 datasets = 6 runs
# Methods: fixmatch, flexmatch
# Datasets: pathmnist, kvasir, isic2019
# Metrics: ACC (Hungarian), NMI, ARI (evaluated every 5 epochs)
# ==============================================================================

set -e

# Configuration
MAIN_OUTPUT_DIR="$VTRL_MED_ROOT/checkpoints/stage2_fixmatch_flexmatch"
SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"

# Methods
methods=(
    "fixmatch"
    "flexmatch"
)

# ==============================================================================
#  Dataset configurations
# ==============================================================================

# Dataset names
datasets=("pathmnist" "kvasir" "isic2019")

# Stage 1 checkpoint dirs
declare -A stage1_dirs
stage1_dirs["pathmnist"]="$VTRL_MED_ROOT/checkpoints/sam_spice/stage1/pathmnist"
stage1_dirs["kvasir"]="$VTRL_MED_ROOT/checkpoints/sam_spice_kvasir_isic/stage1/kvasir"
stage1_dirs["isic2019"]="$VTRL_MED_ROOT/checkpoints/sam_spice_kvasir_isic/stage1/isic2019"

# Data dirs (only needed for kvasir and isic2019)
declare -A data_dirs
data_dirs["kvasir"]="$VTRL_MED_ROOT/data/kvasir-dataset"
data_dirs["isic2019"]="$VTRL_MED_ROOT/data/isic2019"

# ==============================================================================
#  Run Experiments
# ==============================================================================

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo ""
        echo "===================================================================="
        echo "Method: $method | Dataset: $dataset"
        echo "===================================================================="

        stage1_dir="${stage1_dirs[$dataset]}"
        output_dir="$MAIN_OUTPUT_DIR/$method/$dataset"
        mkdir -p "$output_dir"

        # Check if already completed
        result_file="$output_dir/stage2_${method}_results_${dataset}_224.json"
        if [ -f "$result_file" ]; then
            echo "Already completed: $result_file"
            echo "Skipping..."
            continue
        fi

        # Check Stage 1 checkpoint exists
        stage1_checkpoint="$stage1_dir/stage1_sam_simclr_${dataset}_224.pth"
        if [ ! -f "$stage1_checkpoint" ]; then
            echo "Stage 1 checkpoint not found: $stage1_checkpoint"
            echo "Skipping $method on $dataset..."
            continue
        fi

        # Build command
        cmd="python scripts/stage2_unified.py \
            --dataset $dataset \
            --pseudo_label_method $method \
            --checkpoint_dir $stage1_dir \
            --output_dir $output_dir \
            --sam_checkpoint $SAM_CHECKPOINT \
            --sam_model_type vit_h \
            --img_size 224 \
            --epochs 15 \
            --batch_size 512 \
            --lr 6e-4 \
            --num_heads 10 \
            --eval_frequency 5 \
            --num_workers 16 \
            --fixmatch_threshold 0.95 \
            --seed 42"

        # Add data_dir for non-MedMNIST datasets
        if [ -n "${data_dirs[$dataset]}" ]; then
            cmd="$cmd --data_dir ${data_dirs[$dataset]}"
        fi

        echo "Running: $cmd"
        eval $cmd

        if [ $? -eq 0 ]; then
            echo "Completed: $method on $dataset"
        else
            echo "Failed: $method on $dataset"
        fi
    done
done

# ==============================================================================
#  Summary
# ==============================================================================

echo ""
echo "===================================================================="
echo "FixMatch & FlexMatch Experiment Complete!"
echo "===================================================================="
echo ""
echo "Results saved in: $MAIN_OUTPUT_DIR/"
echo ""
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        result_file="$MAIN_OUTPUT_DIR/$method/$dataset/stage2_${method}_results_${dataset}_224.json"
        if [ -f "$result_file" ]; then
            echo "  $method/$dataset: $result_file"
        else
            echo "  $method/$dataset: MISSING"
        fi
    done
done
echo "===================================================================="
