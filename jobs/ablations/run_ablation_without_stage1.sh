#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Ablation_Without_Stage1

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

# ==============================================================================
#  Ablation Study: Without Stage 1
# ==============================================================================
# Runs: Kvasir → PathMNIST → ISIC2019
# Stage 2: 50 epochs, Stage 3: 50 epochs
# ==============================================================================

set -e

# Configuration
ABLATION_DIR="ablation_study/without_stage1"
SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE="vit_h"
IMG_SIZE=224
SEED=42

# Training configuration
STAGE2_EPOCHS=50
STAGE3_EPOCHS=50

# Dataset configurations (KVASIR FIRST)
declare -A dataset_configs
dataset_configs["kvasir"]="type=custom,num_classes=8,data_dir=$VTRL_MED_ROOT/data/kvasir-dataset"
dataset_configs["pathmnist"]="type=medmnist,num_classes=9,data_dir="
dataset_configs["isic2019"]="type=custom,num_classes=8,data_dir=$VTRL_MED_ROOT/data/isic2019"

# Dataset order (KVASIR FIRST)
datasets=("kvasir" "pathmnist" "isic2019")

echo "===================================================================="
echo "Ablation Study: Without Stage 1"
echo "===================================================================="
echo ""
echo "Testing: Does Stage 1 (SAM fine-tuning) improve performance?"
echo "Method: Skip Stage 1, use pretrained SAM directly"
echo ""
echo "Configuration:"
echo "  - Stage 2 epochs: $STAGE2_EPOCHS"
echo "  - Stage 3 epochs: $STAGE3_EPOCHS"
echo ""
echo "Dataset order:"
echo "  1. Kvasir (8 classes) - FIRST"
echo "  2. PathMNIST (9 classes)"
echo "  3. ISIC2019 (8 classes)"
echo ""
echo "Output directory: $ABLATION_DIR"
echo "===================================================================="
echo ""

# Create directory structure
mkdir -p "$ABLATION_DIR"

# ==============================================================================
#  Stage 2: Clustering with Pretrained SAM (Without Stage 1)
# ==============================================================================

echo ""
echo "===================================================================="
echo "STAGE 2: Clustering with Pretrained SAM (Without Stage 1)"
echo "===================================================================="
echo ""

for dataset in "${datasets[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Dataset: $dataset"
    echo "--------------------------------------------------------------------"
    
    # Parse dataset config
    config="${dataset_configs[$dataset]}"
    IFS=',' read -ra config_parts <<< "$config"
    
    data_type=""
    num_classes=""
    data_dir=""
    
    for part in "${config_parts[@]}"; do
        key="${part%%=*}"
        value="${part#*=}"
        
        case "$key" in
            type) data_type="$value" ;;
            num_classes) num_classes="$value" ;;
            data_dir) data_dir="$value" ;;
        esac
    done
    
    # Create output directory
    stage2_dir="$ABLATION_DIR/$dataset/stage2"
    mkdir -p "$stage2_dir"
    
    # Check if already completed
    stage2_checkpoint="$stage2_dir/stage2_wo_stage1_${dataset}_${IMG_SIZE}.pth"
    if [ -f "$stage2_checkpoint" ]; then
        echo "✓ Stage 2 already completed: $stage2_checkpoint"
        echo "  Skipping..."
        continue
    fi

    # Build command
    cmd="python scripts/stage2_unified.py \
        --dataset $dataset \
        --pseudo_label_method wo_stage1 \
        --checkpoint_dir /nonexistent \
        --output_dir $stage2_dir \
        --sam_checkpoint $SAM_CHECKPOINT \
        --sam_model_type $SAM_MODEL_TYPE \
        --img_size $IMG_SIZE \
        --epochs $STAGE2_EPOCHS \
        --batch_size 1024 \
        --lr 6e-4 \
        --num_heads 10 \
        --eval_frequency 10 \
        --num_workers 16 \
        --top_k_ratio 0.5 \
        --seed $SEED"

    # Add data_dir for custom datasets
    if [ "$data_type" = "custom" ] && [ -n "$data_dir" ]; then
        # Check if data directory exists
        if [ ! -d "$data_dir" ]; then
            echo "❌ ERROR: Data directory not found: $data_dir"
            echo "   Skipping $dataset..."
            continue
        fi
        cmd="$cmd --data_dir $data_dir"
    fi

    # Run Stage 2
    echo ""
    echo "Running Stage 2 (Without Stage 1 fine-tuning)..."
    echo "  Epochs: $STAGE2_EPOCHS"
    echo "  Output: $stage2_dir"
    echo ""

    eval $cmd

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Stage 2 completed for $dataset"
        echo ""
        echo "Stage 2 Results for $dataset:"
        if [ -f "$stage2_dir/stage2_wo_stage1_results_${dataset}_${IMG_SIZE}.json" ]; then
            python -c "
import json
with open('$stage2_dir/stage2_wo_stage1_results_${dataset}_${IMG_SIZE}.json', 'r') as f:
    results = json.load(f)
    print(f\"  Best ACC: {results['best_head_acc']:.4f}\")
    print(f\"  Best Head: #{results['best_head_idx']}\")
" 2>/dev/null
        fi
        echo ""
    else
        echo ""
        echo "❌ Stage 2 failed for $dataset"
        echo "   Stopping here. Please check the error above."
        exit 1
    fi
done

# ==============================================================================
#  Stage 3: Joint Fine-Tuning
# ==============================================================================

echo ""
echo "===================================================================="
echo "STAGE 3: Joint Fine-Tuning (from Stage 2 Without Stage 1)"
echo "===================================================================="
echo ""

for dataset in "${datasets[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Dataset: $dataset"
    echo "--------------------------------------------------------------------"

    # Parse dataset config
    config="${dataset_configs[$dataset]}"
    IFS=',' read -ra config_parts <<< "$config"

    data_type=""
    num_classes=""
    data_dir=""

    for part in "${config_parts[@]}"; do
        key="${part%%=*}"
        value="${part#*=}"

        case "$key" in
            type) data_type="$value" ;;
            num_classes) num_classes="$value" ;;
            data_dir) data_dir="$value" ;;
        esac
    done

    # Create output directory
    stage2_dir="$ABLATION_DIR/$dataset/stage2"
    stage3_dir="$ABLATION_DIR/$dataset/stage3"
    mkdir -p "$stage3_dir"

    # Check if Stage 2 exists
    stage2_checkpoint="$stage2_dir/stage2_wo_stage1_${dataset}_${IMG_SIZE}.pth"
    if [ ! -f "$stage2_checkpoint" ]; then
        echo "❌ ERROR: Stage 2 checkpoint not found: $stage2_checkpoint"
        echo "   Please run Stage 2 first"
        continue
    fi

    # Check if Stage 3 already completed
    stage3_checkpoint="$stage3_dir/stage3_sam_${dataset}_best.pth"
    if [ -f "$stage3_checkpoint" ]; then
        echo "✓ Stage 3 already completed - skipping"

        # Print existing Stage 3 results
        if [ -f "$stage3_dir/stage3_sam_results_${dataset}.json" ]; then
            python -c "
import json
with open('$stage3_dir/stage3_sam_results_${dataset}.json', 'r') as f:
    results = json.load(f)
    metrics = results['final_best_metrics']
    print(f\"  Stage 3 Final Results (existing):\")
    print(f\"    Visual ACC: {metrics['vis_acc']:.4f}\")
    print(f\"    Visual NMI: {metrics['vis_nmi']:.4f}\")
    print(f\"    Visual ARI: {metrics['vis_ari']:.4f}\")
    print(f\"    ECE: {metrics['ece']:.4f}\")
" 2>/dev/null
        fi
        echo ""
        continue
    fi

    # Create a temporary Stage 1 checkpoint from pretrained SAM
    temp_stage1_dir="$ABLATION_DIR/$dataset/stage1_pretrained"
    mkdir -p "$temp_stage1_dir"
    temp_stage1_checkpoint="$temp_stage1_dir/stage1_sam_simclr_${dataset}_${IMG_SIZE}.pth"

    if [ ! -f "$temp_stage1_checkpoint" ]; then
        echo "Creating pretrained SAM checkpoint..."

        python -c "
import torch
from segment_anything import sam_model_registry

print('Loading pretrained SAM...')
sam = sam_model_registry['$SAM_MODEL_TYPE'](checkpoint='$SAM_CHECKPOINT')
torch.save(sam.image_encoder.state_dict(), '$temp_stage1_checkpoint')
print('✓ Created pretrained SAM checkpoint')
"
    fi

    # Rename Stage 2 checkpoint to match Stage 3 expectations
    stage2_checkpoint_for_s3="$stage2_dir/stage2_ours_${dataset}_${IMG_SIZE}.pth"

    if [ ! -f "$stage2_checkpoint_for_s3" ]; then
        echo "Linking Stage 2 checkpoint for Stage 3..."
        ln -sf "$(basename $stage2_checkpoint)" "$stage2_checkpoint_for_s3"
    fi

    # Build command
    cmd="python scripts/stage3_unified.py \
        --dataset $dataset \
        --checkpoint_dir $temp_stage1_dir \
        --stage2_dir $stage2_dir \
        --output_dir $stage3_dir \
        --sam_checkpoint $SAM_CHECKPOINT \
        --sam_model_type $SAM_MODEL_TYPE \
        --img_size $IMG_SIZE \
        --unfrozen_blocks 4 \
        --epochs $STAGE3_EPOCHS \
        --batch_size 512 \
        --lr_backbone 5e-6 \
        --lr_head 5e-5 \
        --num_neighbors 50 \
        --reliability_threshold 0.8 \
        --confidence_threshold 0.8 \
        --lambda_balance_start 2.0 \
        --lambda_balance_end 0.5 \
        --lambda_anneal_epochs 25 \
        --lambda_entropy 0.2 \
        --lambda_swap 0.1 \
        --eval_frequency 5 \
        --num_workers 16 \
        --seed $SEED"

    # Add data_dir for custom datasets
    if [ "$data_type" = "custom" ] && [ -n "$data_dir" ]; then
        cmd="$cmd --data_dir $data_dir"
    fi

    # Run Stage 3
    echo ""
    echo "Running Stage 3 (Joint Fine-Tuning)..."
    echo "  Epochs: $STAGE3_EPOCHS"
    echo "  Stage 1 (pretrained): $temp_stage1_checkpoint"
    echo "  Stage 2: $stage2_checkpoint"
    echo "  Output: $stage3_dir"
    echo ""

    eval $cmd

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Stage 3 completed for $dataset"
        echo ""
        echo "Stage 3 Results for $dataset:"
        if [ -f "$stage3_dir/stage3_sam_results_${dataset}.json" ]; then
            python -c "
import json
with open('$stage3_dir/stage3_sam_results_${dataset}.json', 'r') as f:
    results = json.load(f)
    metrics = results['final_best_metrics']
    print(f\"  Visual ACC: {metrics['vis_acc']:.4f}\")
    print(f\"  Visual NMI: {metrics['vis_nmi']:.4f}\")
    print(f\"  Visual ARI: {metrics['vis_ari']:.4f}\")
    print(f\"  ECE: {metrics['ece']:.4f}\")
" 2>/dev/null
        fi
        echo ""
    else
        echo ""
        echo "❌ Stage 3 failed for $dataset"
        echo "   Stopping here. Please check the error above."
        exit 1
    fi
done

# ==============================================================================
#  Summary
# ==============================================================================

echo ""
echo "===================================================================="
echo "Ablation Study Complete: Without Stage 1"
echo "===================================================================="
echo ""
echo "Results directory: $ABLATION_DIR"
echo ""
echo "Training configuration:"
echo "  - Stage 2: $STAGE2_EPOCHS epochs"
echo "  - Stage 3: $STAGE3_EPOCHS epochs"
echo ""

# Print summary table
echo "Summary of Results:"
echo "-------------------"
for dataset in "${datasets[@]}"; do
    echo ""
    echo "Dataset: $dataset"

    # Stage 2 results
    s2_results="$ABLATION_DIR/$dataset/stage2/stage2_wo_stage1_results_${dataset}_${IMG_SIZE}.json"
    if [ -f "$s2_results" ]; then
        echo -n "  Stage 2: "
        python -c "
import json
with open('$s2_results', 'r') as f:
    results = json.load(f)
    print(f\"ACC = {results['best_head_acc']:.4f}\")
" 2>/dev/null || echo "Error reading results"
    else
        echo "  Stage 2: Not completed"
    fi

    # Stage 3 results
    s3_results="$ABLATION_DIR/$dataset/stage3/stage3_sam_results_${dataset}.json"
    if [ -f "$s3_results" ]; then
        echo -n "  Stage 3: "
        python -c "
import json
with open('$s3_results', 'r') as f:
    results = json.load(f)
    metrics = results['final_best_metrics']
    print(f\"Visual ACC = {metrics['vis_acc']:.4f}, NMI = {metrics['vis_nmi']:.4f}, ARI = {metrics['vis_ari']:.4f}\")
" 2>/dev/null || echo "Error reading results"
    else
        echo "  Stage 3: Not completed"
    fi
done

echo ""
echo "===================================================================="
