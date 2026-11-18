#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Ablation_SwAV_Full

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

# Configuration
ABLATION_DIR="ablation_study/swav"
SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE="vit_h"
IMG_SIZE=224
SEED=42
METHOD="swav"

STAGE2_EPOCHS=50
STAGE3_EPOCHS=50
STAGE1_BASE_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice/stage1"

declare -A dataset_configs
dataset_configs["kvasir"]="type=custom,num_classes=8,data_dir=$VTRL_MED_ROOT/data/kvasir-dataset"
dataset_configs["pathmnist"]="type=medmnist,num_classes=9,data_dir="
dataset_configs["isic2019"]="type=custom,num_classes=8,data_dir=$VTRL_MED_ROOT/data/isic2019"

datasets=("kvasir" "pathmnist" "isic2019")

echo "===================================================================="
echo "Ablation Study: SwAV (Stage 2 + Stage 3)"
echo "===================================================================="
echo ""
echo "Method: SwAV (Sinkhorn-Knopp optimal transport)"
echo "Stage 2 epochs: $STAGE2_EPOCHS"
echo "Stage 3 epochs: $STAGE3_EPOCHS"
echo "Output: $ABLATION_DIR"
echo "===================================================================="
echo ""

mkdir -p "$ABLATION_DIR"

# ==============================================================================
#  Stage 2
# ==============================================================================

echo ""
echo "===================================================================="
echo "STAGE 2: SwAV Clustering"
echo "===================================================================="
echo ""

for dataset in "${datasets[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Dataset: $dataset - Stage 2"
    echo "--------------------------------------------------------------------"
    
    config="${dataset_configs[$dataset]}"
    IFS=',' read -ra config_parts <<< "$config"
    
    data_type=""
    data_dir=""
    
    for part in "${config_parts[@]}"; do
        key="${part%%=*}"
        value="${part#*=}"
        case "$key" in
            type) data_type="$value" ;;
            data_dir) data_dir="$value" ;;
        esac
    done
    
    if [ "$data_type" = "custom" ]; then
        stage1_dir="$VTRL_MED_ROOT/checkpoints/sam_spice_kvasir_isic/stage1/$dataset"
    else
        stage1_dir="$STAGE1_BASE_DIR/$dataset"
    fi
    
    stage2_dir="$ABLATION_DIR/$dataset/stage2"
    mkdir -p "$stage2_dir"
    
    stage2_checkpoint="$stage2_dir/stage2_${METHOD}_${dataset}_${IMG_SIZE}.pth"
    if [ -f "$stage2_checkpoint" ]; then
        echo "✓ Stage 2 already completed - skipping"
        continue
    fi

    cmd="python scripts/stage2_unified.py \
        --dataset $dataset \
        --pseudo_label_method $METHOD \
        --checkpoint_dir $stage1_dir \
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
        --seed $SEED"

    if [ "$data_type" = "custom" ] && [ -n "$data_dir" ]; then
        if [ ! -d "$data_dir" ]; then
            echo "❌ ERROR: Data directory not found: $data_dir"
            continue
        fi
        cmd="$cmd --data_dir $data_dir"
    fi

    eval $cmd

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Stage 2 completed for $dataset"

        if [ -f "$stage2_dir/stage2_${METHOD}_results_${dataset}_${IMG_SIZE}.json" ]; then
            python -c "
import json
with open('$stage2_dir/stage2_${METHOD}_results_${dataset}_${IMG_SIZE}.json', 'r') as f:
    results = json.load(f)
    print(f\"  Stage 2 Best ACC: {results['best_head_acc']:.4f}\")
" 2>/dev/null
        fi
        echo ""
    else
        echo "❌ Stage 2 failed for $dataset"
        exit 1
    fi
done

# ==============================================================================
#  Stage 3
# ==============================================================================

echo ""
echo "===================================================================="
echo "STAGE 3: Joint Fine-Tuning"
echo "===================================================================="
echo ""

for dataset in "${datasets[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Dataset: $dataset - Stage 3"
    echo "--------------------------------------------------------------------"

    config="${dataset_configs[$dataset]}"
    IFS=',' read -ra config_parts <<< "$config"

    data_type=""
    data_dir=""

    for part in "${config_parts[@]}"; do
        key="${part%%=*}"
        value="${part#*=}"
        case "$key" in
            type) data_type="$value" ;;
            data_dir) data_dir="$value" ;;
        esac
    done

    if [ "$data_type" = "custom" ]; then
        stage1_dir="$VTRL_MED_ROOT/checkpoints/sam_spice_kvasir_isic/stage1/$dataset"
    else
        stage1_dir="$STAGE1_BASE_DIR/$dataset"
    fi

    stage2_dir="$ABLATION_DIR/$dataset/stage2"
    stage3_dir="$ABLATION_DIR/$dataset/stage3"
    mkdir -p "$stage3_dir"

    stage2_checkpoint="$stage2_dir/stage2_${METHOD}_${dataset}_${IMG_SIZE}.pth"
    if [ ! -f "$stage2_checkpoint" ]; then
        echo "❌ ERROR: Stage 2 checkpoint not found"
        continue
    fi

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

    stage2_checkpoint_for_s3="$stage2_dir/stage2_ours_${dataset}_${IMG_SIZE}.pth"
    if [ ! -f "$stage2_checkpoint_for_s3" ]; then
        ln -sf "$(basename $stage2_checkpoint)" "$stage2_checkpoint_for_s3"
    fi

    cmd="python scripts/stage3_unified.py \
        --dataset $dataset \
        --checkpoint_dir $stage1_dir \
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

    if [ "$data_type" = "custom" ] && [ -n "$data_dir" ]; then
        cmd="$cmd --data_dir $data_dir"
    fi

    eval $cmd

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Stage 3 completed for $dataset"

        if [ -f "$stage3_dir/stage3_sam_results_${dataset}.json" ]; then
            python -c "
import json
with open('$stage3_dir/stage3_sam_results_${dataset}.json', 'r') as f:
    results = json.load(f)
    metrics = results['final_best_metrics']
    print(f\"  Stage 3 Final Results:\")
    print(f\"    Visual ACC: {metrics['vis_acc']:.4f}\")
    print(f\"    Visual NMI: {metrics['vis_nmi']:.4f}\")
    print(f\"    Visual ARI: {metrics['vis_ari']:.4f}\")
    print(f\"    ECE: {metrics['ece']:.4f}\")
" 2>/dev/null
        fi
        echo ""
    else
        echo "❌ Stage 3 failed for $dataset"
        exit 1
    fi
done

# ==============================================================================
#  Summary
# ==============================================================================

echo ""
echo "===================================================================="
echo "Ablation Complete: SwAV (Full Pipeline)"
echo "===================================================================="
echo ""
echo "Summary:"
for dataset in "${datasets[@]}"; do
    echo ""
    echo "Dataset: $dataset"

    s2_results="$ABLATION_DIR/$dataset/stage2/stage2_${METHOD}_results_${dataset}_${IMG_SIZE}.json"
    s3_results="$ABLATION_DIR/$dataset/stage3/stage3_sam_results_${dataset}.json"

    if [ -f "$s2_results" ]; then
        python -c "
import json
with open('$s2_results', 'r') as f:
    results = json.load(f)
    print(f\"  Stage 2: ACC = {results['best_head_acc']:.4f}\")
" 2>/dev/null
    fi

    if [ -f "$s3_results" ]; then
        python -c "
import json
with open('$s3_results', 'r') as f:
    results = json.load(f)
    metrics = results['final_best_metrics']
    print(f\"  Stage 3: Visual ACC = {metrics['vis_acc']:.4f}, NMI = {metrics['vis_nmi']:.4f}, ARI = {metrics['vis_ari']:.4f}\")
" 2>/dev/null
    fi
done

echo ""
echo "===================================================================="
