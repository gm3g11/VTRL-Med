#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Ablation_NoStage2

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

# Configuration
ABLATION_DIR="ablation_study/without_stage2"
SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE="vit_h"
IMG_SIZE=224
SEED=42

STAGE3_EPOCHS=50
STAGE1_BASE_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice/stage1"

declare -A dataset_configs
dataset_configs["pathmnist"]="type=medmnist,num_classes=9,data_dir="
dataset_configs["isic2019"]="type=custom,num_classes=8,data_dir=$VTRL_MED_ROOT/data/isic2019"

# Skip kvasir since it's already completed
datasets=("pathmnist" "isic2019")

echo "===================================================================="
echo "Ablation Study: Without Stage 2"
echo "===================================================================="
echo ""
echo "Testing: Is Stage 2 (clustering pre-training) necessary?"
echo "Method: Skip Stage 2, use k-means initialization in Stage 3"
echo ""
echo "Configuration:"
echo "  - Stage 1: Using existing checkpoints"
echo "  - Stage 2: SKIPPED"
echo "  - Stage 3 epochs: $STAGE3_EPOCHS"
echo "  - Reliability threshold: 0.6 (lower for k-means init)"
echo "  - Confidence threshold: 0.6 (lower for k-means init)"
echo ""
echo "Dataset order:"
echo "  1. PathMNIST (9 classes)"
echo "  2. ISIC2019 (8 classes)"
echo "  NOTE: Kvasir SKIPPED (already completed)"
echo ""
echo "Output directory: $ABLATION_DIR"
echo "===================================================================="
echo ""

mkdir -p "$ABLATION_DIR"

echo ""
echo "===================================================================="
echo "STAGE 3: Direct Fine-Tuning (Without Stage 2)"
echo "===================================================================="
echo ""

for dataset in "${datasets[@]}"; do
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Dataset: $dataset"
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

    stage1_checkpoint="$stage1_dir/stage1_sam_simclr_${dataset}_${IMG_SIZE}.pth"
    if [ ! -f "$stage1_checkpoint" ]; then
        echo "❌ ERROR: Stage 1 checkpoint not found: $stage1_checkpoint"
        continue
    fi

    echo "✓ Found Stage 1 checkpoint: $stage1_checkpoint"

    # Create fake Stage 2 directory (Stage 3 expects it to check if it exists)
    stage2_dir="$ABLATION_DIR/$dataset/stage2_fake"
    mkdir -p "$stage2_dir"

    stage3_dir="$ABLATION_DIR/$dataset/stage3"
    mkdir -p "$stage3_dir"

    stage3_checkpoint="$stage3_dir/stage3_sam_${dataset}_best.pth"
    if [ -f "$stage3_checkpoint" ]; then
        echo "✓ Stage 3 already completed"

        if [ -f "$stage3_dir/stage3_sam_results_${dataset}.json" ]; then
            echo ""
            echo "Stage 3 Results for $dataset:"
            python -c "
import json
with open('$stage3_dir/stage3_sam_results_${dataset}.json', 'r') as f:
    results = json.load(f)
    print(f\"  Has Stage 2: {results['has_stage2']}\")
    metrics = results['final_best_metrics']
    print(f\"  Visual:   ACC={metrics['vis_acc']:.4f}, NMI={metrics['vis_nmi']:.4f}, ARI={metrics['vis_ari']:.4f}\")
    print(f\"  Topology: ACC={metrics['topo_acc']:.4f}, NMI={metrics['topo_nmi']:.4f}, ARI={metrics['topo_ari']:.4f}\")
"
        fi
        continue
    fi

    echo ""
    echo "Running Stage 3 (Direct Fine-Tuning without Stage 2)..."
    echo "  Epochs: $STAGE3_EPOCHS"
    echo "  Stage 1: $stage1_checkpoint"
    echo "  Stage 2: SKIPPED (will use k-means initialization)"
    echo "  Reliability threshold: 0.6 (lower for k-means)"
    echo "  Confidence threshold: 0.6 (lower for k-means)"
    echo "  Output: $stage3_dir"
    echo ""

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
        --reliability_threshold 0.6 \
        --confidence_threshold 0.6 \
        --lambda_balance_start 2.0 \
        --lambda_balance_end 0.5 \
        --lambda_anneal_epochs 25 \
        --lambda_entropy 0.2 \
        --lambda_swap 0.1 \
        --phase_a_end 25 \
        --phase_b_end 40 \
        --tw_phase_a 0.5 \
        --tw_phase_b 0.2 \
        --tw_phase_c 0.05 \
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
            echo ""
            echo "Stage 3 Results for $dataset:"
            python -c "
import json
with open('$stage3_dir/stage3_sam_results_${dataset}.json', 'r') as f:
    results = json.load(f)
    print(f\"  Has Stage 2: {results['has_stage2']}\")
    metrics = results['final_best_metrics']
    print(f\"  Visual:   ACC={metrics['vis_acc']:.4f}, NMI={metrics['vis_nmi']:.4f}, ARI={metrics['vis_ari']:.4f}\")
    print(f\"  Topology: ACC={metrics['topo_acc']:.4f}, NMI={metrics['topo_nmi']:.4f}, ARI={metrics['topo_ari']:.4f}\")
"
        fi
        echo ""
    else
        echo "❌ Stage 3 failed for $dataset"
        exit 1
    fi
done

echo ""
echo "===================================================================="
echo "Ablation Complete: Without Stage 2"
echo "===================================================================="
echo ""
echo "Summary (All 3 Datasets):"

# Include kvasir in summary (already completed)
for dataset in "kvasir" "pathmnist" "isic2019"; do
    s3_results="$ABLATION_DIR/$dataset/stage3/stage3_sam_results_${dataset}.json"

    if [ -f "$s3_results" ]; then
        echo ""
        echo "Dataset: $dataset"
        python -c "
import json
with open('$s3_results', 'r') as f:
    results = json.load(f)
    print(f\"  Has Stage 2: {results['has_stage2']}\")
    metrics = results['final_best_metrics']
    print(f\"  Visual:   ACC={metrics['vis_acc']:.4f}, NMI={metrics['vis_nmi']:.4f}, ARI={metrics['vis_ari']:.4f}\")
    print(f\"  Topology: ACC={metrics['topo_acc']:.4f}, NMI={metrics['topo_nmi']:.4f}, ARI={metrics['topo_ari']:.4f}\")
" 2>/dev/null
    fi
done

echo ""
echo "===================================================================="
