#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Ablation_FixedWeights

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

set -e

# Configuration
SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE="vit_h"
IMG_SIZE=224
SEED=42

STAGE3_EPOCHS=50
STAGE1_BASE_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice/stage1"
STAGE2_BASE_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice/stage2"

declare -A dataset_configs
dataset_configs["kvasir"]="type=custom,num_classes=8,data_dir=$VTRL_MED_ROOT/data/kvasir-dataset"
dataset_configs["pathmnist"]="type=medmnist,num_classes=9,data_dir="
dataset_configs["isic2019"]="type=custom,num_classes=8,data_dir=$VTRL_MED_ROOT/data/isic2019"

datasets=("kvasir" "pathmnist" "isic2019")

# Array of fixed weights to test
declare -a FIXED_WEIGHTS=("0.5" "0.2")

echo "===================================================================="
echo "Ablation Study: Fixed Topology Weights"
echo "===================================================================="
echo ""
echo "Testing fixed topology weights: ${FIXED_WEIGHTS[@]}"
echo "Stage 3 epochs: $STAGE3_EPOCHS"
echo ""
echo "Experiments:"
echo "  1. Fixed Weight = 0.5"
echo "  2. Fixed Weight = 0.2"
echo ""
echo "===================================================================="
echo ""

# Loop through each fixed weight
for WEIGHT in "${FIXED_WEIGHTS[@]}"; do
    ABLATION_DIR="ablation_study/fixed_weight_${WEIGHT//.}"  # Remove decimal: 0.5 -> 05
    
    echo ""
    echo "===================================================================="
    echo "Running: Fixed Topology Weight = $WEIGHT"
    echo "===================================================================="
    echo ""
    echo "Output directory: $ABLATION_DIR"
    echo ""
    
    mkdir -p "$ABLATION_DIR"
    
    for dataset in "${datasets[@]}"; do
        echo ""
        echo "--------------------------------------------------------------------"
        echo "Dataset: $dataset - Fixed Weight $WEIGHT"
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
            stage2_dir="$VTRL_MED_ROOT/checkpoints/sam_spice_kvasir_isic/stage2/$dataset"
        else
            stage1_dir="$STAGE1_BASE_DIR/$dataset"
            stage2_dir="$STAGE2_BASE_DIR/$dataset"
        fi
        
        stage3_dir="$ABLATION_DIR/$dataset/stage3"
        mkdir -p "$stage3_dir"
        
        stage3_checkpoint="$stage3_dir/stage3_sam_${dataset}_best.pth"
        if [ -f "$stage3_checkpoint" ]; then
            echo "✓ Stage 3 already completed"
            
            if [ -f "$stage3_dir/stage3_sam_results_${dataset}.json" ]; then
                echo ""
                echo "Existing results:"
                python -c "
import json
with open('$stage3_dir/stage3_sam_results_${dataset}.json', 'r') as f:
    results = json.load(f)
    metrics = results['final_best_metrics']
    print(f\"  Visual:   ACC={metrics['vis_acc']:.4f}, NMI={metrics['vis_nmi']:.4f}, ARI={metrics['vis_ari']:.4f}\")
    print(f\"  Topology: ACC={metrics['topo_acc']:.4f}, NMI={metrics['topo_nmi']:.4f}, ARI={metrics['topo_ari']:.4f}\")
"
            fi
            continue
        fi
        
        echo ""
        echo "Running Stage 3 with fixed weight = $WEIGHT..."
        
        # All phase weights set to the same value
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
            --phase_a_end 25 \
            --phase_b_end 40 \
            --tw_phase_a $WEIGHT \
            --tw_phase_b $WEIGHT \
            --tw_phase_c $WEIGHT \
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
                echo "Results:"
                python -c "
import json
with open('$stage3_dir/stage3_sam_results_${dataset}.json', 'r') as f:
    results = json.load(f)
    metrics = results['final_best_metrics']
    print(f\"  Visual:   ACC={metrics['vis_acc']:.4f}, NMI={metrics['vis_nmi']:.4f}, ARI={metrics['vis_ari']:.4f}\")
    print(f\"  Topology: ACC={metrics['topo_acc']:.4f}, NMI={metrics['topo_nmi']:.4f}, ARI={metrics['topo_ari']:.4f}\")
"
            fi
            echo ""
        else
            echo "❌ Stage 3 failed for $dataset with weight $WEIGHT"
            exit 1
        fi
    done
    
    echo ""
    echo "===================================================================="
    echo "Summary for Fixed Weight = $WEIGHT"
    echo "===================================================================="
    echo ""
    
    for dataset in "${datasets[@]}"; do
        s3_results="$ABLATION_DIR/$dataset/stage3/stage3_sam_results_${dataset}.json"
        
        if [ -f "$s3_results" ]; then
            echo "Dataset: $dataset"
            python -c "
import json
with open('$s3_results', 'r') as f:
    results = json.load(f)
    metrics = results['final_best_metrics']
    print(f\"  Visual:   ACC={metrics['vis_acc']:.4f}, NMI={metrics['vis_nmi']:.4f}, ARI={metrics['vis_ari']:.4f}\")
    print(f\"  Topology: ACC={metrics['topo_acc']:.4f}, NMI={metrics['topo_nmi']:.4f}, ARI={metrics['topo_ari']:.4f}\")
    print()
" 2>/dev/null
        fi
    done
    
    echo ""
done

# Final comparison
echo ""
echo "===================================================================="
echo "FINAL COMPARISON: All Fixed Weights"
echo "===================================================================="
echo ""

for WEIGHT in "${FIXED_WEIGHTS[@]}"; do
    ABLATION_DIR="ablation_study/fixed_weight_${WEIGHT//.}"
    echo "Fixed Weight = $WEIGHT:"
    echo "-----------------------------------"
    
    for dataset in "${datasets[@]}"; do
        s3_results="$ABLATION_DIR/$dataset/stage3/stage3_sam_results_${dataset}.json"
        
        if [ -f "$s3_results" ]; then
            python -c "
import json
with open('$s3_results', 'r') as f:
    results = json.load(f)
    metrics = results['final_best_metrics']
    print(f\"  {dataset:12s}: Visual ACC={metrics['vis_acc']:.4f}, Topo ACC={metrics['topo_acc']:.4f}\")
" 2>/dev/null
        fi
    done
    echo ""
done

echo "===================================================================="
echo "Ablation Complete: Fixed Topology Weights"
echo "===================================================================="
