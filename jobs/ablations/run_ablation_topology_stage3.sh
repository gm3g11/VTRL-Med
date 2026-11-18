#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Ablation_Phases

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
STAGE2_BASE_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice/stage2"  # Fixed: removed duplicate "stage2"

declare -A dataset_configs
dataset_configs["kvasir"]="type=custom,num_classes=8,data_dir=$VTRL_MED_ROOT/data/kvasir-dataset"
dataset_configs["pathmnist"]="type=medmnist,num_classes=9,data_dir="
dataset_configs["isic2019"]="type=custom,num_classes=8,data_dir=$VTRL_MED_ROOT/data/isic2019"

datasets=("kvasir" "pathmnist" "isic2019")

echo "===================================================================="
echo "Ablation Study: Phase Transitions"
echo "===================================================================="
echo ""
echo "Testing different phase configurations:"
echo "  1. w/o Phase A: Skip topology learning, start at Phase B"
echo "  2. w/o Phase B: Skip mutual refinement, jump A→C"
echo "  3. w/o Phase C: Skip visual dominance, stay in Phase B"
echo ""
echo "Stage 3 epochs: $STAGE3_EPOCHS"
echo "===================================================================="
echo ""

# Define phase configurations
# Format: "name|dir_suffix|phase_a_end|phase_b_end|description"
declare -a PHASE_CONFIGS=(
    "w/o Phase A|no_phase_a|0|15|Phase B (0-15) → Phase C (15-50)"
    "w/o Phase B|no_phase_b|25|25|Phase A (0-25) → Phase C (25-50)"
    "w/o Phase C|no_phase_c|25|50|Phase A (0-25) → Phase B (25-50)"
)

# Loop through each phase configuration
for config in "${PHASE_CONFIGS[@]}"; do
    IFS='|' read -ra CONFIG_PARTS <<< "$config"
    PHASE_NAME="${CONFIG_PARTS[0]}"
    DIR_SUFFIX="${CONFIG_PARTS[1]}"
    PHASE_A_END="${CONFIG_PARTS[2]}"
    PHASE_B_END="${CONFIG_PARTS[3]}"
    PHASE_DESC="${CONFIG_PARTS[4]}"
    
    ABLATION_DIR="ablation_study/$DIR_SUFFIX"
    
    echo ""
    echo "===================================================================="
    echo "Running: $PHASE_NAME"
    echo "===================================================================="
    echo ""
    echo "Configuration: $PHASE_DESC"
    echo "Output directory: $ABLATION_DIR"
    echo ""
    
    mkdir -p "$ABLATION_DIR"
    
    for dataset in "${datasets[@]}"; do
        echo ""
        echo "--------------------------------------------------------------------"
        echo "Dataset: $dataset - $PHASE_NAME"
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
        echo "Running Stage 3 ($PHASE_NAME)..."
        echo "  Phase A end: $PHASE_A_END"
        echo "  Phase B end: $PHASE_B_END"
        
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
            --phase_a_end $PHASE_A_END \
            --phase_b_end $PHASE_B_END \
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
            echo "❌ Stage 3 failed for $dataset with $PHASE_NAME"
            exit 1
        fi
    done
    
    echo ""
    echo "===================================================================="
    echo "Summary for $PHASE_NAME"
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
echo "FINAL COMPARISON: Phase Ablations"
echo "===================================================================="
echo ""

for config in "${PHASE_CONFIGS[@]}"; do
    IFS='|' read -ra CONFIG_PARTS <<< "$config"
    PHASE_NAME="${CONFIG_PARTS[0]}"
    DIR_SUFFIX="${CONFIG_PARTS[1]}"
    ABLATION_DIR="ablation_study/$DIR_SUFFIX"
    
    echo "$PHASE_NAME:"
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
echo "Phase Configurations Tested:"
echo "-----------------------------------"
echo "  w/o Phase A: Epochs 0-15 (Phase B), 15-50 (Phase C)"
echo "  w/o Phase B: Epochs 0-25 (Phase A), 25-50 (Phase C)"
echo "  w/o Phase C: Epochs 0-25 (Phase A), 25-50 (Phase B)"
echo "===================================================================="
echo ""
echo "Ablation Complete: Phase Transitions"
echo "===================================================================="
