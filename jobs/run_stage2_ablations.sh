#!/bin/bash
#$ -q gpu@@coba-h100
#$ -l gpu_card=1
#$ -pe smp 16
#$ -m abe
#$ -N Stage2_Ablations

VTRL_MED_ROOT="${VTRL_MED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$VTRL_MED_ROOT/med_clustering.env"
cd "$VTRL_MED_ROOT"

# ==============================================================================
#  Stage 2 Ablation Study
# ==============================================================================
# Tests different pseudo-labeling methods:
#   1. w/o Stage 1 (pretrained SAM)
#   2. FixMatch (confidence thresholding)
#   3. SwAV (optimal transport)
#   4. SPICE (prototype + reliable filtering)
#   5. Ours - Confidence Only
#   6. Ours - Prototype Only
#   7. Ours - Full (confidence + prototype)
# ==============================================================================

set -e

# Configuration
MAIN_OUTPUT_DIR="$VTRL_MED_ROOT/checkpoints/ablation_study"
SAM_CHECKPOINT="$VTRL_MED_ROOT/weights/sam_vit_h_4b8939.pth"
STAGE1_BASE_DIR="$VTRL_MED_ROOT/checkpoints/sam_spice/stage1"

# Datasets to test
medmnist_datasets=(
    "pathmnist"
    "dermamnist"
    "octmnist"
)

# Ablation methods
methods=(
    "w/o_stage1"
    "fixmatch"
    "swav"
    "spice"
    "confidence_only"
    "prototype_only"
    "ours"
)

# ==============================================================================
#  Run Ablations
# ==============================================================================

for dataset in "${medmnist_datasets[@]}"; do
    echo ""
    echo "===================================================================="
    echo "Dataset: $dataset"
    echo "===================================================================="
    
    stage1_dir="$STAGE1_BASE_DIR/$dataset"
    
    # Check if Stage 1 exists (except for w/o_stage1)
    stage1_checkpoint="$stage1_dir/stage1_sam_simclr_${dataset}_224.pth"
    if [ ! -f "$stage1_checkpoint" ]; then
        echo "⚠️  Stage 1 checkpoint not found: $stage1_checkpoint"
        echo "   Please run Stage 1 first for non-w/o_stage1 methods"
        echo "   Skipping methods that require Stage 1..."
        # Only run w/o_stage1
        methods_to_run=("w/o_stage1")
    else
        echo "✓ Stage 1 checkpoint found"
        methods_to_run=("${methods[@]}")
    fi
    
    for method in "${methods_to_run[@]}"; do
        echo ""
        echo "--------------------------------------------------------------------"
        echo "Running: $method"
        echo "--------------------------------------------------------------------"
        
        output_dir="$MAIN_OUTPUT_DIR/stage2_${method}/$dataset"
        mkdir -p "$output_dir"
        
        # Check if already completed
        result_file="$output_dir/stage2_${method}_results_${dataset}_224.json"
        if [ -f "$result_file" ]; then
            echo "✓ Already completed: $result_file"
            echo "  Skipping..."
            continue
        fi
        
        # Run Stage 2 with specific method
        python scripts/stage2_unified.py \
            --dataset "$dataset" \
            --pseudo_label_method "$method" \
            --checkpoint_dir "$stage1_dir" \
            --output_dir "$output_dir" \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --sam_model_type vit_h \
            --img_size 224 \
            --epochs 15 \
            --batch_size 512 \
            --lr 6e-4 \
            --num_heads 10 \
            --eval_frequency 5 \
            --num_workers 16 \
            --top_k_ratio 0.5 \
            --fixmatch_threshold 0.95 \
            --spice_reliability_threshold 0.8 \
            --spice_num_neighbors 50 \
            --seed 42
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed: $method on $dataset"
        else
            echo "❌ Failed: $method on $dataset"
        fi
    done
done

# ==============================================================================
#  Summary
# ==============================================================================

echo ""
echo "===================================================================="
echo "Ablation Study Complete!"
echo "===================================================================="
echo ""
echo "Results saved in: $MAIN_OUTPUT_DIR/"
echo ""
echo "Methods tested:"
for method in "${methods[@]}"; do
    echo "  - $method"
done
echo ""
echo "To compare results, check the JSON files in each method's directory."
echo "===================================================================="
