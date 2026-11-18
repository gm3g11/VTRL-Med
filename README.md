# VTRL-Med

**SAM + SimCLR topology-aware pseudo-labeling for deep clustering of medical images.**

VTRL-Med clusters medical images without labels in three stages:

| Stage | Script | What it does |
|---|---|---|
| 1 | `scripts/stage1_unified.py` | Adapts the SAM ViT image encoder with SimCLR (NT-Xent loss); only the last `--unfrozen_blocks` transformer blocks are trained. |
| 2 | `scripts/stage2_unified.py` | Trains multiple clustering heads on Stage 1 features with pseudo-labels and keeps the best head. Supports our method and ablations / baselines (`confidence_only`, `prototype_only`, `fixmatch`, `flexmatch`, `spice`, `swav`, `wo_stage1`). |
| 3 | `scripts/stage3_unified.py` | Jointly fine-tunes the visual backbone and a topology branch (persistent homology computed on the fly with `cripser`), with a phase-scheduled topology loss weight. `stage3_unified_notopo.py` is the visual-only variant. |

Supported datasets: the 2D [MedMNIST](https://medmnist.com/) datasets (e.g. PathMNIST, DermaMNIST, BloodMNIST, OCTMNIST, ...), Kvasir, and ISIC 2019.

## Repository structure

```
VTRL-Med/
├── scripts/            # Main pipeline: Stage 1 / 2 / 3 (+ pseudo-label diagnostics)
├── vtrlmed/            # Package: dataset loaders (MedMNIST, Kvasir, ISIC 2019), topology utils
├── topocl/             # Topology contrastive-learning modules (ROI pre-computation, Stage 2/3 variants)
├── evaluation/         # Clustering metrics (ACC/NMI/ARI), k-means backbone comparison, linear probe, statistics
├── analysis/           # Feature-space and topology analyses
├── jobs/               # SGE job scripts used for the paper experiments
│   └── ablations/      # Ablation study job scripts
├── med_clustering.env  # Environment setup sourced by the job scripts
├── requirements.txt
└── setup.py
```

## Installation

Tested with Python 3.8, PyTorch 2.4.1 and CUDA 12.1.

```bash
git clone <this-repo-url> VTRL-Med
cd VTRL-Med

conda create -n med_clustering python=3.8 -y
conda activate med_clustering

# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install -e .
```

`evaluation/evaluate_sam2_stage1.py` additionally needs [SAM 2](https://github.com/facebookresearch/sam2), either installed or cloned into `tools/segment_anything_2`.

## Data and weights

All scripts are run from the repository root and expect the following layout (these directories are git-ignored):

```
data/
├── kvasir-dataset/              # Kvasir v2: one folder per class
│   ├── dyed-lifted-polyps/*.jpg
│   └── ...
└── isic2019/                    # ISIC 2019: one folder per class, under train/ and test/
    ├── train/<class>/*.jpg
    └── test/<class>/*.jpg
weights/
└── sam_vit_h_4b8939.pth
```

- **MedMNIST** datasets are downloaded automatically by the `medmnist` package.
- **Kvasir** and **ISIC 2019** must be downloaded manually and arranged as above. Both are pooled and re-split 85/15 (train/test) with a fixed seed.
- **SAM ViT-H** checkpoint: [`sam_vit_h_4b8939.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) from the [Segment Anything](https://github.com/facebookresearch/segment-anything) repository.

## Usage

Example for PathMNIST (replace `--dataset` / add `--data_dir data/kvasir-dataset` for Kvasir, etc.):

```bash
# Stage 1: SAM + SimCLR
python scripts/stage1_unified.py --dataset pathmnist \
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \
    --epochs 200 --batch_size 512 --lr 5e-4 --unfrozen_blocks 4 --use_amp \
    --output_dir checkpoints/stage1/pathmnist

# Stage 2: clustering heads with pseudo-labels
python scripts/stage2_unified.py --dataset pathmnist --pseudo_label_method ours \
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \
    --checkpoint_dir checkpoints/stage1/pathmnist \
    --output_dir checkpoints/stage2/pathmnist

# Stage 3: joint fine-tuning with topology
python scripts/stage3_unified.py --dataset pathmnist \
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \
    --checkpoint_dir checkpoints/stage1/pathmnist \
    --stage2_dir checkpoints/stage2/pathmnist \
    --output_dir checkpoints/stage3/pathmnist
```

Checkpoints are saved as `stage1_sam_simclr_<dataset>_<img_size>.pth`, `stage2_<method>_<dataset>_<img_size>.pth` and `stage3_sam_<dataset>_best.pth`. Stage 3 loads `stage2_ours_<dataset>_<img_size>.pth` and falls back to k-means initialization if it is missing. Run any script with `--help` for all options.

### Job scripts

The scripts in `jobs/` reproduce the paper experiments on an SGE cluster. Before using them:

1. Edit the `#$` scheduler headers (queue, GPU, cores) for your cluster.
2. Edit `med_clustering.env` (conda environment name, `module load` lines) for your system.
3. Tell the job where the repository is. When run with `bash jobs/<script>.sh` this is detected automatically; under SGE pass it explicitly:

```bash
qsub -v VTRL_MED_ROOT=$PWD jobs/run_stage1_unified.sh
```

### Evaluation and analysis

Several scripts in `evaluation/` and `analysis/` import from `scripts/` (e.g. `from stage3_unified import ...`), so put both the repository root and `scripts/` on `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD:$PWD/scripts:$PYTHONPATH
python evaluation/evaluate_stage2_clustering.py --help
```

## Citation

If you find this code useful, please cite:

```bibtex
@article{vtrlmed,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

## Acknowledgements

This code builds on [Segment Anything](https://github.com/facebookresearch/segment-anything), [SPICE](https://github.com/niuchuangnn/SPICE), [MedMNIST](https://github.com/MedMNIST/MedMNIST) and [CubicalRipser](https://github.com/shizuo-kaji/CubicalRipser_3dim).

## License

This project is released under the [MIT License](LICENSE).
