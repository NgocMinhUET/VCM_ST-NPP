# ST-NPP: Comprehensive Guide for Research Paper Results

This guide provides detailed instructions for setting up, training, and evaluating the Spatio-Temporal Neural Preprocessing (ST-NPP) project to produce results suitable for an ISI Q1-level research paper.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation](#evaluation)
6. [Results Generation](#results-generation)
7. [Paper Integration](#paper-integration)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended for batch training)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: 500GB+ for datasets and models
- **CPU**: 4+ cores recommended

### Software Requirements
- **Operating System**: Ubuntu 18.04/20.04 (recommended), Windows 10/11 with WSL2, or macOS 12+
- **CUDA**: CUDA 11.3 or higher (for GPU acceleration)
- **cuDNN**: Compatible with CUDA version
- **Python**: 3.8 or 3.9
- **PyTorch**: 1.10.0 or higher
- **FFmpeg**: Recent version with libx265 (HEVC) support

## Environment Setup

### Creating Conda Environment
```bash
# Clone the repository
git clone https://github.com/username/st-npp.git
cd st-npp

# Create and activate conda environment using environment.yml
conda env create -f environment.yml
conda activate st-npp

# Alternatively, use pip with requirements.txt
# pip install -r requirements.txt
```

### Installing External Tools

#### FFmpeg Installation
**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```bash
# Run the FFmpeg setup script
.\scripts\setup_ffmpeg.bat
```

#### Verify Installation
```bash
# Verify environment
python scripts/verify_system.py
```

## Dataset Preparation

### COCO Video Dataset
```bash
# Create dataset directory
mkdir -p datasets/coco_video

# Download COCO Video dataset
python scripts/download_coco_video.py --output_dir datasets/coco_video

# Preprocess dataset
python scripts/preprocess_coco_video.py --dataset_dir datasets/coco_video --resize 224 224
```

### KITTI Semantic Dataset
```bash
# Create dataset directory
mkdir -p datasets/kitti_semantic

# Download KITTI Semantic dataset
python scripts/download_kitti_semantic.py --output_dir datasets/kitti_semantic

# Preprocess dataset
python scripts/preprocess_kitti_semantic.py --dataset_dir datasets/kitti_semantic --resize 224 224
```

### MOTChallenge Dataset
```bash
# Create dataset directory
mkdir -p datasets/MOTChallenge

# Download MOT16 dataset
python scripts/download_mot16.py --output_dir datasets/MOTChallenge

# Preprocess dataset
python scripts/prepare_mot16_dataset.py --dataset_dir datasets/MOTChallenge/MOT16 --output_dir datasets/MOTChallenge/processed
```

## Training Pipeline

The training process follows three main stages:

1. **Train Proxy Network**: Train the differentiable proxy for the HEVC codec
2. **Train ST-NPP and QAL**: Train the preprocessing network with quantization adaptation
3. **Joint Fine-tuning**: Fine-tune all components together

### 1. Train Proxy Network

```bash
python scripts/train_proxy.py \
  --dataset datasets/MOTChallenge/processed --use_mot_dataset \
  --batch_size 8 \
  --epochs 50 \
  --learning_rate 1e-4 \
  --qp_values 22,27,32,37 \
  --output_dir trained_models/proxy \
  --log_dir logs/proxy
```

**Key Parameters:**
- `--batch_size`: Adjust based on GPU memory
- `--epochs`: 50-100 epochs recommended
- `--qp_values`: QP values for training (standard: 22,27,32,37)
- `--learning_rate`: 1e-3 to 1e-5 depending on stability

**Output:**
- Models saved to: `trained_models/proxy/`
- Best model: `proxy_network_best_v*.pt`
- TensorBoard logs: `logs/proxy/`

### 2. Train ST-NPP and QAL

```bash
python scripts/train_stnpp.py \
  --dataset datasets/MOTChallenge/processed \
  --batch_size 8 \
  --epochs 50 \
  --stnpp_backbone resnet50 \
  --temporal_model 3dcnn \
  --learning_rate 1e-4 \
  --qp_values 22,27,32,37 \
  --output_dir trained_models \
  --log_dir logs
```

**Key Parameters:**
- `--stnpp_backbone`: Options: resnet18, resnet34, resnet50, efficientnet-b4
- `--temporal_model`: Options: 3dcnn, convlstm
- `--fusion_type`: Options: concatenation, attention

**Output:**
- ST-NPP model: `trained_models/stnpp/stnpp_best_v*.pt`
- QAL model: `trained_models/qal/qal_best_v*.pt`

### 3. Joint Fine-tuning

```bash
python scripts/train_joint.py \
  --stnpp_model trained_models/stnpp/stnpp_best_v*.pt \
  --qal_model trained_models/qal/qal_best_v*.pt \
  --proxy_model trained_models/proxy/proxy_network_best_v*.pt \
  --dataset datasets/MOTChallenge/processed \
  --batch_size 8 \
  --epochs 20 \
  --learning_rate 1e-5 \
  --lambda_distortion 1.0 \
  --lambda_rate 0.1 \
  --lambda_perception 0.01 \
  --output_dir trained_models/joint \
  --log_dir logs/joint
```

**Key Parameters:**
- `--lambda_distortion`: Weight for distortion loss (1.0 typical)
- `--lambda_rate`: Weight for rate loss (0.1-0.5 typical)
- `--lambda_perception`: Weight for perceptual loss (0.01-0.05 typical)

**Output:**
- Joint ST-NPP model: `trained_models/joint/stnpp_joint_best_v*.pt`
- Joint QAL model: `trained_models/joint/qal_joint_best_v*.pt`

### Resuming Training from Checkpoints

To resume training from a checkpoint:

```bash
python scripts/train_stnpp.py \
  --resume_stnpp trained_models/stnpp/stnpp_latest_v*.pt \
  --resume_qal trained_models/qal/qal_latest_v*.pt \
  --dataset datasets/MOTChallenge/processed \
  [other parameters]
```

### Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir=logs
```

Access TensorBoard at http://localhost:6006 to monitor:
- Training/validation losses
- Learning rates
- Sample visualizations
- Performance metrics

## Evaluation

### Comprehensive Evaluation

For a complete evaluation across all metrics and datasets:

```bash
python run_comprehensive_evaluation.py \
  --stnpp_model trained_models/joint/stnpp_joint_best_v*.pt \
  --qal_model trained_models/joint/qal_joint_best_v*.pt \
  --dataset_path datasets/MOTChallenge/MOT16 \
  --sequence MOT16-04 \
  --qp_range 22,27,32,37 \
  --output_dir results/comprehensive \
  --tasks detection,tracking,segmentation \
  --save_videos \
  --plot_curves
```

### Task-Specific Evaluation

#### Object Detection Evaluation

```bash
python scripts/evaluate_detection.py \
  --dataset datasets/coco_video/test \
  --stnpp_model trained_models/joint/stnpp_joint_best_v*.pt \
  --qal_model trained_models/joint/qal_joint_best_v*.pt \
  --qp_values 22,27,32,37 \
  --detector yolov5 \
  --output_dir results/detection
```

#### Semantic Segmentation Evaluation

```bash
python scripts/evaluate_segmentation.py \
  --dataset datasets/kitti_semantic/test \
  --stnpp_model trained_models/joint/stnpp_joint_best_v*.pt \
  --qal_model trained_models/joint/qal_joint_best_v*.pt \
  --qp_values 22,27,32,37 \
  --segmentation_model deeplabv3 \
  --output_dir results/segmentation
```

#### Object Tracking Evaluation

```bash
python scripts/evaluate_compression_tracking.py \
  --dataset_path datasets/MOTChallenge/MOT16 \
  --sequence MOT16-04 \
  --methods our,h264,h265,vp9 \
  --model_path trained_models/joint/stnpp_joint_best_v*.pt \
  --qal_model trained_models/joint/qal_joint_best_v*.pt \
  --qp_values 22 27 32 37 \
  --tracker sort \
  --output_dir results/tracking \
  --save_videos \
  --plot_curves
```

### Codec Comparison Evaluation

```bash
python compare_compression_methods.py \
  --input_video data/sample_videos/video.mp4 \
  --methods our,h264,h265,vp9 \
  --model_path trained_models/joint/stnpp_joint_best_v*.pt \
  --qal_model trained_models/joint/qal_joint_best_v*.pt \
  --h264_crf 18,23,28,33 \
  --h265_crf 18,23,28,33 \
  --vp9_crf 18,23,28,33 \
  --output_dir results/compression_comparison
```

## Results Generation

### Generate BD-Rate Analysis

BD-Rate measures the bitrate savings at equal quality.

```bash
python generate_report.py \
  --input_dir results/tracking \
  --output_file results/reports/bdrate_analysis.pdf \
  --metrics mota,idf1,mAP,mIoU
```

### Generate RD Curves

```bash
python scripts/plot_rd_curves.py \
  --results_dir results/tracking \
  --output_dir results/figures \
  --metrics mota,bpp \
  --methods our,h264,h265,vp9 \
  --title "Rate-Distortion Performance" \
  --format pdf,png \
  --dpi 300
```

### Generate Metric Comparison Tables

```bash
python scripts/generate_tables.py \
  --results_dir results/comprehensive \
  --output_dir results/tables \
  --format csv,latex
```

### Generate Visual Comparison Figures

```bash
python scripts/generate_visual_comparison.py \
  --input_dir results/tracking/visualization \
  --output_dir results/figures/visual_comparison \
  --sequence MOT16-04 \
  --frame_ids 15,30,45,60 \
  --methods our,h264,h265 \
  --qp 27
```

## Paper Integration

### Tables for Paper

Key tables for paper integration are located at:
- `results/tables/bdrate_summary.csv`: BD-Rate savings across methods
- `results/tables/performance_metrics.csv`: Task performance metrics
- `results/tables/ablation_study.csv`: Ablation study results

For LaTeX integration:
```bash
# Generate LaTeX tables
python scripts/generate_latex_tables.py --input_dir results/tables --output_dir paper/tables
```

### Figures for Paper

Key figures for paper integration:
- `results/figures/rd_curves_tracking.pdf`: Rate-Distortion curves
- `results/figures/rd_curves_detection.pdf`: Detection performance curves
- `results/figures/visual_comparison/*.pdf`: Visual comparison samples

### Sample Video Integration

For video demos accompanying the paper:
- `results/tracking/videos/comparison_sequence.mp4`: Side-by-side comparison
- `results/tracking/videos/our_method_qp27.mp4`: Our method result
- `results/tracking/videos/h265_crf27.mp4`: HEVC baseline

## Troubleshooting

### Common Issues and Solutions

#### CUDA Out of Memory
- Reduce batch size
- Use gradient accumulation
- Process smaller video segments

#### FFmpeg Issues
- Ensure FFmpeg is installed with libx265 support
- Check path is correctly set
- For Windows, use the provided setup script

#### Dataset Loading Problems
- Check dataset paths
- Ensure preprocessing completed successfully
- Verify frame extraction was successful

#### Training Instability
- Reduce learning rate
- Use gradient clipping
- Check normalization is applied correctly

#### Evaluation Errors
- Ensure models are loaded correctly
- Check QP values are valid
- Verify codec support in FFmpeg

### Performance Optimization

For better performance:
- Use SSD for dataset storage
- Increase num_workers for data loading
- Use mixed precision training (torch.cuda.amp)
- Consider distributed training for large datasets

For more detailed troubleshooting, refer to the [Project Wiki](https://github.com/username/st-npp/wiki/Troubleshooting). 