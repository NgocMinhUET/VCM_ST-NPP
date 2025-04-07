# Spatio-Temporal Neural Preprocessing (ST-NPP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

A neural preprocessing framework for standard-compatible video coding in machine vision tasks. This framework implements neural preprocessing techniques to optimize video for both compression efficiency and downstream vision task performance.

## 📋 Project Overview

The ST-NPP framework consists of three main components:

1. **Spatio-Temporal Neural Pre-Processor (ST-NPP)**: Reduces spatial and temporal redundancy in video frames
2. **Quantization Adaptation Layer (QAL)**: Adapts the preprocessing based on codec QP values
3. **Proxy Network**: A differentiable approximation of standard codecs for end-to-end training

The system works with standard video codecs (HEVC/H.265, VVC) and improves machine vision tasks including:
- Object detection (mAP)
- Semantic segmentation (mIoU)
- Object tracking (MOTA/IDF1)

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- FFmpeg with HEVC/x265 support

### Quick Setup

```bash
# Clone repository
git clone https://github.com/username/st-npp.git
cd st-npp

# Create and activate conda environment
conda env create -f environment.yml
conda activate st-npp

# Verify installation
python scripts/verify_system.py
```

### Data Preparation

The framework supports three datasets:

```bash
# Prepare MOTChallenge dataset (tracking)
python scripts/download_mot16.py --output_dir datasets/MOTChallenge
python scripts/prepare_mot16_dataset.py --dataset_dir datasets/MOTChallenge/MOT16

# For COCO Video and KITTI datasets
# See PROJECT_GUIDE.md for detailed instructions
```

## 🏋️ Training

The training process follows a 3-stage pipeline:

### 1. Train Proxy Network

```bash
python scripts/train_proxy.py \
  --dataset datasets/MOTChallenge/processed \
  --batch_size 8 \
  --epochs 50 \
  --qp_values 22,27,32,37 \
  --output_dir trained_models/proxy
```

### 2. Train ST-NPP and QAL

```bash
python scripts/train_stnpp.py \
  --dataset datasets/MOTChallenge/processed \
  --batch_size 8 \
  --epochs 50 \
  --stnpp_backbone resnet50 \
  --temporal_model 3dcnn \
  --qp_values 22,27,32,37
```

### 3. Joint Fine-tuning

```bash
python scripts/train_joint.py \
  --stnpp_model trained_models/stnpp/stnpp_best_v*.pt \
  --qal_model trained_models/qal/qal_best_v*.pt \
  --proxy_model trained_models/proxy/proxy_network_best_v*.pt \
  --dataset datasets/MOTChallenge/processed
```

## 📊 Evaluation

### Comprehensive Evaluation

```bash
python run_comprehensive_evaluation.py \
  --stnpp_model trained_models/joint/stnpp_joint_best_v*.pt \
  --qal_model trained_models/joint/qal_joint_best_v*.pt \
  --dataset_path datasets/MOTChallenge/MOT16 \
  --sequence MOT16-04 \
  --qp_range 22,27,32,37 \
  --tasks detection,tracking,segmentation \
  --plot_curves
```

### Tracking Performance Evaluation

```bash
python scripts/evaluate_compression_tracking.py \
  --dataset_path datasets/MOTChallenge/MOT16 \
  --sequence MOT16-04 \
  --methods our,h264,h265,vp9 \
  --model_path trained_models/joint/stnpp_joint_best_v*.pt \
  --qp_values 22 27 32 37 \
  --tracker sort \
  --output_dir results/tracking
```

### Codec Comparison

```bash
python compare_compression_methods.py \
  --input_video data/sample_videos/video.mp4 \
  --methods our,h264,h265,vp9 \
  --model_path trained_models/joint/stnpp_joint_best_v*.pt \
  --qal_model trained_models/joint/qal_joint_best_v*.pt
```

## 📈 Results & Visualization

### Generate Results for Paper

```bash
# Generate BD-Rate analysis
python generate_report.py --input_dir results/tracking

# Plot RD curves
python scripts/plot_rd_curves.py --results_dir results/tracking

# Generate tables
python scripts/generate_tables.py --results_dir results/comprehensive
```

### Expected Outputs

- **BD-Rate Savings**: Tables with bitrate savings compared to standard codecs
- **RD Curves**: Rate-distortion curves showing performance vs. bitrate
- **Visual Comparisons**: Side-by-side visual comparisons of compression quality
- **Task Performance**: Metrics for detection, segmentation, and tracking

## 🛠️ Troubleshooting

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, use gradient accumulation |
| FFmpeg errors | Verify FFmpeg installation with libx265 support |
| Slow training | Use mixed precision, increase workers, use SSD storage |
| Dataset loading errors | Check paths and preprocessing steps |

## 📚 Directory Structure

```
.
├── datasets/                # Dataset storage
├── models/                  # Model definitions
├── scripts/                 # Training and evaluation scripts
├── src/                     # Core source code
│   ├── models/              # Model implementations
│   └── evaluation.py        # Evaluation metrics
├── trained_models/          # Saved model weights
├── results/                 # Evaluation results
│   ├── figures/             # Plots and visualizations
│   └── tables/              # Metric tables
├── environment.yml          # Conda environment
└── requirements.txt         # Python dependencies
```

## 📖 Documentation

- [PROJECT_GUIDE.md](PROJECT_GUIDE.md): Comprehensive guide for research paper results
- [scripts/README.md](scripts/README.md): Detailed documentation of all scripts
- [MODELS.md](models/README.md): Technical details of model architectures
- [EVALUATION.md](docs/EVALUATION.md): Explanation of metrics and evaluation protocols

## 📄 Citation

```
@article{author2023stnpp,
  title={Spatio-Temporal Neural Preprocessing for Standard-Compatible Video Coding in Machine Vision Tasks},
  author={Author, A. and Researcher, B.},
  journal={Journal of Visual Communication and Image Representation},
  year={2023}
}
```

## 📃 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 