# Training Task-Aware Video Compression with MOT16 Dataset

This document explains how to train the task-aware video compression model using the MOT16 dataset for multi-object tracking tasks.

## Overview

The MOT16 (Multiple Object Tracking) dataset is a benchmark dataset used for evaluating multi-object tracking algorithms. It consists of 14 sequences (7 for training and 7 for testing) with annotated bounding boxes for tracked objects.

The task-aware video compression model has been enhanced to support tracking tasks using the MOT16 dataset. This implementation includes:

1. A data adapter to convert the MOT16 dataset to the format expected by the model
2. Enhanced loss functions with robust error handling
3. Improved validation and training procedures
4. Utilities for monitoring and visualization

## Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8 or higher
- CUDA-capable GPU (recommended)
- MOT16 dataset (download from [MOTChallenge](https://motchallenge.net/))

### Installation

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Download the MOT16 dataset and extract it to a local directory

## Dataset Preparation

The MOT16 dataset needs to be converted to the format expected by the task-aware video compression model. This can be done using the provided scripts:

```bash
python scripts/convert_mot16.py --mot_root /path/to/MOT16 --output_root /path/to/processed/dataset
```

This script will:
1. Parse the MOT16 dataset structure
2. Extract frame sequences of the specified length
3. Convert the annotations to YOLO format with tracking IDs
4. Organize the data into the expected directory structure for the task-aware model

## Training

To train the model with the MOT16 dataset, you have two options:

### Option 1: Using the Preparation Script

The simplest way is to use the preparation script that handles both dataset conversion and training:

```bash
python scripts/prepare_mot16_training.py \
    --mot_root /path/to/MOT16 \
    --output_root /path/to/processed/dataset \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --qp 30
```

### Option 2: Using the Training Script Directly

If you've already converted the dataset, you can use the training script directly:

```bash
python train.py \
    --dataset /path/to/processed/dataset \
    --task_type tracking \
    --seq_length 5 \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --qp 30
```

### Windows Batch File

For Windows users, a batch file is provided for convenience:

```
run_mot16_training.bat
```

This batch file will automatically run the preparation script with default parameters.

## Training Parameters

Key parameters for training:

- `--seq_length`: Number of frames in each sequence (default: 5)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--qp`: Quantization parameter (default: 30, higher = more compression)
- `--random_qp`: Use random QP values during training
- `--task_weight`: Weight for tracking task loss (default: 1.0)
- `--recon_weight`: Weight for reconstruction loss (default: 1.0)
- `--bitrate_weight`: Weight for bitrate loss (default: 0.1)

## Monitoring and Evaluation

The training process outputs progress and metrics to the console and logs them to TensorBoard. To view the TensorBoard logs:

```bash
tensorboard --logdir checkpoints/mot16/logs
```

This will show:
- Loss curves (task loss, reconstruction loss, bitrate loss)
- Quality metrics (PSNR, SSIM)
- Tracking metrics (MOTA - Multiple Object Tracking Accuracy)
- GPU memory usage

## Troubleshooting

### Common Issues

1. **Out of memory errors**: Reduce batch size or sequence length
2. **Tensor size mismatch**: The model includes robust error handling for tensor size mismatches, but ensure input resolutions are multiples of the model's block_size (default: 8)
3. **Poor tracking performance**: Try adjusting loss weights (increase task_weight) or reduce QP for higher quality

### Dataset Verification

To verify that the dataset was converted correctly:

```bash
python scripts/convert_mot16.py --mot_root /path/to/MOT16 --output_root /path/to/processed/dataset --verify
```

This will check that all sequences have the expected directory structure and matching frame/label files.

## References

- [MOT16 Dataset](https://motchallenge.net/data/MOT16/)
- [MOTChallenge Benchmarks](https://motchallenge.net/)
- [YOLO Format](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) 