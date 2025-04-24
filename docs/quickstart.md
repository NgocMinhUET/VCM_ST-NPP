# Quick Start Guide

This guide will help you get started with the Task-Aware Video Preprocessing system quickly.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/task-aware-video-preprocessing.git
   cd task-aware-video-preprocessing
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg:**
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
   - MacOS: `brew install ffmpeg`
   - Linux: `apt-get install ffmpeg`

## Running Pre-trained Models

We provide pre-trained models for each task (detection, segmentation, tracking).

### Object Detection

```bash
# Download pre-trained model
mkdir -p checkpoints/detection
# Download model to checkpoints/detection/model.pth

# Run inference on a video
python evaluate.py --model_path checkpoints/detection/model.pth --task detection --video_path path/to/video.mp4 --output_dir results/detection
```

### Semantic Segmentation

```bash
# Download pre-trained model
mkdir -p checkpoints/segmentation
# Download model to checkpoints/segmentation/model.pth

# Run inference on a video
python evaluate.py --model_path checkpoints/segmentation/model.pth --task segmentation --video_path path/to/video.mp4 --output_dir results/segmentation
```

### Object Tracking

```bash
# Download pre-trained model
mkdir -p checkpoints/tracking
# Download model to checkpoints/tracking/model.pth

# Run inference on a video
python evaluate.py --model_path checkpoints/tracking/model.pth --task tracking --video_path path/to/video.mp4 --output_dir results/tracking
```

## Training Your Own Models

### 1. Prepare Dataset

For this example, we'll use the MOT16 dataset for tracking:

```bash
# Download and extract MOT16 dataset
mkdir -p data/MOT16
cd data/MOT16
wget https://motchallenge.net/data/MOT16.zip
unzip MOT16.zip
cd ../../

# Preprocess the dataset
python utils/preprocess_mot16.py --input_path data/MOT16 --output_path data/processed/MOT16
```

### 2. Train Proxy Codec First

```bash
# Train the proxy codec
python train.py --mode proxy --config configs/proxy_config.yaml --data_path data/train_videos --output_dir results/proxy
```

### 3. Train End-to-End System

```bash
# Train tracking model end-to-end
python train.py --mode end2end --config configs/tracking_config.yaml --proxy_weights results/proxy/best_model.pth --task tracking --data_path data/processed/MOT16 --output_dir results/tracking
```

### 4. Evaluate Your Model

```bash
# Evaluate on test set
python evaluate.py --model_path results/tracking/best_model.pth --task tracking --data_path data/processed/MOT16/test --output_dir eval_results/tracking
```

## Comparing with Baselines

Run a comprehensive evaluation comparing our method with codec-only baselines:

```bash
./run_comprehensive_evaluation.sh
```

This will generate CSV files and plots in the `comparison_results` directory.

## Visualizing Results

To generate visualizations for a specific model:

```bash
python visualize.py --model_path results/tracking/best_model.pth --task tracking --video_path path/to/video.mp4 --output_dir visualizations/tracking
```

## Next Steps

- Check [Configuration Guide](configuration_guide.md) for detailed parameter explanations
- See [MOT16 Guide](MOT16_guide.md) for working with the tracking dataset
- Explore the codebase to understand model implementations:
  - `models/st_npp.py`: Spatio-Temporal Neural Preprocessing
  - `models/qal.py`: Quantization Adaptation Layer
  - `models/task_networks/detector.py`: Object detection network
  - `models/task_networks/segmenter.py`: Semantic segmentation network
  - `models/task_networks/tracker.py`: Object tracking network 