# Scripts Documentation

This directory contains scripts for setting up the environment, running the video compression pipeline, and evaluating the results.

## Setup Scripts

### FFmpeg Setup

FFmpeg is a critical dependency for this project, used for video processing and comparing our compression method with standard video codecs (H.264, H.265, VP9, AV1).

- **setup_ffmpeg.bat** - Windows script to automatically download and install FFmpeg
- **setup_ffmpeg.sh** - Linux/macOS script to automatically download and install FFmpeg

Usage:
```bash
# Windows
scripts\setup_ffmpeg.bat

# Linux/macOS
bash scripts/setup_ffmpeg.sh
```

### Sample Videos

The project requires video samples to test compression. Use this script to download a set of sample videos:

- **download_sample_videos.py** - Downloads sample videos for testing

Usage:
```bash
# Download basic sample videos
python scripts/download_sample_videos.py

# Download all datasets including MOT16 and DAVIS (larger download)
python scripts/download_sample_videos.py --dataset all

# Download to a specific directory
python scripts/download_sample_videos.py --output_dir path/to/videos
```

### System Verification

Use this script to verify your system setup:

- **verify_system.py** - Comprehensive check of system requirements and dependencies

Usage:
```bash
# Basic verification
python scripts/verify_system.py

# Verify and attempt to fix issues
python scripts/verify_system.py --fix

# Run a quick compression test
python scripts/verify_system.py --run_test
```

## Training Scripts

These scripts are used to train the autoencoder models:

- **train_proxy.py** - Trains the proxy network for compression
- **train_stnpp.py** - Trains the spatio-temporal network with plus-plus enhancements
- **train_mot_simplified.py** - Trains a simplified model specifically for MOT (Multiple Object Tracking) data
- **train_proxy_mot.py** - Train proxy network using MOT dataset
- **train_improved_autoencoder.py** - Train the improved autoencoder with vector quantization

## Evaluation Scripts

Use these scripts to evaluate the compression performance:

- **evaluate.py** - Main evaluation script for various metrics
- **evaluate_mot_tracking.py** - Evaluate tracking performance on MOT dataset
- **evaluate_compression_tracking.py** - Evaluate how different compression methods affect tracking performance
- **evaluate_improved_compression.py** - Evaluate the improved autoencoder's compression performance
- **evaluate_autoencoder.py** - Evaluate basic autoencoder metrics
- **compare_compression_methods.py** - Compare our method with standard codecs (H.264, H.265, VP9)

Usage:
```bash
# Compare our method with standard codecs
python scripts/compare_compression_methods.py --sequence_path data/sample_videos/pedestrians

# Evaluate tracking performance with different compression methods
python scripts/evaluate_compression_tracking.py --input_video data/sample_videos/pedestrians.mp4

# Quick test with sample data (no actual compression)
python scripts/compare_compression_methods.py --sequence_path data/sample_videos/pedestrians --use_sample_data
```

## Example Workflows

### Complete Setup and Testing

```bash
# 1. Setup environment
# Windows
scripts\setup_ffmpeg.bat
# Linux/macOS
bash scripts/setup_ffmpeg.sh

# 2. Download sample videos
python scripts/download_sample_videos.py

# 3. Verify system
python scripts/verify_system.py --fix

# 4. Run comparison with sample data
python scripts/compare_compression_methods.py --sequence_path data/sample_videos/pedestrians --use_sample_data

# 5. Train model (if not already trained)
python scripts/train_improved_autoencoder.py --epochs 10 --batch_size 8

# 6. Evaluate compression
python scripts/evaluate_improved_compression.py --input_video data/sample_videos/pedestrians.mp4

# 7. Compare with standard codecs
python scripts/compare_compression_methods.py --sequence_path data/sample_videos/pedestrians

# 8. Evaluate tracking performance
python scripts/evaluate_compression_tracking.py --input_video data/sample_videos/pedestrians.mp4
```

### Quick Sample Evaluation

For a quick test without actual compression (using sample data):

```bash
python scripts/compare_compression_methods.py --sequence_path dummy --use_sample_data
```

This will generate sample results and plots to verify the evaluation pipeline works correctly.

## Common Issues and Solutions

1. **FFmpeg not found**: Run the appropriate setup script for your platform.
2. **CUDA not available**: The code will fall back to CPU, but this will be much slower.
3. **Model file not found**: Ensure you've trained the model or are pointing to the correct path.
4. **Sample videos not found**: Run the download_sample_videos.py script to get test videos.

For more detailed troubleshooting, run the verify_system.py script with the --verbose flag. 