# Video Compression for Machine Vision

A neural preprocessing pipeline optimizing video compression for machine vision tasks. This project implements a complete system that improves compression efficiency while maintaining or enhancing the performance of downstream machine vision tasks.

## Project Overview

This project addresses the challenge of optimizing video compression specifically for machine vision applications. Traditional codecs like HEVC are designed for human perception, but machine vision algorithms have different requirements for visual information. Our system introduces neural preprocessing modules that adapt video content before compression to preserve features critical for computer vision tasks.

### Key Features

- **Spatio-Temporal Neural Preprocessing**: Reduces redundancy in video content while preserving machine vision-relevant features
- **Quantization Adaptation Layer**: Adapts feature maps based on codec quantization parameters
- **Differentiable Proxy Network**: Approximates the non-differentiable codec behavior for end-to-end training
- **Multi-task Evaluation**: Comprehensive evaluation on detection, segmentation, and tracking tasks

## Architecture

The pipeline consists of four main modules:

1. **Spatio-Temporal Neural Preprocessing (ST-NPP)**: A dual-branch network with spatial and temporal paths that reduces redundancy while preserving task-relevant features.
2. **Quantization Adaptation Layer (QAL)**: Adapts the output of ST-NPP based on the quantization parameter of the codec.
3. **HEVC Codec**: Standard video compression using the HEVC/H.265 codec.
4. **Differentiable Proxy Network**: A fully differentiable approximation of the codec that enables end-to-end training.

```
Input Video → ST-NPP → QAL → HEVC Codec → Compressed Bitstream
                                   ↓
                            Downstream Tasks
                        (Detection, Segmentation, Tracking)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8 or higher
- CUDA-compatible GPU (recommended)
- FFmpeg with HEVC/H.265 support

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/video-compression-mv.git
   cd video-compression-mv
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install FFmpeg (if not already installed):
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

4. Download pre-trained models (optional):
   ```
   ./scripts/download_models.sh
   ```

## Usage

### Training

#### Training the Proxy Network

```bash
python scripts/train_proxy.py --dataset_path /path/to/videos \
                              --batch_size 4 \
                              --epochs 100 \
                              --output_dir trained_models
```

#### Training the ST-NPP with QAL

```bash
python scripts/train_stnpp.py --dataset_path /path/to/videos \
                            --proxy_model_path trained_models/proxy_network.pt \
                            --batch_size 4 \
                            --epochs 150 \
                            --output_dir trained_models
```

### Evaluation

```bash
python scripts/evaluate.py --video_path /path/to/test/videos \
                         --stnpp_model_path trained_models/stnpp_final.pt \
                         --qal_model_path trained_models/qal_final.pt \
                         --evaluate_detection --evaluate_segmentation --evaluate_tracking
```

#### Using MOT16 Dataset for Tracking Evaluation

For tracking evaluation using the MOT16 dataset:

```bash
python scripts/evaluate.py --video_path /path/to/test/videos \
                         --stnpp_model_path trained_models/stnpp_final.pt \
                         --qal_model_path trained_models/qal_final.pt \
                         --evaluate_tracking \
                         --mot_dataset_path D:/NCS/propose/dataset/MOT16
```

This will use the ground truth annotations from the MOT16 dataset to evaluate tracking performance on both the original videos and the compressed videos.

## Model Details

### Spatio-Temporal Neural Preprocessing (ST-NPP)

The ST-NPP module consists of:
- **Spatial Branch**: Pre-trained CNN backbone (ResNet50, ResNet34, or EfficientNet) to extract spatial features
- **Temporal Branch**: Either 3D CNN or ConvLSTM to model temporal dependencies
- **Fusion Module**: Combines spatial and temporal features through concatenation or attention mechanisms

### Quantization Adaptation Layer (QAL)

The QAL adapts feature maps based on the QP value of the codec:
- **Standard QAL**: Uses an MLP to predict scaling factors based on QP
- **Conditional QAL**: Uses both QP and feature context for adaptation
- **Pixelwise QAL**: Applies different scaling factors to different spatial positions

### Differentiable Proxy Network

A 3D CNN-based autoencoder that approximates the rate-distortion behavior of the HEVC codec:
- **Encoder**: Compresses input to a latent representation
- **Decoder**: Reconstructs from the latent representation
- **Rate-Distortion Loss**: Approximates bitrate and distortion for optimization

## Dataset Support

The system supports evaluation on:
- **Detection**: COCO dataset
- **Segmentation**: KITTI dataset
- **Tracking**: MOTChallenge dataset

## Results

Our experimental evaluation shows:
- **BD-Rate**: XX% reduction compared to standard HEVC
- **Detection Performance**: X.X% improvement in mAP at the same bitrate
- **Segmentation Performance**: X.X% improvement in mIoU
- **Tracking Performance**: X.X% improvement in MOTA and IDF1 scores

Detailed evaluation results and visualizations are available in the `evaluation_results` directory after running the evaluation script.

## Project Structure

```
.
├── models/                  # Model implementations
│   ├── stnpp.py            # Spatio-Temporal Neural Preprocessing
│   ├── qal.py              # Quantization Adaptation Layer
│   └── proxy_network.py    # Differentiable Proxy Network
├── scripts/                 # Training and evaluation scripts
│   ├── train_proxy.py      # Training script for proxy network
│   ├── train_stnpp.py      # Training script for ST-NPP with QAL
│   └── evaluate.py         # Evaluation script
├── datasets/                # Dataset loaders and utilities
│   ├── video/              # Video datasets
│   ├── detection/          # Detection datasets
│   ├── segmentation/       # Segmentation datasets
│   └── tracking/           # Tracking datasets
├── trained_models/          # Pre-trained model weights
├── evaluation_results/      # Evaluation outputs and visualizations
└── requirements.txt         # Dependencies
```

## Acknowledgments

- This work uses the HEVC reference software
- Portions of the evaluation code are based on standard benchmarks for detection, segmentation, and tracking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite our work:

```
@article{author2023video,
  title={Video Compression for Machine Vision},
  author={Author, A. and Author, B.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2023}
}
```

## Troubleshooting

### No Video Files Found Error

If you see an error like this:
```
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

or

```
ValueError: No video files found in /path/to/dataset. Make sure the path exists and contains video files with extensions: ['.mp4', '.avi', '.mov', '.mkv']
```

This means the system couldn't find any video files in the specified directory. To fix this:

1. Make sure the `--dataset_path` argument points to a directory that contains video files
2. Check that your video files have one of the supported extensions: .mp4, .avi, .mov, or .mkv
3. Place your video files in the `data/sample_videos` directory and specify this path:
   ```
   python scripts/train_proxy.py --dataset_path data/sample_videos
   ```

### Video Files Found But No Valid Sequences

If you see this error:
```
ValueError: No valid sequences could be created from the videos in /path/to/dataset. Check that the videos have at least 16 frames and are readable.
```

This means the script found video files but couldn't extract enough frames to create sequences. To fix this:

1. Ensure your videos are at least 16 frames long (for default settings)
2. Check that the video files are not corrupt and can be opened with a media player
3. If your videos are very short, you can reduce the `--time_steps` parameter:
   ```
   python scripts/train_proxy.py --dataset_path data/sample_videos --time_steps 8
   ```

## Video Compression with Autoencoder

This project also includes a video compression tool using the trained autoencoder model. The tool can compress a video file, reconstruct it, and generate a comparison video showing the original and reconstructed frames side by side.

### Usage

```bash
python video_compression.py --input_video <path_to_video_file> [options]
```

### Parameters

- `--input_video`: (Required) Path to the input video file you want to compress
- `--model_path`: Path to the trained autoencoder model (default: "trained_models/mot16_model/autoencoder_best.pt")
- `--time_steps`: Number of frames to process at once (default: 16)
- `--output_dir`: Directory to save compressed and reconstructed videos (default: "results/compressed_videos")

### Example

```bash
python video_compression.py --input_video data/sample_video.mp4 --output_dir results/my_compressed_videos
```

### Output

The script generates the following outputs:
1. A reconstructed video file
2. A comparison video showing original and reconstructed frames side by side
3. A text file with compression statistics (compression ratio, file sizes, etc.) 