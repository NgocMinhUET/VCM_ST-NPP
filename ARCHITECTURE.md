# VCM-ST-NPP Architecture Overview

## System Architecture

The Task-Aware Video Preprocessing (VCM-ST-NPP) system consists of several interconnected components that work together to optimize video compression for machine vision tasks. The architecture follows a modular design where each component addresses a specific aspect of the task-aware compression pipeline.

```
                                 ┌───────────────┐
                                 │ Raw Video     │
                                 └───────┬───────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────┐
│                       ST-NPP Module                           │
│                                                               │
│  ┌─────────────────┐    ┌────────────────┐    ┌────────────┐  │
│  │ Spatial         │    │ Temporal       │    │ Feature    │  │
│  │ Transform       │───►│ Correlation    │───►│ Extraction │  │
│  └─────────────────┘    └────────────────┘    └────────────┘  │
│                                                               │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                       QAL Module                              │
│                                                               │
│  ┌─────────────────┐    ┌────────────────┐    ┌────────────┐  │
│  │ Importance Map  │    │ Adaptive       │    │ Soft       │  │
│  │ Generation      │───►│ Quantization   │───►│ Quantizer  │  │
│  └─────────────────┘    └────────────────┘    └────────────┘  │
│                                                               │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                   Standard Codec (Proxy)                      │
│                                                               │
│  ┌─────────────────┐    ┌────────────────┐    ┌────────────┐  │
│  │ Entropy Coding  │    │ Motion         │    │ Transform  │  │
│  │ Estimation      │───►│ Compensation   │───►│ Coding     │  │
│  └─────────────────┘    └────────────────┘    └────────────┘  │
│                                                               │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                   Task Networks                               │
│                                                               │
│  ┌─────────────────┐    ┌────────────────┐    ┌────────────┐  │
│  │ Object          │    │ Semantic       │    │ Object     │  │
│  │ Detection       │    │ Segmentation   │    │ Tracking   │  │
│  └─────────────────┘    └────────────────┘    └────────────┘  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. ST-NPP (Spatio-Temporal Neural Preprocessing)

**Purpose**: Transform raw video frames into a representation that is optimized for both compression and downstream vision tasks.

**Key Features**:
- Spatial transformation: Applies convolutional processing to extract relevant spatial features
- Temporal correlation: Leverages temporal redundancy across frames
- Feature extraction: Identifies and enhances task-relevant features while suppressing irrelevant details

**Implementation**: Found in `models/st_npp.py`

### 2. QAL (Quantization and Adaptation Layer)

**Purpose**: Adaptive quantization of the ST-NPP output based on feature importance for vision tasks.

**Key Features**:
- Importance map generation: Identifies regions crucial for downstream tasks
- Adaptive quantization: Allocates more bits to important regions
- Soft quantization: Ensures differentiability during training

**Implementation**: Found in `models/qal.py`

### 3. Proxy Codec

**Purpose**: Simulates the behavior of standard video codecs (H.264/AVC, H.265/HEVC) during training.

**Key Features**:
- Entropy coding estimation: Approximates bitrate requirements
- Motion compensation: Models temporal prediction in standard codecs
- Transform coding: Simulates DCT/DST transforms used in standard codecs

**Implementation**: Found in `models/proxy_codec.py`

### 4. Combined Model

**Purpose**: Integrates all components into a unified pipeline for end-to-end optimization.

**Key Features**:
- End-to-end training: Jointly optimizes all components
- Task-rate-distortion optimization: Balances compression efficiency and task performance
- Codec compatibility: Ensures outputs are compatible with standard codecs

**Implementation**: Found in `models/combined_model.py`

### 5. Task Networks

**Purpose**: Specialized networks for vision tasks that process compressed video.

**Models**:
- Object Detection: YOLO-based network for identifying and localizing objects
- Semantic Segmentation: U-Net architecture for pixel-level semantic classification
- Object Tracking: Siamese network for tracking objects across frames

**Implementation**: Found in `models/task_networks/`

## Training and Optimization

The system is trained with a multi-objective loss function that balances:
1. Compression efficiency (rate)
2. Reconstruction quality (distortion)
3. Task performance (detection/segmentation/tracking accuracy)

This is implemented through the `TaskAwareLoss` in `utils/loss_utils.py`, which combines:
- `RateDistortionLoss`: Balances bitrate and reconstruction quality
- `PerceptualLoss`: Ensures perceptual quality preservation
- `TemporalConsistencyLoss`: Maintains temporal coherence
- Task-specific losses (detection, segmentation, tracking)

## Evaluation Framework

The system's performance is evaluated using:
- Compression metrics: bitrate, PSNR, MS-SSIM
- Task-specific metrics: mAP for detection, mIoU for segmentation, MOTA for tracking
- Rate-distortion-task (RDT) curves: Visualize the trade-off between compression and task performance

Details can be found in `EVALUATION.md`. 