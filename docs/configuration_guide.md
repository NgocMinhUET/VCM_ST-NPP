# Configuration Guide for Task-Aware Video Preprocessing

This guide explains the configuration parameters for all components in our task-aware video preprocessing system. Use this as a reference when setting up training and evaluation.

## Configuration File Format

All configuration files use YAML format. Example configurations are provided in the `configs/` directory.

## Common Configuration Parameters

These parameters are common across different components:

```yaml
# Common parameters
device: cuda            # Device to run on (cuda/cpu)
seed: 42                # Random seed for reproducibility
num_workers: 4          # Number of data loading workers
batch_size: 8           # Batch size for training/evaluation
clip_length: 16         # Number of frames in each video clip
image_size: [256, 256]  # Input image dimensions [height, width]
```

## ST-NPP Configuration

Parameters for the Spatio-Temporal Neural Preprocessing module:

```yaml
stnpp:
  # Backbone configuration
  backbone: resnet18     # Backbone architecture (resnet18/resnet50/efficientnet)
  pretrained: true       # Whether to use pretrained weights
  
  # Spatial branch
  spatial_channels: 64   # Number of spatial feature channels
  spatial_layers: 4      # Number of spatial layers
  
  # Temporal branch
  temporal_type: 3dcnn   # Temporal branch type (3dcnn/convlstm)
  temporal_channels: 64  # Number of temporal feature channels
  temporal_layers: 2     # Number of temporal layers
  
  # Fusion
  fusion_type: attention # Fusion method (attention/concat)
  output_channels: 128   # Output feature channels
  
  # Downsampling
  downsample_factor: 2   # Spatial downsampling factor
  temporal_reduction: 2  # Temporal reduction factor
```

## QAL Configuration

Parameters for the Quantization Adaptation Layer:

```yaml
qal:
  # Model architecture
  hidden_layers: [256, 256]  # Hidden layer dimensions
  activation: relu           # Activation function (relu/gelu/silu)
  dropout: 0.1               # Dropout rate
  
  # Quantization parameters
  num_centers: 256           # Number of quantization centers
  feature_dim: 128           # Feature dimension
  temperature: 1.0           # Temperature for soft quantization
  
  # QP adaptation
  qp_embedding_dim: 32       # QP embedding dimension
  min_qp: 0                  # Minimum QP value
  max_qp: 51                 # Maximum QP value
  
  # Rate-distortion optimization
  use_importance_map: true   # Whether to use importance map
  importance_channels: 1     # Number of importance map channels
```

## Proxy Codec Configuration

Parameters for the Proxy Codec (used for training):

```yaml
proxy_codec:
  # Architecture
  base_channels: 64            # Base number of channels
  num_layers: 4                # Number of encoder/decoder layers
  use_attention: true          # Whether to use attention mechanisms
  
  # Training
  pretrain_epochs: 100         # Number of epochs for pretraining
  loss_type: mse+lpips         # Loss type (mse/ssim/lpips/combination)
  
  # QP modeling
  qp_conditioning: true        # Whether to condition on QP
  qp_embedding_dim: 32         # QP embedding dimension
  
  # Rate modeling
  estimate_bitrate: true       # Whether to estimate bitrate
  bitrate_lambda: 0.01         # Lambda for bitrate loss
```

## Task Network Configurations

### Object Detection Configuration

```yaml
detection:
  # Architecture
  backbone: darknet53          # Backbone architecture
  num_classes: 80              # Number of object classes (COCO=80)
  anchors_per_level: 3         # Number of anchors per feature level
  
  # Detection head
  num_feature_levels: 3        # Number of feature pyramid levels
  head_channels: 256           # Number of channels in detection head
  
  # Training
  conf_threshold: 0.05         # Confidence threshold during training
  nms_threshold: 0.45          # NMS threshold
  
  # Loss weights
  bbox_loss_weight: 1.0        # Bounding box regression loss weight
  conf_loss_weight: 1.0        # Confidence loss weight
  cls_loss_weight: 0.5         # Classification loss weight
```

### Segmentation Configuration

```yaml
segmentation:
  # Architecture
  model_type: unet             # Model type (unet/attention_unet/deeplabv3plus)
  num_classes: 19              # Number of segmentation classes (19 for KITTI)
  encoder_name: resnet34       # Encoder backbone
  decoder_channels: [256, 128, 64, 32, 16]  # Decoder channel dimensions
  
  # Training
  loss_type: combined          # Loss type (ce/dice/focal/combined)
  class_weights: null          # Optional class weights for imbalanced data
  
  # Augmentation
  use_augmentation: true       # Whether to use data augmentation
  augmentation_prob: 0.5       # Probability of applying augmentation
```

### Tracking Configuration

```yaml
tracking:
  # Architecture
  backbone: resnet18           # Backbone architecture
  feature_dim: 256             # Feature embedding dimension
  
  # Correlation
  correlation_size: 15         # Size of correlation window
  response_size: 31            # Size of response map
  
  # Box regression
  use_box_regression: true     # Whether to use box regression head
  regression_layers: [256, 64] # Box regression head layers
  
  # Training
  template_size: [127, 127]    # Template image size
  search_size: [255, 255]      # Search image size
  pos_threshold: 0.6           # Positive example threshold
  neg_threshold: 0.3           # Negative example threshold
```

## End-to-End Training Configuration

Configuration for training the complete end-to-end system:

```yaml
end2end:
  # Components to use
  use_stnpp: true              # Whether to use ST-NPP
  use_qal: true                # Whether to use QAL
  use_proxy: true              # Whether to use proxy codec (training only)
  task: detection              # Task to train for (detection/segmentation/tracking)
  
  # Training parameters
  epochs: 100                  # Number of training epochs
  lr: 0.0001                   # Learning rate
  weight_decay: 0.0001         # Weight decay
  lr_scheduler: cosine         # Learning rate scheduler
  
  # Loss weights
  task_loss_weight: 1.0        # Task-specific loss weight
  distortion_loss_weight: 0.1  # Distortion loss weight
  rate_loss_weight: 0.01       # Rate loss weight
  
  # Quality presets
  qp_values: [22, 27, 32, 37]  # QP values to train on
  
  # Checkpointing
  save_frequency: 10           # Epoch frequency for saving checkpoints
  checkpoint_dir: checkpoints  # Directory to save checkpoints
```

## Evaluation Configuration

Configuration for evaluating the system:

```yaml
evaluation:
  # Components to use
  use_stnpp: true              # Whether to use ST-NPP 
  use_qal: true                # Whether to use QAL
  use_real_codec: true         # Whether to use real codec (HEVC/H.265)
  task: detection              # Task to evaluate (detection/segmentation/tracking)
  
  # Codec parameters
  codec: libx265               # Video codec to use
  qp_values: [22, 27, 32, 37]  # QP values to evaluate on
  preset: medium               # Encoder preset
  
  # Metrics
  calculate_psnr: true         # Calculate PSNR
  calculate_ssim: true         # Calculate SSIM
  calculate_lpips: false       # Calculate LPIPS (perceptual similarity)
  calculate_bpp: true          # Calculate bits per pixel
  
  # Visualization
  generate_visualizations: true  # Generate visualization images/videos
  save_encoded_videos: true      # Save encoded videos
  num_vis_examples: 5            # Number of examples to visualize
```

## Creating Custom Configurations

To create a custom configuration:

1. Copy one of the example configurations from `configs/`
2. Modify the parameters as needed
3. Save with a descriptive name (e.g., `configs/custom_tracking_4k.yaml`)

Example:
```bash
cp configs/end2end_config.yaml configs/custom_tracking_4k.yaml
# Edit the file with your custom settings
```

## Configuration Tips

- For higher resolution videos, increase `batch_size` and adjust learning rate accordingly
- For different datasets, adjust `num_classes` to match the dataset's class count
- When fine-tuning on specific tasks, increase the corresponding task loss weight
- For different quality-bitrate tradeoffs, experiment with different `qp_values`
- When limited by GPU memory, reduce `clip_length` or `batch_size` 