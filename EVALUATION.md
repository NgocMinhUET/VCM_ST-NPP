# Evaluation Framework for VCM-ST-NPP

## Compression Metrics
- **Rate** (Bits Per Pixel - BPP): Measures the average number of bits used per pixel in the compressed representation
- **Distortion** (PSNR, MS-SSIM): Quantifies the quality of reconstruction
  - PSNR (Peak Signal-to-Noise Ratio): Traditional quality metric based on pixel differences
  - MS-SSIM (Multi-Scale Structural Similarity): Perceptual metric that better correlates with human vision
- **Compression Ratio**: Original size divided by compressed size
- **Encoding/Decoding Time**: Performance metrics for compression efficiency

## Task Performance Metrics

### Object Detection
- **mAP (mean Average Precision)**: Primary metric for object detection quality
- **AP50, AP75**: AP at IoU thresholds of 0.5 and 0.75
- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth bounding boxes
- **Precision/Recall**: Measures detection accuracy at different confidence thresholds

### Semantic Segmentation
- **mIoU (mean Intersection over Union)**: Primary metric for segmentation quality
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Dice Coefficient**: Harmonic mean of precision and recall for segmentation masks
- **Per-class IoU**: IoU values for individual semantic classes

### Object Tracking
- **MOTA (Multiple Object Tracking Accuracy)**: Primary metric for tracking performance
- **IDF1 (Identity F1 Score)**: Ratio of correctly identified detections over average number of ground truth and computed detections
- **Track Fragmentation**: Number of interruptions in established tracked targets
- **ID Switches**: Number of times the ID of a tracked object changes
- **False Positives/Negatives**: Number of incorrect/missed detections

## Evaluation Protocols

### Standard-Compatible Evaluation
1. **Raw Video Input**: Uncompressed frames from standard datasets (MOT16, KITTI, etc.)
2. **ST-NPP Processing**: Application of neural preprocessing to produce codec-ready frames
3. **Standard Codec Compression**: Compression using HEVC/H.265 at various quality levels (QP)
4. **Decompression and Task Processing**: Running specific vision tasks on decompressed frames
5. **Metric Calculation**: Computing both compression and task-specific metrics

### Benchmark Comparisons
The project includes a comprehensive framework for comparing against:
- **Standard Codecs**: H.264/AVC, H.265/HEVC, VP9
- **Neural Compression Methods**: Without task-specific optimization
- **Joint Optimization Methods**: Similar approaches from literature

### Rate-Distortion-Task Performance Analysis
- **RD Curves**: PSNR/MS-SSIM vs. Bit Rate plots
- **RT Curves**: Task performance vs. Bit Rate plots
- **Ablation Studies**: Impact of individual components (ST-NPP, QAL, etc.)
- **Framework Robustness**: Performance across different video characteristics and scenes

## Execution
- `evaluate.py`: Single-model evaluation on specific tasks
- `compare_compression_methods.py`: Benchmark comparison against standard methods
- `run_comprehensive_evaluation.py`: End-to-end pipeline for full evaluation suite
- `run_comprehensive_evaluation.sh`: Convenience script for typical evaluation scenarios 