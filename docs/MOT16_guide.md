# MOT16 Dataset Guide for Task-Aware Video Preprocessing

This guide provides detailed instructions for using the MOT16 dataset with our task-aware video preprocessing system, specifically for the multiple object tracking task.

## Dataset Overview

The MOT16 dataset is part of the MOTChallenge benchmark and consists of 14 video sequences (7 for training and 7 for testing) with pedestrian tracking annotations.

### Key Statistics:
- 7 training sequences (02, 04, 05, 09, 10, 11, 13)
- 7 test sequences (01, 03, 06, 07, 08, 12, 14)
- Video resolution: varies by sequence
- Frame rate: 30 FPS (for most sequences)
- Annotations: bounding boxes for pedestrians

## Dataset Structure

After downloading and extracting, the dataset will have the following structure:

```
MOT16/
├── train/
│   ├── MOT16-02/
│   │   ├── img1/          # JPG image frames (000001.jpg, 000002.jpg, etc.)
│   │   ├── gt/            # Ground truth annotations
│   │   │   └── gt.txt     # Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
│   │   └── det/           # Pre-computed detections (if available)
│   ├── MOT16-04/
│   │   └── ...
│   └── ...
└── test/
    ├── MOT16-01/
    │   ├── img1/          # JPG image frames
    │   └── det/           # Pre-computed detections
    ├── MOT16-03/
    │   └── ...
    └── ...
```

## Ground Truth Format

The ground truth annotations in `gt.txt` use the following format:
```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>
```

Example:
```
1,1,912,484,97,109,0,7,1
```

Where:
- `frame`: Frame number (1-based)
- `id`: Object ID (unique for each trajectory)
- `bb_left`, `bb_top`, `bb_width`, `bb_height`: Bounding box coordinates
- `conf`: Detection confidence (0 for ground truth)
- `class`: Object class (7 = pedestrian)
- `visibility`: Visibility ratio (0-1)

## Using MOT16 with Our System

### Data Preparation

1. Download the MOT16 dataset from the [MOTChallenge website](https://motchallenge.net/data/MOT16/).
2. Extract the dataset to `data/MOT16/`.
3. Use our data preprocessing tools to convert the dataset into the format required by our system:

```bash
python utils/preprocess_mot16.py --input_path data/MOT16 --output_path data/processed/MOT16
```

### Training the Tracking Model

To train our tracking model on the MOT16 dataset:

```bash
python train.py --mode end2end --config configs/tracking_config.yaml --task tracking --data_path data/processed/MOT16 --output_dir results/tracking
```

### Evaluation on MOT16

To evaluate the tracking performance on the MOT16 test set:

```bash
python evaluate.py --model_path results/tracking/best_model.pth --task tracking --data_path data/processed/MOT16/test --output_dir eval_results/tracking
```

This will generate:
- Tracking results in MOT format
- Evaluation metrics (MOTA, IDF1, etc.)
- Visualizations of tracking results

## Performance Metrics

The tracking performance is evaluated using the following metrics:

- **MOTA** (Multiple Object Tracking Accuracy): overall tracking accuracy
- **IDF1** (ID F1 Score): ID global min-cost matching F1 score
- **MT** (Mostly Tracked): number of mostly tracked trajectories (>80% tracked)
- **ML** (Mostly Lost): number of mostly lost trajectories (<20% tracked)
- **FP** (False Positives): number of false positive detections
- **FN** (False Negatives): number of missed detections
- **IDSW** (ID Switches): number of times the ID of a tracked object changes

## Acknowledgements

We thank the creators of the MOT16 dataset:

```
@article{milan2016mot16,
  title={MOT16: A benchmark for multi-object tracking},
  author={Milan, Anton and Leal-Taix{\'e}, Laura and Reid, Ian and Roth, Stefan and Schindler, Konrad},
  journal={arXiv preprint arXiv:1603.00831},
  year={2016}
}
``` 