"""
Data utilities for task-aware video compression.

This module provides dataset classes and data processing utilities for
loading video data for task-aware compression and analysis.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Union, Optional, Callable
import random
from pathlib import Path
import json
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import shutil
from torchvision import transforms as F


class VideoDataset(Dataset):
    """
    Base class for video datasets.
    """
    def __init__(self, 
                root_dir: str,
                clip_len: int = 16,
                frame_stride: int = 1,
                resolution: Tuple[int, int] = (256, 256),
                split: str = "train",
                transform: Optional[Callable] = None):
        """
        Initialize VideoDataset.
        
        Args:
            root_dir: Root directory of the dataset
            clip_len: Number of frames in each clip
            frame_stride: Stride between frames
            resolution: Target resolution (H, W)
            split: Data split ('train', 'val', 'test')
            transform: Optional transform to be applied on frames
        """
        self.root_dir = Path(root_dir)
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.split = split
        self.transform = transform
        
        # Default transform if none is provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
            ])
        
        # Will be implemented by subclasses
        self.video_paths = []
        self.annotations = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """
        Load dataset structure.
        Finds all video files in the dataset directory and prepares them for loading.
        """
        # Find all video files in the split directory
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Look for video files with common extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        for ext in video_extensions:
            self.video_paths.extend(list(split_dir.glob(f'**/*{ext}')))
        
        if len(self.video_paths) == 0:
            print(f"Warning: No video files found in {split_dir}")
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video clip and its annotations.
        
        Args:
            idx: Index of the video
            
        Returns:
            Dictionary with video frames and annotations
        """
        video_path = self.video_paths[idx]
        
        # Open the video file
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Choose a random starting point if the video is longer than the clip length
        if total_frames > self.clip_len * self.frame_stride:
            start_frame = random.randint(0, total_frames - self.clip_len * self.frame_stride)
        else:
            start_frame = 0
        
        # Read frames
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for i in range(self.clip_len):
            # Jump to the next frame based on stride
            frame_idx = start_frame + i * self.frame_stride
            
            if frame_idx >= total_frames:
                # If we run out of frames, repeat the last frame
                frame_idx = total_frames - 1
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # If reading failed, generate a black frame
                frame = np.zeros((self.resolution[0], self.resolution[1], 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            frame = Image.fromarray(frame)
            
            # Apply transforms
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
        
        # Release the video
        cap.release()
        
        # Stack frames
        frames = torch.stack(frames)
        
        return {
            'frames': frames,  # [T, C, H, W]
            'video_path': str(video_path),
            'start_frame': start_frame,
            'fps': fps
        }


class MOTDataset(Dataset):
    """
    Dataset for Multi-Object Tracking data (MOT Challenge format).
    """
    def __init__(self, 
                root_dir: str,
                clip_len: int = 16,
                frame_stride: int = 1,
                resolution: Tuple[int, int] = (640, 1080),
                split: str = "train",
                transform: Optional[Callable] = None,
                task_type: str = "tracking"):
        """
        Initialize MOTDataset.
        
        Args:
            root_dir: Root directory of the MOT dataset
            clip_len: Number of frames in each clip
            frame_stride: Stride between frames
            resolution: Target resolution (H, W)
            split: Data split ('train', 'val', 'test')
            transform: Optional transform to be applied on frames
            task_type: Task type ('detection', 'tracking')
        """
        self.root_dir = Path(root_dir)
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.split = split
        self.transform = transform
        self.task_type = task_type
        
        # Default transform if none is provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
            ])
        
        self.sequences = []
        self.annotations = {}
        self.video_paths = []  # Initialize video_paths list
        
        self._load_dataset()
    
    def _load_dataset(self):
        """
        Load MOT dataset sequences and annotations.
        """
        # Find all sequences in the split folder (MOT16/train, MOT16/test, etc.)
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Get all sequence folders
        seq_folders = [f for f in split_dir.iterdir() if f.is_dir()]
        
        for seq_folder in seq_folders:
            seq_name = seq_folder.name
            self.sequences.append(seq_name)
            
            # Parse ground truth annotations if available
            gt_file = seq_folder / "gt" / "gt.txt"
            frame_mappings = {}
            
            if gt_file.exists():
                # Format: <frame_id> <track_id> <bb_left> <bb_top> <bb_width> <bb_height> <conf> <class> <visibility>
                with open(gt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 7:
                            continue
                        
                        frame_id = int(parts[0])
                        track_id = int(parts[1])
                        left = float(parts[2])
                        top = float(parts[3])
                        width = float(parts[4])
                        height = float(parts[5])
                        conf = float(parts[6])
                        class_id = int(parts[7]) if len(parts) > 7 else 1  # Default to person
                        
                        # Skip if confidence is 0
                        if conf == 0:
                            continue
                        
                        if frame_id not in frame_mappings:
                            frame_mappings[frame_id] = {
                                'boxes': [],
                                'track_ids': [],
                                'classes': []
                            }
                        
                        frame_mappings[frame_id]['boxes'].append([left, top, left + width, top + height])
                        frame_mappings[frame_id]['track_ids'].append(track_id)
                        frame_mappings[frame_id]['classes'].append(class_id)
            
            # Store annotations for this sequence
            self.annotations[seq_name] = frame_mappings
            
            # Count number of frames in the sequence
            img_dir = seq_folder / "img1"
            frame_count = len(list(img_dir.glob('*.jpg')))
            
            # Add clips from this sequence to the dataset
            # Each clip is a continuous segment of frame_stride * clip_len frames
            # with starting points sampled at intervals
            step = max(1, frame_count // 20)  # Create about 20 clips per sequence
            
            for start_frame in range(1, frame_count - self.clip_len * self.frame_stride + 1, step):
                self.video_paths.append({
                    'sequence': seq_name,
                    'start_frame': start_frame,
                    'frame_count': frame_count
                })
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video clip and its annotations.
        
        Args:
            idx: Index of the clip
            
        Returns:
            Dictionary with video frames and annotations
        """
        item = self.video_paths[idx]
        seq_name = item['sequence']
        start_frame = item['start_frame']
        
        seq_dir = self.root_dir / self.split / seq_name / "img1"
        
        # Load frames
        frames = []
        frame_ids = []
        
        for i in range(self.clip_len):
            frame_id = start_frame + i * self.frame_stride
            frame_ids.append(frame_id)
            
            # Load image
            img_path = seq_dir / f"{frame_id:06d}.jpg"
            img = Image.open(img_path).convert('RGB')
            
            # Apply transform
            if self.transform:
                img = self.transform(img)
            
            frames.append(img)
        
        # Stack frames
        frames = torch.stack(frames)
        
        # Load annotations for these frames
        boxes = []
        track_ids = []
        classes = []
        
        for frame_id in frame_ids:
            if seq_name in self.annotations and frame_id in self.annotations[seq_name]:
                # Get boxes for this frame
                frame_boxes = self.annotations[seq_name][frame_id]['boxes']
                frame_track_ids = self.annotations[seq_name][frame_id]['track_ids']
                frame_classes = self.annotations[seq_name][frame_id]['classes']
                
                # Normalize box coordinates to [0, 1]
                norm_boxes = []
                for box in frame_boxes:
                    # Format is [x1, y1, x2, y2]
                    x1 = box[0] / self.resolution[1]
                    y1 = box[1] / self.resolution[0]
                    x2 = box[2] / self.resolution[1]
                    y2 = box[3] / self.resolution[0]
                    norm_boxes.append([x1, y1, x2, y2])
                
                boxes.append(torch.tensor(norm_boxes, dtype=torch.float32))
                track_ids.append(torch.tensor(frame_track_ids, dtype=torch.int64))
                classes.append(torch.tensor(frame_classes, dtype=torch.int64))
            else:
                # Empty annotations for this frame
                boxes.append(torch.zeros((0, 4), dtype=torch.float32))
                track_ids.append(torch.zeros((0,), dtype=torch.int64))
                classes.append(torch.zeros((0,), dtype=torch.int64))
        
        result = {
            'frames': frames,  # [T, C, H, W]
            'boxes': boxes,    # List of T tensors with shape [N_i, 4]
            'classes': classes, # List of T tensors with shape [N_i]
            'sequence': seq_name,
            'frame_ids': torch.tensor(frame_ids, dtype=torch.int64)
        }
        
        if self.task_type == 'tracking':
            result['track_ids'] = track_ids  # List of T tensors with shape [N_i]
        
        return result


class KITTIDataset(Dataset):
    """
    Dataset for KITTI tracking benchmark.
    """
    def __init__(self, 
                root_dir: str,
                clip_len: int = 16,
                frame_stride: int = 1,
                resolution: Tuple[int, int] = (384, 1280),
                split: str = "training",
                transform: Optional[Callable] = None,
                task_type: str = "tracking"):
        """
        Initialize KITTIDataset.
        
        Args:
            root_dir: Root directory of the KITTI dataset
            clip_len: Number of frames in each clip
            frame_stride: Stride between frames
            resolution: Target resolution (H, W)
            split: Data split ('training', 'testing')
            transform: Optional transform to be applied on frames
            task_type: Task type ('detection', 'tracking')
        """
        self.root_dir = Path(root_dir)
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.resolution = resolution
        self.split = split
        self.transform = transform
        self.task_type = task_type
        
        # Default transform if none is provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
            ])
        
        self.sequences = []
        self.annotations = {}
        self.video_paths = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """
        Load KITTI dataset sequences and annotations.
        """
        # Dataset structure:
        # root_dir/
        #   data_tracking_image_2/
        #     training/image_02/
        #       0000/
        #         000000.png, 000001.png, ...
        #       0001/
        #         ...
        #   data_tracking_label_2/
        #     training/label_02/
        #       0000.txt, 0001.txt, ...
        
        # Find all sequences in the split folder
        img_dir = self.root_dir / f"data_tracking_image_2/{self.split}/image_02"
        if not img_dir.exists():
            raise ValueError(f"Image directory {img_dir} does not exist")
        
        # Get all sequence folders
        seq_folders = [f for f in img_dir.iterdir() if f.is_dir()]
        
        for seq_folder in seq_folders:
            seq_name = seq_folder.name
            self.sequences.append(seq_name)
            
            # Parse ground truth annotations if available
            label_file = self.root_dir / f"data_tracking_label_2/{self.split}/label_02/{seq_name}.txt"
            frame_mappings = {}
            
            if label_file.exists() and self.split == "training":
                # Format: <frame> <track_id> <type> <truncated> <occluded> <alpha> <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> <dim_height> <dim_width> <dim_length> <loc_x> <loc_y> <loc_z> <rotation_y>
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 17:
                            continue
                        
                        frame_id = int(parts[0])
                        track_id = int(parts[1])
                        obj_type = parts[2]
                        
                        # Skip non-vehicle objects if desired
                        if obj_type not in ['Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist']:
                            continue
                        
                        # Map object type to class id
                        class_map = {
                            'Car': 1, 
                            'Van': 2, 
                            'Truck': 3, 
                            'Pedestrian': 4, 
                            'Cyclist': 5
                        }
                        class_id = class_map.get(obj_type, 0)
                        
                        # Get bounding box coordinates [left, top, right, bottom]
                        left = float(parts[6])
                        top = float(parts[7])
                        right = float(parts[8])
                        bottom = float(parts[9])
                        
                        # Add to lists
                        if frame_id not in frame_mappings:
                            frame_mappings[frame_id] = {
                                'boxes': [],
                                'track_ids': [],
                                'classes': []
                            }
                        
                        frame_mappings[frame_id]['boxes'].append([left, top, right, bottom])
                        frame_mappings[frame_id]['track_ids'].append(track_id)
                        frame_mappings[frame_id]['classes'].append(class_id)
            
            # Store annotations for this sequence
            self.annotations[seq_name] = frame_mappings
            
            # Count number of frames in the sequence
            frame_count = len(list(seq_folder.glob('*.png')))
            
            # Add clips from this sequence to the dataset
            # Each clip is a continuous segment of frame_stride * clip_len frames
            # with starting points sampled at intervals
            step = max(1, frame_count // 10)  # Create about 10 clips per sequence
            
            for start_frame in range(0, frame_count - self.clip_len * self.frame_stride + 1, step):
                self.video_paths.append({
                    'sequence': seq_name,
                    'start_frame': start_frame,
                    'frame_count': frame_count
                })
    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video clip and its annotations.
        
        Args:
            idx: Index of the clip
            
        Returns:
            Dictionary with video frames and annotations
        """
        item = self.video_paths[idx]
        seq_name = item['sequence']
        start_frame = item['start_frame']
        
        seq_dir = self.root_dir / f"data_tracking_image_2/{self.split}/image_02/{seq_name}"
        
        # Load frames
        frames = []
        frame_ids = []
        
        for i in range(self.clip_len):
            frame_id = start_frame + i * self.frame_stride
            frame_ids.append(frame_id)
            
            # Load image
            img_path = seq_dir / f"{frame_id:06d}.png"
            img = Image.open(img_path).convert('RGB')
            
            # Apply transform
            if self.transform:
                img = self.transform(img)
            
            frames.append(img)
        
        # Stack frames
        frames = torch.stack(frames)
        
        # Load annotations for these frames
        boxes = []
        track_ids = []
        classes = []
        
        for frame_id in frame_ids:
            if seq_name in self.annotations and frame_id in self.annotations[seq_name]:
                # Get boxes for this frame
                frame_boxes = self.annotations[seq_name][frame_id]['boxes']
                frame_track_ids = self.annotations[seq_name][frame_id]['track_ids']
                frame_classes = self.annotations[seq_name][frame_id]['classes']
                
                # Normalize box coordinates to [0, 1]
                norm_boxes = []
                for box in frame_boxes:
                    # Format is [x1, y1, x2, y2]
                    x1 = box[0] / self.resolution[1]
                    y1 = box[1] / self.resolution[0]
                    x2 = box[2] / self.resolution[1]
                    y2 = box[3] / self.resolution[0]
                    norm_boxes.append([x1, y1, x2, y2])
                
                boxes.append(torch.tensor(norm_boxes, dtype=torch.float32))
                track_ids.append(torch.tensor(frame_track_ids, dtype=torch.int64))
                classes.append(torch.tensor(frame_classes, dtype=torch.int64))
            else:
                # Empty annotations for this frame
                boxes.append(torch.zeros((0, 4), dtype=torch.float32))
                track_ids.append(torch.zeros((0,), dtype=torch.int64))
                classes.append(torch.zeros((0,), dtype=torch.int64))
        
        result = {
            'frames': frames,  # [T, C, H, W]
            'boxes': boxes,    # List of T tensors with shape [N_i, 4]
            'classes': classes, # List of T tensors with shape [N_i]
            'sequence': seq_name,
            'frame_ids': torch.tensor(frame_ids, dtype=torch.int64)
        }
        
        if self.task_type == 'tracking':
            result['track_ids'] = track_ids  # List of T tensors with shape [N_i]
        
        return result


class DummyVideoDataset(Dataset):
    """
    Dummy video dataset for testing purposes.
    """
    def __init__(self, 
                num_samples: int = 100,
                clip_len: int = 16,
                resolution: Tuple[int, int] = (256, 256),
                num_channels: int = 3,
                task_type: str = "detection"):
        """
        Initialize DummyVideoDataset.
        
        Args:
            num_samples: Number of dummy samples
            clip_len: Number of frames in each clip
            resolution: Frame resolution (H, W)
            num_channels: Number of channels per frame
            task_type: Task type ('detection', 'segmentation', 'tracking')
        """
        self.num_samples = num_samples
        self.clip_len = clip_len
        self.resolution = resolution
        self.num_channels = num_channels
        self.task_type = task_type
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a random video clip and annotations.
        
        Args:
            idx: Index of the clip
            
        Returns:
            Dictionary with video frames and annotations
        """
        # Generate random frames
        frames = torch.rand(self.clip_len, self.num_channels, *self.resolution)
        
        # Generate random annotations
        if self.task_type == "detection" or self.task_type == "tracking":
            # Generate random boxes
            boxes = []
            classes = []
            track_ids = []
            
            for _ in range(self.clip_len):
                # Random number of objects per frame
                num_objects = random.randint(1, 5)
                
                # Generate random boxes [x1, y1, x2, y2] in [0, 1]
                frame_boxes = torch.rand(num_objects, 4)
                # Ensure x1 < x2 and y1 < y2
                frame_boxes[:, 2:] = frame_boxes[:, :2] + frame_boxes[:, 2:] * 0.3
                frame_boxes = torch.clamp(frame_boxes, 0, 1)
                
                # Generate random classes (1-10)
                frame_classes = torch.randint(1, 11, (num_objects,))
                
                # Generate random track IDs for tracking
                if self.task_type == "tracking":
                    frame_track_ids = torch.arange(num_objects) + 1
                    track_ids.append(frame_track_ids)
                
                boxes.append(frame_boxes)
                classes.append(frame_classes)
            
            result = {
                'frames': frames,  # [T, C, H, W]
                'boxes': boxes,    # List of T tensors with shape [N_i, 4]
                'classes': classes, # List of T tensors with shape [N_i]
                'sequence': f"dummy_{idx}",
                'frame_ids': torch.arange(self.clip_len)
            }
            
            if self.task_type == "tracking":
                result['track_ids'] = track_ids
                
        elif self.task_type == "segmentation":
            # Generate random segmentation masks
            masks = []
            
            for _ in range(self.clip_len):
                # Generate random segmentation mask with C classes
                num_classes = 10
                mask = torch.randint(0, num_classes, (self.resolution[0], self.resolution[1]))
                masks.append(mask)
            
            result = {
                'frames': frames,  # [T, C, H, W]
                'masks': torch.stack(masks),  # [T, H, W]
                'sequence': f"dummy_{idx}",
                'frame_ids': torch.arange(self.clip_len)
            }
        
        return result


def get_transforms(task_type: str = 'detection', resolution: Tuple[int, int] = (256, 256), augment: bool = True):
    """
    Get appropriate transforms for the specified task type.
    
    Args:
        task_type: Type of task ('detection', 'segmentation', 'tracking')
        resolution: Target resolution (H, W), defaults to (256, 256)
        augment: Whether to apply data augmentation transforms, defaults to True
        
    Returns:
        If augment is True: tuple of (train_transform, val_transform)
        If augment is False: single transform
        
    Raises:
        ValueError: If an unsupported task type is provided
    """
    # Basic transforms for both training and validation
    basic_transforms = [
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # Additional transforms for training with augmentation
    augmentation_transforms = []
    if augment:
        if task_type in ['detection', 'tracking', 'segmentation']:
            augmentation_transforms = [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
    
    # Create transform compositions
    train_transform = transforms.Compose(augmentation_transforms + basic_transforms)
    val_transform = transforms.Compose(basic_transforms)
    
    # Return appropriate transforms based on augment flag
    if augment:
        return train_transform, val_transform
    else:
        return val_transform


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized labels.
    
    Args:
        batch: List of data samples
        
    Returns:
        Dictionary with collated data
        
    Raises:
        RuntimeError: If there's an error collating the batch
    """
    try:
        # Extract all keys from the batch
        batch_dict = {key: [] for key in batch[0].keys()}
        
        # Group items by key
        for sample in batch:
            for key, value in sample.items():
                batch_dict[key].append(value)
        
        # Stack frames
        batch_dict['frames'] = torch.stack(batch_dict['frames'])
        
        # QP values can be stacked or converted to tensor
        if isinstance(batch_dict['qp'][0], (int, float)):
            batch_dict['qp'] = torch.tensor(batch_dict['qp'])
        else:
            batch_dict['qp'] = torch.stack(batch_dict['qp'])
        
        # Labels might be of variable size, so we just keep them as a list
        # batch_dict['labels'] already contains the list of labels
        
        return batch_dict
    except Exception as e:
        raise RuntimeError(f"Error collating batch: {str(e)}")


def get_video_dataloader(
    dataset: Union[Dataset, str],
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    task_type: str = "detection",
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for video datasets.
    
    Args:
        dataset: Dataset instance or string name of dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        task_type: Type of task (detection, segmentation, tracking)
        **dataset_kwargs: Additional arguments for dataset creation
        
    Returns:
        DataLoader instance
    """
    # Create dataset if string is provided
    if isinstance(dataset, str):
        if dataset == "dummy":
            dataset = DummyVideoDataset(**dataset_kwargs)
        elif dataset == "mot":
            dataset = MOTDataset(**dataset_kwargs)
        elif dataset == "kitti":
            dataset = KITTIDataset(**dataset_kwargs)
        else:
            raise ValueError(f"Unknown dataset name: {dataset}")
    
    # Select appropriate collate function based on task type
    if task_type == "detection":
        collate_fn = collate_video_detection
    elif task_type == "segmentation":
        collate_fn = collate_video_segmentation
    elif task_type == "tracking":
        collate_fn = collate_video_tracking
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader


class VideoSequenceDataset(Dataset):
    """Dataset for loading video sequences with task labels."""
    
    def __init__(
        self,
        root_dir: str,
        task_type: str = 'detection',
        split: str = 'train',
        seq_length: int = 5,
        transform = None,
        random_qp: bool = False,
        qp_range: Tuple[int, int] = (22, 37)
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory of the dataset
            task_type: Type of task ('detection', 'segmentation', 'tracking')
            split: Dataset split ('train', 'val', 'test')
            seq_length: Number of frames in each sequence
            transform: Optional transform to be applied on images
            random_qp: Whether to use random QP values
            qp_range: Range of QP values if random_qp is True
        
        Raises:
            RuntimeError: If the dataset directory doesn't exist or no valid sequences could be found
        """
        self.root = root_dir
        self.task_type = task_type
        self.split = split
        self.seq_length = seq_length
        self.transform = transform
        self.random_qp = random_qp
        self.qp_range = qp_range
        
        print(f"Initializing VideoSequenceDataset for {task_type} task, {split} split")
        print(f"Looking for data in: {root_dir}")
        
        # Find all valid sequences
        self.sequences = self._find_sequences()

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single video sequence with its labels.
        
        Args:
            idx: Index of the sequence to retrieve
            
        Returns:
            Dictionary containing:
                - frames: Tensor of shape [seq_length, C, H, W]
                - labels: List of labels for each frame
                - qp: QP value for the sequence
                - video_name: Name of the video
                - frame_indices: Indices of frames in the sequence
                
        Raises:
            FileNotFoundError: If an image or label file doesn't exist
            RuntimeError: If there's an error processing the images or labels
        """
        sequence = self.sequences[idx]
        
        # Load all frames in the sequence
        frames = []
        labels = []
        
        for frame_path, label_path in zip(sequence['frames'], sequence['labels']):
            # Load and process the image
            img = self._load_image(frame_path)
            frames.append(img)
            
            # Load and process the label
            label = self._load_label(label_path)
            labels.append(label)
        
        # Stack frames into a single tensor
        frames = torch.stack(frames, dim=0)
        
        # Generate QP value
        if self.random_qp:
            qp = random.randint(self.qp_range[0], self.qp_range[1])
        else:
            qp = 32  # Default QP value
            
        return {
            'frames': frames,
            'labels': labels,
            'qp': qp,
            'video_name': sequence['video_name'],
            'frame_indices': sequence['frame_indices']
        }

    def _find_sequences(self) -> List[Dict]:
        """
        Find all valid video sequences in the dataset.
        
        Returns:
            List of dictionaries, each containing 'frames' and 'labels' paths for a sequence
            
        Raises:
            RuntimeError: If the dataset root directory doesn't exist or no valid sequences are found
        """
        # Verify the root directory exists
        if not os.path.exists(self.root):
            raise RuntimeError(f"Dataset root directory does not exist: {self.root}")
            
        # Construct paths for task and split
        task_dir = os.path.join(self.root, self.task_type)
        split_dir = os.path.join(task_dir, self.split)
        
        if not os.path.exists(split_dir):
            raise RuntimeError(f"Split directory not found: {split_dir}")
            
        # Find all video directories
        video_dirs = []
        for item in os.listdir(split_dir):
            video_path = os.path.join(split_dir, item)
            if os.path.isdir(video_path):
                video_dirs.append(video_path)
                
        if not video_dirs:
            # Try alternative directory structures
            print(f"No video directories found in standard structure {split_dir}")
            # Check if the split_dir itself contains frames and labels directly
            frames_dir = os.path.join(split_dir, 'frames')
            if os.path.exists(frames_dir) and os.path.isdir(frames_dir):
                print(f"Found frames directory directly under {split_dir}")
                labels_dir = os.path.join(split_dir, 'labels')
                masks_dir = os.path.join(split_dir, 'masks')
                tracks_dir = os.path.join(split_dir, 'tracks')
                
                # Use first available labels directory based on task
                label_dir = None
                if self.task_type == 'tracking' and os.path.exists(tracks_dir):
                    label_dir = tracks_dir
                elif self.task_type == 'segmentation' and os.path.exists(masks_dir):
                    label_dir = masks_dir
                elif os.path.exists(labels_dir):
                    label_dir = labels_dir
                
                if label_dir:
                    # Create sequences directly from frames_dir and label_dir
                    return self._create_sequences_from_dirs(frames_dir, label_dir, os.path.basename(split_dir))
            
            raise RuntimeError(f"No video directories found in {split_dir}")
            
        sequences = []
        
        for video_dir in tqdm(video_dirs, desc=f"Loading {self.task_type} {self.split} sequences"):
            frames_dir = os.path.join(video_dir, 'frames')
            if not os.path.exists(frames_dir):
                print(f"Warning: Frames directory not found for {video_dir}, skipping")
                continue
                
            # Find label path based on task type
            if self.task_type == 'detection':
                labels_dir = os.path.join(video_dir, 'labels')
            elif self.task_type == 'segmentation':
                labels_dir = os.path.join(video_dir, 'masks')
            elif self.task_type == 'tracking':
                labels_dir = os.path.join(video_dir, 'tracks')
            else:
                labels_dir = os.path.join(video_dir, 'labels')  # Default
                
            if not os.path.exists(labels_dir):
                print(f"Warning: Labels directory not found for {video_dir}, skipping")
                continue
                
            # Get all frame files and sort them
            frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            frame_files.sort()
            
            if not frame_files:
                print(f"Warning: No frame files found in {frames_dir}, skipping")
                continue
                
            # Get corresponding label files
            label_files = []
            for frame_file in frame_files:
                # Find matching label based on task type
                frame_stem = os.path.splitext(frame_file)[0]
                
                if self.task_type == 'detection':
                    label_file = f"{frame_stem}.txt"
                elif self.task_type == 'segmentation':
                    label_file = f"{frame_stem}.png"
                elif self.task_type == 'tracking':
                    label_file = f"{frame_stem}.txt"
                else:
                    label_file = f"{frame_stem}.txt"  # Default
                    
                label_path = os.path.join(labels_dir, label_file)
                
                # Check if label file exists
                if not os.path.exists(label_path):
                    print(f"Warning: Label file {label_path} not found, skipping frame")
                    continue
                    
                label_files.append(label_file)
                
            if not label_files:
                print(f"Warning: No valid label files found for {video_dir}, skipping")
                continue
                
            # Create sequences with seq_length frames
            for i in range(0, len(frame_files) - self.seq_length + 1):
                seq_frames = [os.path.join(frames_dir, frame_files[i + j]) for j in range(self.seq_length)]
                seq_labels = [os.path.join(labels_dir, label_files[i + j]) for j in range(self.seq_length)]
                
                sequences.append({
                    'frames': seq_frames,
                    'labels': seq_labels,
                    'video_name': os.path.basename(video_dir),
                    'frame_indices': list(range(i, i + self.seq_length))
                })
        
        if not sequences:
            raise RuntimeError(
                f"No valid sequences found for {self.task_type} dataset in {self.split} split. "
                f"Please check that the dataset is properly organized."
            )
            
        print(f"Found {len(sequences)} valid sequences for {self.task_type} {self.split}")
        return sequences
        
    def _create_sequences_from_dirs(self, frames_dir, labels_dir, video_name):
        """
        Create sequences from frames and labels directories directly.
        
        Args:
            frames_dir: Directory containing frame images
            labels_dir: Directory containing label files
            video_name: Name of the video/sequence
            
        Returns:
            List of dictionaries, each containing 'frames' and 'labels' paths for a sequence
        """
        # Get all frame files and sort them
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        frame_files.sort()
        
        if not frame_files:
            print(f"Warning: No frame files found in {frames_dir}")
            return []
            
        # Get corresponding label files
        sequence_frames = []
        sequence_labels = []
        
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            
            # Find matching label based on task type
            frame_stem = os.path.splitext(frame_file)[0]
            
            if self.task_type == 'detection':
                label_file = f"{frame_stem}.txt"
            elif self.task_type == 'segmentation':
                label_file = f"{frame_stem}.png"
            elif self.task_type == 'tracking':
                label_file = f"{frame_stem}.txt"
            else:
                label_file = f"{frame_stem}.txt"  # Default
                
            label_path = os.path.join(labels_dir, label_file)
            
            # Check if label file exists
            if not os.path.exists(label_path):
                print(f"Warning: Label file {label_path} not found for {frame_file}, skipping")
                continue
                
            sequence_frames.append(frame_path)
            sequence_labels.append(label_path)
        
        # Create sequences with seq_length frames
        sequences = []
        for i in range(0, len(sequence_frames) - self.seq_length + 1):
            seq_frames = sequence_frames[i:i + self.seq_length]
            seq_labels = sequence_labels[i:i + self.seq_length]
            
            sequences.append({
                'frames': seq_frames,
                'labels': seq_labels,
                'video_name': video_name,
                'frame_indices': list(range(i, i + self.seq_length))
            })
        
        print(f"Created {len(sequences)} sequences from {frames_dir} and {labels_dir}")
        return sequences
        
    def _load_image(self, path: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            path: Path to the image file
            
        Returns:
            Preprocessed image as torch.Tensor
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            RuntimeError: If there's an error processing the image
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
            
        try:
            img = Image.open(path).convert('RGB')
            
            # Apply transforms if available
            if self.transform:
                img = self.transform(img)
            else:
                img = F.to_tensor(img)
                
            return img
        except Exception as e:
            raise RuntimeError(f"Error loading image {path}: {str(e)}")
            
    def _load_label(self, path: str) -> Union[torch.Tensor, Dict]:
        """
        Load a label from path based on task type.
        
        Args:
            path: Path to label file
            
        Returns:
            Label tensor or dictionary depending on task type
            
        Raises:
            FileNotFoundError: If the label file doesn't exist
            RuntimeError: If there's an error processing the label
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Label file not found: {path}")
            
        try:
            if self.task_type == 'detection':
                # Parse detection labels in YOLO format
                with open(path, 'r') as f:
                    lines = f.readlines()
                
                boxes = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # [class_id, x, y, w, h]
                        box = [float(p) for p in parts[:5]]
                        boxes.append(box)
                
                return torch.tensor(boxes) if boxes else torch.zeros(0, 5)
                
            elif self.task_type == 'segmentation':
                # Load segmentation mask
                with Image.open(path) as mask_pil:
                    mask = torch.from_numpy(np.array(mask_pil))
                    # Resize if needed
                    if mask.shape[0] != 256 or mask.shape[1] != 256:
                        mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(
                            transforms.ToPILImage()(mask.unsqueeze(0).float())
                        )
                        mask = torch.from_numpy(np.array(mask))
                    return mask
                    
            elif self.task_type == 'tracking':
                # Parse tracking labels
                with open(path, 'r') as f:
                    lines = f.readlines()
                
                tracks = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        # [track_id, class_id, x, y, w, h]
                        track = [float(p) for p in parts[:6]]
                        tracks.append(track)
                
                return torch.tensor(tracks) if tracks else torch.zeros(0, 6)
                
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading label {path}: {str(e)}")


def get_dataloader(
    dataset_name: str, 
    batch_size: int,
    task_type: str = 'detection',
    split: str = 'train',
    num_workers: int = 4,
    seq_length: int = 5,
    random_qp: bool = False,
    qp_range: Tuple[int, int] = (22, 37),
    shuffle: bool = True,
    transform = None
) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset or path to dataset directory
        batch_size: Batch size for the dataloader
        task_type: Type of task ('detection', 'segmentation', 'tracking')
        split: Dataset split ('train', 'val', 'test')
        num_workers: Number of worker processes for data loading
        seq_length: Number of frames in each sequence
        random_qp: Whether to use random QP values
        qp_range: Range of QP values if random_qp is True
        shuffle: Whether to shuffle the dataset
        transform: Optional transforms to apply to the images
        
    Returns:
        DataLoader for the specified dataset
        
    Raises:
        RuntimeError: If the dataset directory doesn't exist or no valid sequences were found
        ValueError: If an unsupported dataset name or task type is provided
    """
    print(f"Creating dataloader for {dataset_name} dataset, {task_type} task, {split} split")
    
    # Apply default transforms if none provided
    if transform is None:
        transform = get_transforms(task_type)
    
    try:
        # First, check if the expected directory structure exists
        split_dir = os.path.join(dataset_name, task_type, split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            print(f"Generating a dummy dataset at {split_dir}")
            
            # Create temporary directory for dummy dataset
            temp_output_dir = Path(dataset_name)
            
            # Generate dummy dataset
            dummy_generator = DummyDatasetGenerator(
                output_root=temp_output_dir,
                seq_length=seq_length,
                num_sequences=5,
                frames_per_sequence=30,
                task_type=task_type,
                split=split
            )
            dummy_generator.generate()
            
            print(f"Using dummy {task_type} dataset for {split} split")
            
        # Create dataset object
        dataset = VideoSequenceDataset(
            root_dir=dataset_name,
            task_type=task_type,
            split=split,
            seq_length=seq_length,
            transform=transform,
            random_qp=random_qp,
            qp_range=qp_range
        )
    except RuntimeError as e:
        print(f"Error creating dataset: {str(e)}")
        print("Falling back to dummy dataset...")
        
        # Create temporary directory for dummy dataset
        temp_output_dir = Path(dataset_name)
        
        # Generate dummy dataset
        dummy_generator = DummyDatasetGenerator(
            output_root=temp_output_dir,
            seq_length=seq_length,
            num_sequences=5,
            frames_per_sequence=30,
            task_type=task_type,
            split=split
        )
        dummy_generator.generate()
        
        # Try creating the dataset again with the dummy data
        try:
            dataset = VideoSequenceDataset(
                root_dir=dataset_name,
                task_type=task_type,
                split=split,
                seq_length=seq_length,
                transform=transform,
                random_qp=random_qp,
                qp_range=qp_range
            )
        except Exception as e2:
            raise RuntimeError(f"Failed to create dataset even with dummy data: {str(e2)}")
    
    # Create DataLoader with appropriate collate function
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Created dataloader with {len(dataset)} sequences, batch size {batch_size}")
    return loader


class MOT16DataAdapter:
    """
    Adapter for MOT16 dataset.
    Converts MOT16 format to the format expected by task-aware video compression model.
    """
    def __init__(
        self,
        mot_root: str,
        output_root: str,
        seq_length: int = 5,
        split: str = "train",
        stride: int = 1,
    ):
        """
        Initialize MOT16 dataset adapter.
        
        Args:
            mot_root: Path to MOT16 dataset
            output_root: Path to output directory
            seq_length: Number of frames in each sequence
            split: Data split ('train' or 'test')
            stride: Stride for frame sampling
        """
        self.mot_root = Path(mot_root)
        self.output_root = Path(output_root)
        self.seq_length = seq_length
        self.split = split
        self.stride = stride
        
        # Output directories
        self.output_frames_dir = self.output_root / "tracking" / split / "frames"
        self.output_labels_dir = self.output_root / "tracking" / split / "labels"
        
        # Create output directories
        self.output_frames_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all sequences - try original expected structure first
        self.sequences = list((self.mot_root / split).glob("MOT16-*"))
        
        # If no sequences found, try alternative structures
        if not self.sequences:
            print(f"No sequences found in {self.mot_root / split}")
            
            # Try finding sequences directly in the root directory
            direct_sequences = list(self.mot_root.glob("MOT16-*"))
            if direct_sequences:
                print(f"Found {len(direct_sequences)} sequences directly in {self.mot_root}")
                self.sequences = direct_sequences
            else:
                # Try find sequences in 'train' and 'test' subdirectories regardless of specified split
                for alt_split in ['train', 'test']:
                    alt_sequences = list((self.mot_root / alt_split).glob("MOT16-*"))
                    if alt_sequences:
                        print(f"Found {len(alt_sequences)} sequences in alternative split {alt_split}")
                        self.sequences = alt_sequences
                        self.split = alt_split  # Update split to match what was found
                        break
        
        if not self.sequences:
            print(f"No MOT16 sequences found in {self.mot_root} or its subdirectories")
            return
            
        print(f"Found {len(self.sequences)} MOT16 sequences in {self.split} split")
    
    def convert(self):
        """
        Convert MOT16 dataset to the expected format.
        """
        for seq_path in self.sequences:
            seq_name = seq_path.name
            print(f"Processing sequence {seq_name}...")
            
            # Create output directories for this sequence
            seq_frames_dir = self.output_frames_dir / seq_name
            seq_labels_dir = self.output_labels_dir / seq_name
            seq_frames_dir.mkdir(exist_ok=True)
            seq_labels_dir.mkdir(exist_ok=True)
            
            # Find all frame images
            img_dir = seq_path / "img1"
            frames = sorted(list(img_dir.glob("*.jpg")))
            
            if not frames:
                print(f"No frames found in {img_dir}")
                continue
                
            print(f"Found {len(frames)} frames in {seq_name}")
            
            # Parse ground truth
            gt_file = seq_path / "gt" / "gt.txt"
            frame_annots = {}
            
            if gt_file.exists():
                with open(gt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 7:
                            continue
                        
                        frame_id = int(parts[0])
                        track_id = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        w = float(parts[4])
                        h = float(parts[5])
                        conf = float(parts[6])
                        class_id = int(parts[7]) if len(parts) > 7 else 0  # Default class ID
                        
                        # Skip if confidence is 0
                        if conf == 0:
                            continue
                        
                        if frame_id not in frame_annots:
                            frame_annots[frame_id] = []
                        
                        # Convert to YOLO format: class_id, track_id, x_center, y_center, width, height
                        # in normalized coordinates
                        frame_annots[frame_id].append([class_id, track_id, x, y, w, h])
            else:
                print(f"No ground truth file found at {gt_file}")
            
            # Process frame sequences
            for i in range(0, len(frames) - self.seq_length * self.stride + 1, self.stride):
                seq_id = i // self.stride
                
                # Extract sequence frames
                seq_frames = frames[i:i + self.seq_length * self.stride:self.stride]
                
                if len(seq_frames) < self.seq_length:
                    continue
                
                # Get frame IDs from filenames
                frame_ids = [int(f.stem) for f in seq_frames]
                
                # Copy frames to output directory
                for idx, (frame_path, frame_id) in enumerate(zip(seq_frames, frame_ids)):
                    dst_path = seq_frames_dir / f"{seq_id:06d}_{idx:02d}.jpg"
                    try:
                        shutil.copy(frame_path, dst_path)
                    except Exception as e:
                        print(f"Error copying frame {frame_path} to {dst_path}: {e}")
                
                # Create annotation files in YOLO format for tracking
                for idx, frame_id in enumerate(frame_ids):
                    label_path = seq_labels_dir / f"{seq_id:06d}_{idx:02d}.txt"
                    
                    if frame_id in frame_annots:
                        # Get image dimensions for normalization
                        img = Image.open(seq_frames[idx])
                        img_width, img_height = img.size
                        
                        with open(label_path, 'w') as f:
                            for annot in frame_annots[frame_id]:
                                class_id, track_id, x, y, w, h = annot
                                
                                # Convert to normalized coordinates
                                x_center = (x + w/2) / img_width
                                y_center = (y + h/2) / img_height
                                norm_w = w / img_width
                                norm_h = h / img_height
                                
                                # Write in YOLO format with track_id
                                f.write(f"{class_id} {track_id} {x_center} {y_center} {norm_w} {norm_h}\n")
                    else:
                        # Create empty label file if no annotations
                        with open(label_path, 'w') as f:
                            pass
            
            print(f"Processed sequence {seq_name}")
        
        print("Conversion completed successfully")
    
    def verify(self):
        """
        Verify the converted dataset.
        """
        # Check that the expected directories and files exist
        tracking_dir = self.output_root / "tracking" / self.split
        frames_dir = tracking_dir / "frames"
        labels_dir = tracking_dir / "labels"
        
        if not frames_dir.exists() or not labels_dir.exists():
            print(f"Error: Missing directories {frames_dir} or {labels_dir}")
            return False
        
        sequences = [d for d in frames_dir.iterdir() if d.is_dir()]
        
        if not sequences:
            print(f"Error: No sequences found in {frames_dir}")
            return False
        
        success = True
        for seq_dir in sequences:
            seq_name = seq_dir.name
            
            # Check for corresponding label directory
            label_dir = labels_dir / seq_name
            if not label_dir.exists():
                print(f"Error: Missing label directory for sequence {seq_name}")
                success = False
                continue
            
            # Count frames and labels
            frame_files = list(seq_dir.glob("*.jpg"))
            label_files = list(label_dir.glob("*.txt"))
            
            if len(frame_files) != len(label_files):
                print(f"Warning: Mismatch in sequence {seq_name}: {len(frame_files)} frames, {len(label_files)} labels")
            
            if not frame_files:
                print(f"Error: No frames found in sequence {seq_name}")
                success = False
            
            print(f"Sequence {seq_name}: {len(frame_files)} frames, {len(label_files)} labels")
        
        if success:
            print("Dataset verification passed")
        else:
            print("Dataset verification failed")
        
        return success


class DummyDatasetGenerator:
    """
    Creates a dummy dataset for tracking tasks in the expected directory structure.
    """
    def __init__(
        self,
        output_root: str,
        seq_length: int = 5,
        num_sequences: int = 10,
        frames_per_sequence: int = 30,
        image_size: Tuple[int, int] = (256, 256),
        task_type: str = 'tracking',
        split: str = 'train'
    ):
        """
        Initialize dummy dataset generator.
        
        Args:
            output_root: Path to output directory
            seq_length: Number of frames in each sequence
            num_sequences: Number of sequences to generate
            frames_per_sequence: Number of frames in each video sequence
            image_size: Size of generated images (H, W)
            task_type: Type of task ('detection', 'segmentation', 'tracking')
            split: Data split ('train', 'val', 'test')
        """
        self.output_root = Path(output_root)
        self.seq_length = seq_length
        self.num_sequences = num_sequences
        self.frames_per_sequence = frames_per_sequence
        self.image_size = image_size
        self.task_type = task_type
        self.split = split
        
        # Output directories
        self.task_dir = self.output_root / task_type
        self.split_dir = self.task_dir / split
        
        # Create directory structure
        self.split_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self):
        """
        Generate dummy dataset with the expected directory structure.
        """
        print(f"Generating dummy {self.task_type} dataset for {self.split} split...")
        
        for seq_idx in range(self.num_sequences):
            # Create sequence directory
            seq_name = f"sequence_{seq_idx:02d}"
            seq_dir = self.split_dir / seq_name
            frames_dir = seq_dir / "frames"
            
            # Determine labels directory based on task type
            if self.task_type == 'detection':
                labels_dir = seq_dir / "labels"
            elif self.task_type == 'segmentation':
                labels_dir = seq_dir / "masks"
            elif self.task_type == 'tracking':
                labels_dir = seq_dir / "tracks"
            else:
                labels_dir = seq_dir / "labels"  # Default
            
            # Create directories
            frames_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate frames and labels
            for frame_idx in range(self.frames_per_sequence):
                # Generate dummy image (black with frame number)
                img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
                # Add some color and text to make it visually distinguishable
                cv2.rectangle(img, (50, 50), (self.image_size[1]-50, self.image_size[0]-50), 
                              (0, 100, 200), 2)
                cv2.putText(img, f"Seq {seq_idx:02d}, Frame {frame_idx:03d}", 
                           (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save image
                img_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(img_path), img)
                
                # Generate dummy labels based on task type
                label_path = labels_dir / f"frame_{frame_idx:06d}.txt"
                
                with open(label_path, 'w') as f:
                    # Generate random number of objects (1-5)
                    num_objects = random.randint(1, 5)
                    
                    for obj_idx in range(num_objects):
                        if self.task_type == 'tracking':
                            # Tracking format: class_id track_id x_center y_center width height
                            class_id = random.randint(0, 2)  # Random class ID
                            track_id = obj_idx  # Consistent track ID
                            x_center = random.uniform(0.2, 0.8)
                            y_center = random.uniform(0.2, 0.8)
                            width = random.uniform(0.05, 0.2)
                            height = random.uniform(0.05, 0.2)
                            f.write(f"{class_id} {track_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        elif self.task_type == 'detection':
                            # Detection format: class_id x_center y_center width height
                            class_id = random.randint(0, 2)  # Random class ID
                            x_center = random.uniform(0.2, 0.8)
                            y_center = random.uniform(0.2, 0.8)
                            width = random.uniform(0.05, 0.2)
                            height = random.uniform(0.05, 0.2)
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        elif self.task_type == 'segmentation':
                            # For segmentation, just create a dummy text file
                            # (we would normally create a segmentation mask image)
                            f.write(f"Dummy segmentation for frame {frame_idx}\n")
            
            print(f"Generated sequence {seq_name} with {self.frames_per_sequence} frames")
        
        print(f"Generated dummy {self.task_type} dataset with {self.num_sequences} sequences")
        return self.output_root


# Test code
if __name__ == "__main__":
    # Test dummy dataset
    dummy_dataset = DummyVideoDataset(
        num_samples=10,
        clip_len=8,
        resolution=(256, 256),
        task_type="detection"
    )
    
    sample = dummy_dataset[0]
    print(f"Sample frames shape: {sample['frames'].shape}")
    print(f"Sample boxes shape: {sample['boxes'][0].shape}")
    
    # Test dataloader
    dataloader = get_video_dataloader(
        dummy_dataset,
        task_type="detection",
        batch_size=2
    )
    
    for batch in dataloader:
        print(f"Batch frames shape: {batch['frames'].shape}")
        print(f"Batch boxes: {len(batch['boxes'])} samples with {len(batch['boxes'][0])} frames each")
        break 