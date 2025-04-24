import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import glob
from collections import defaultdict

# Transforms for video frames
class FrameTransform:
    """
    Transform class for video frames
    """
    def __init__(self, img_size=(224, 224), normalize=True):
        """
        Initialize transform
        
        Args:
            img_size (tuple): Target image size (height, width)
            normalize (bool): Whether to normalize the images
        """
        self.img_size = img_size
        self.normalize = normalize
        
        # Define transforms
        transform_list = [
            transforms.Resize(img_size),
            transforms.ToTensor()
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
            
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, frame):
        """
        Apply transform to a single frame
        
        Args:
            frame: Input frame (numpy array or PIL Image)
            
        Returns:
            torch.Tensor: Transformed frame
        """
        if isinstance(frame, np.ndarray):
            # Convert numpy array to PIL Image
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        return self.transform(frame)


class VideoDataset(Dataset):
    """
    Base dataset class for video clip loading
    """
    def __init__(self, video_paths, clip_len=16, frame_stride=1, transform=None):
        """
        Initialize dataset
        
        Args:
            video_paths (list): List of video file paths
            clip_len (int): Number of frames per clip
            frame_stride (int): Stride between frames
            transform: Transforms to apply to frames
        """
        self.video_paths = video_paths
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.transform = transform or FrameTransform()
    
    def __len__(self):
        return len(self.video_paths)
    
    def read_video(self, video_path):
        """
        Read video file using OpenCV
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            list: List of frames (numpy arrays)
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return frames
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames
    
    def get_clip_frames(self, video_frames):
        """
        Extract clip from video frames
        
        Args:
            video_frames (list): List of video frames
            
        Returns:
            list: List of clip frames
        """
        # Ensure we have enough frames
        if len(video_frames) < self.clip_len * self.frame_stride:
            # If not enough frames, duplicate the last frame
            video_frames.extend([video_frames[-1]] * (self.clip_len * self.frame_stride - len(video_frames)))
        
        # Randomly select a starting point for the clip
        start_idx = np.random.randint(0, len(video_frames) - self.clip_len * self.frame_stride + 1)
        
        # Extract frames with the given stride
        clip_frames = [video_frames[start_idx + i * self.frame_stride] for i in range(self.clip_len)]
        
        return clip_frames
    
    def __getitem__(self, idx):
        """
        Get a video clip
        
        Args:
            idx (int): Index of the video
            
        Returns:
            dict: Dictionary containing clip tensor
        """
        video_path = self.video_paths[idx]
        
        # Read video frames
        video_frames = self.read_video(video_path)
        
        # Get clip frames
        clip_frames = self.get_clip_frames(video_frames)
        
        # Apply transforms to each frame
        clip_tensors = [self.transform(frame) for frame in clip_frames]
        
        # Stack tensors to create a clip tensor [C, T, H, W]
        clip_tensor = torch.stack(clip_tensors, dim=1)
        
        return {'video': clip_tensor, 'path': video_path}


class MOTDataset(Dataset):
    """
    Dataset class for Multi-Object Tracking (MOT) data
    """
    def __init__(self, root_dir, sequence_names=None, clip_len=16, frame_stride=1, transform=None, target_size=(224, 224)):
        """
        Initialize MOT dataset
        
        Args:
            root_dir (str): Root directory of MOT dataset
            sequence_names (list): List of sequence names to include
            clip_len (int): Number of frames per clip
            frame_stride (int): Stride between frames
            transform: Transforms to apply to frames
            target_size (tuple): Target frame size (width, height)
        """
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.transform = transform or FrameTransform(img_size=target_size)
        self.target_size = target_size
        
        # Find all sequences in the dataset
        self.sequences = []
        
        if sequence_names is None:
            # Use all sequences in the train folder
            sequence_dirs = [os.path.join(root_dir, 'train', d) for d in os.listdir(os.path.join(root_dir, 'train'))
                           if os.path.isdir(os.path.join(root_dir, 'train', d))]
        else:
            # Use specified sequences
            sequence_dirs = [os.path.join(root_dir, 'train', name) for name in sequence_names
                           if os.path.isdir(os.path.join(root_dir, 'train', name))]
        
        # Process each sequence
        for seq_dir in sequence_dirs:
            seq_name = os.path.basename(seq_dir)
            
            # Get frames
            frame_dir = os.path.join(seq_dir, 'img1')
            frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
            
            # Get ground truth
            gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
            detections = self._parse_mot_gt(gt_file)
            
            # Create clips
            for i in range(0, len(frame_paths) - self.clip_len * self.frame_stride + 1, self.clip_len):
                # Get frame paths for this clip
                clip_frame_paths = [frame_paths[i + j * self.frame_stride] for j in range(self.clip_len)]
                
                # Get frame numbers
                frame_nums = [int(os.path.basename(path).split('.')[0]) for path in clip_frame_paths]
                
                # Get detections for these frames
                clip_detections = {frame_num: detections.get(frame_num, []) for frame_num in frame_nums}
                
                self.sequences.append({
                    'sequence': seq_name,
                    'frame_paths': clip_frame_paths,
                    'frame_nums': frame_nums,
                    'detections': clip_detections
                })
    
    def __len__(self):
        return len(self.sequences)
    
    def _parse_mot_gt(self, gt_file):
        """
        Parse MOT ground truth file
        
        Args:
            gt_file (str): Path to ground truth file
            
        Returns:
            dict: Dictionary mapping frame numbers to lists of detections
        """
        detections = defaultdict(list)
        
        with open(gt_file, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                frame = int(data[0])
                track_id = int(data[1])
                x = float(data[2])
                y = float(data[3])
                w = float(data[4])
                h = float(data[5])
                confidence = float(data[6])
                class_id = int(data[7])
                visibility = float(data[8]) if len(data) > 8 else 1.0
                
                # Skip boxes with 0 confidence (ignored regions)
                if confidence == 0:
                    continue
                
                detections[frame].append({
                    'track_id': track_id,
                    'box': [x, y, w, h],  # [left, top, width, height]
                    'class_id': class_id,
                    'visibility': visibility
                })
        
        return detections
    
    def _resize_bbox(self, bbox, orig_size, target_size):
        """
        Resize bounding box coordinates
        
        Args:
            bbox (list): Bounding box [x, y, w, h]
            orig_size (tuple): Original image size (width, height)
            target_size (tuple): Target image size (width, height)
            
        Returns:
            list: Resized bounding box
        """
        x, y, w, h = bbox
        
        # Calculate resize factors
        w_factor = target_size[0] / orig_size[0]
        h_factor = target_size[1] / orig_size[1]
        
        # Resize coordinates
        x_new = x * w_factor
        y_new = y * h_factor
        w_new = w * w_factor
        h_new = h * h_factor
        
        return [x_new, y_new, w_new, h_new]
    
    def __getitem__(self, idx):
        """
        Get a video clip with tracking annotations
        
        Args:
            idx (int): Index of the sequence
            
        Returns:
            dict: Dictionary containing clip tensor and tracking annotations
        """
        seq_data = self.sequences[idx]
        
        # Read frames
        frames = []
        for frame_path in seq_data['frame_paths']:
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Error: Could not read image {frame_path}")
                # Create a blank frame
                frame = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            frames.append(frame)
        
        # Get original size of first frame
        orig_size = (frames[0].shape[1], frames[0].shape[0])  # (width, height)
        
        # Apply transforms to each frame
        clip_tensors = [self.transform(frame) for frame in frames]
        
        # Stack tensors to create a clip tensor [C, T, H, W]
        clip_tensor = torch.stack(clip_tensors, dim=1)
        
        # Process detections for each frame
        boxes = []
        track_ids = []
        class_ids = []
        
        for i, frame_num in enumerate(seq_data['frame_nums']):
            frame_boxes = []
            frame_track_ids = []
            frame_class_ids = []
            
            for det in seq_data['detections'][frame_num]:
                # Resize bounding box to target size
                resized_box = self._resize_bbox(det['box'], orig_size, self.target_size)
                
                # Convert [x, y, w, h] to [x1, y1, x2, y2]
                x, y, w, h = resized_box
                box = [x, y, x + w, y + h]
                
                frame_boxes.append(box)
                frame_track_ids.append(det['track_id'])
                frame_class_ids.append(det['class_id'])
            
            boxes.append(frame_boxes)
            track_ids.append(frame_track_ids)
            class_ids.append(frame_class_ids)
        
        return {
            'video': clip_tensor,
            'boxes': boxes,
            'track_ids': track_ids,
            'class_ids': class_ids,
            'frame_ids': seq_data['frame_nums'],
            'sequence': seq_data['sequence']
        }


class KITTIDataset(Dataset):
    """
    Dataset class for KITTI data for object detection
    """
    def __init__(self, root_dir, split='training', clip_len=16, frame_stride=1, transform=None, target_size=(224, 224)):
        """
        Initialize KITTI dataset
        
        Args:
            root_dir (str): Root directory of KITTI dataset
            split (str): 'training' or 'testing'
            clip_len (int): Number of frames per clip
            frame_stride (int): Stride between frames
            transform: Transforms to apply to frames
            target_size (tuple): Target frame size (width, height)
        """
        self.root_dir = root_dir
        self.split = split
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.transform = transform or FrameTransform(img_size=target_size)
        self.target_size = target_size
        
        # Find all sequences in the dataset
        self.sequences = []
        
        # For KITTI, sequences are organized in directories by date
        date_dirs = [d for d in os.listdir(os.path.join(root_dir, split, 'image_02'))
                   if os.path.isdir(os.path.join(root_dir, split, 'image_02', d))]
        
        for date_dir in date_dirs:
            # Each date directory contains multiple sequences
            seq_dirs = [d for d in os.listdir(os.path.join(root_dir, split, 'image_02', date_dir))
                      if os.path.isdir(os.path.join(root_dir, split, 'image_02', date_dir, d))]
            
            for seq_dir in seq_dirs:
                # Get frames
                frame_dir = os.path.join(root_dir, split, 'image_02', date_dir, seq_dir)
                frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
                
                # Get ground truth if available (for training split)
                label_dir = os.path.join(root_dir, split, 'label_02', date_dir)
                label_file = os.path.join(label_dir, f'{seq_dir}.txt') if os.path.exists(label_dir) else None
                
                detections = self._parse_kitti_labels(label_file) if label_file and os.path.exists(label_file) else {}
                
                # Create clips
                for i in range(0, len(frame_paths) - self.clip_len * self.frame_stride + 1, self.clip_len):
                    # Get frame paths for this clip
                    clip_frame_paths = [frame_paths[i + j * self.frame_stride] for j in range(self.clip_len)]
                    
                    # Get frame numbers
                    frame_nums = [int(os.path.basename(path).split('.')[0]) for path in clip_frame_paths]
                    
                    # Get detections for these frames
                    clip_detections = {frame_num: detections.get(frame_num, []) for frame_num in frame_nums}
                    
                    self.sequences.append({
                        'sequence': f'{date_dir}_{seq_dir}',
                        'frame_paths': clip_frame_paths,
                        'frame_nums': frame_nums,
                        'detections': clip_detections
                    })
    
    def __len__(self):
        return len(self.sequences)
    
    def _parse_kitti_labels(self, label_file):
        """
        Parse KITTI label file
        
        Args:
            label_file (str): Path to label file
            
        Returns:
            dict: Dictionary mapping frame numbers to lists of detections
        """
        if label_file is None or not os.path.exists(label_file):
            return {}
            
        detections = defaultdict(list)
        
        with open(label_file, 'r') as f:
            for line in f:
                data = line.strip().split(' ')
                frame = int(data[0])
                track_id = int(data[1])
                obj_type = data[2]
                truncated = float(data[3])
                occluded = int(data[4])
                alpha = float(data[5])
                left = float(data[6])
                top = float(data[7])
                right = float(data[8])
                bottom = float(data[9])
                
                # Skip non-relevant object types
                if obj_type in ['DontCare', 'Misc']:
                    continue
                
                # Map object type to class ID
                class_id_map = {
                    'Car': 1,
                    'Van': 2,
                    'Truck': 3,
                    'Pedestrian': 4,
                    'Person_sitting': 5,
                    'Cyclist': 6,
                    'Tram': 7,
                    'Misc': 8,
                    'DontCare': 9
                }
                
                class_id = class_id_map.get(obj_type, 0)
                
                # Convert to [x, y, w, h] format
                width = right - left
                height = bottom - top
                
                detections[frame].append({
                    'track_id': track_id,
                    'box': [left, top, width, height],
                    'class_id': class_id,
                    'obj_type': obj_type,
                    'truncated': truncated,
                    'occluded': occluded,
                    'alpha': alpha
                })
        
        return detections
    
    def _resize_bbox(self, bbox, orig_size, target_size):
        """
        Resize bounding box coordinates
        
        Args:
            bbox (list): Bounding box [x, y, w, h]
            orig_size (tuple): Original image size (width, height)
            target_size (tuple): Target image size (width, height)
            
        Returns:
            list: Resized bounding box
        """
        x, y, w, h = bbox
        
        # Calculate resize factors
        w_factor = target_size[0] / orig_size[0]
        h_factor = target_size[1] / orig_size[1]
        
        # Resize coordinates
        x_new = x * w_factor
        y_new = y * h_factor
        w_new = w * w_factor
        h_new = h * h_factor
        
        return [x_new, y_new, w_new, h_new]
    
    def __getitem__(self, idx):
        """
        Get a video clip with object detection annotations
        
        Args:
            idx (int): Index of the sequence
            
        Returns:
            dict: Dictionary containing clip tensor and detection annotations
        """
        seq_data = self.sequences[idx]
        
        # Read frames
        frames = []
        for frame_path in seq_data['frame_paths']:
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Error: Could not read image {frame_path}")
                # Create a blank frame
                frame = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            frames.append(frame)
        
        # Get original size of first frame
        orig_size = (frames[0].shape[1], frames[0].shape[0])  # (width, height)
        
        # Apply transforms to each frame
        clip_tensors = [self.transform(frame) for frame in frames]
        
        # Stack tensors to create a clip tensor [C, T, H, W]
        clip_tensor = torch.stack(clip_tensors, dim=1)
        
        # Process detections for each frame
        boxes = []
        class_ids = []
        
        for i, frame_num in enumerate(seq_data['frame_nums']):
            frame_boxes = []
            frame_class_ids = []
            
            for det in seq_data['detections'][frame_num]:
                # Resize bounding box to target size
                resized_box = self._resize_bbox(det['box'], orig_size, self.target_size)
                
                # Convert [x, y, w, h] to [x1, y1, x2, y2]
                x, y, w, h = resized_box
                box = [x, y, x + w, y + h]
                
                frame_boxes.append(box)
                frame_class_ids.append(det['class_id'])
            
            boxes.append(frame_boxes)
            class_ids.append(frame_class_ids)
        
        return {
            'video': clip_tensor,
            'boxes': boxes,
            'class_ids': class_ids,
            'frame_ids': seq_data['frame_nums'],
            'sequence': seq_data['sequence']
        }


class DummyVideoDataset(Dataset):
    """
    Dummy dataset class for testing
    """
    def __init__(self, num_samples=100, clip_len=16, height=224, width=224, num_channels=3):
        """
        Initialize dummy dataset
        
        Args:
            num_samples (int): Number of samples in the dataset
            clip_len (int): Number of frames per clip
            height (int): Frame height
            width (int): Frame width
            num_channels (int): Number of channels
        """
        self.num_samples = num_samples
        self.clip_len = clip_len
        self.height = height
        self.width = width
        self.num_channels = num_channels
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a random video clip
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing clip tensor and random annotations
        """
        # Create random clip tensor
        clip = torch.rand(self.num_channels, self.clip_len, self.height, self.width)
        
        # Create random bounding boxes
        boxes = []
        class_ids = []
        
        for _ in range(self.clip_len):
            num_boxes = np.random.randint(1, 5)
            
            frame_boxes = []
            frame_class_ids = []
            
            for _ in range(num_boxes):
                # Random box coordinates (x1, y1, x2, y2)
                x1 = np.random.randint(0, self.width - 50)
                y1 = np.random.randint(0, self.height - 50)
                x2 = np.random.randint(x1 + 10, min(x1 + 100, self.width))
                y2 = np.random.randint(y1 + 10, min(y1 + 100, self.height))
                
                frame_boxes.append([x1, y1, x2, y2])
                frame_class_ids.append(np.random.randint(1, 10))
            
            boxes.append(frame_boxes)
            class_ids.append(frame_class_ids)
        
        return {
            'video': clip,
            'boxes': boxes,
            'class_ids': class_ids,
            'frame_ids': list(range(self.clip_len)),
            'sequence': f'dummy_{idx}'
        }


def collate_video_detection(batch):
    """
    Collate function for video object detection data
    
    Args:
        batch (list): List of samples
        
    Returns:
        dict: Collated batch
    """
    videos = torch.stack([sample['video'] for sample in batch], dim=0)
    boxes = [sample['boxes'] for sample in batch]
    class_ids = [sample['class_ids'] for sample in batch]
    frame_ids = [sample['frame_ids'] for sample in batch]
    sequences = [sample['sequence'] for sample in batch]
    
    return {
        'video': videos,
        'boxes': boxes,
        'class_ids': class_ids,
        'frame_ids': frame_ids,
        'sequences': sequences
    }


def collate_video_segmentation(batch):
    """
    Collate function for video segmentation data
    
    Args:
        batch (list): List of samples
        
    Returns:
        dict: Collated batch
    """
    videos = torch.stack([sample['video'] for sample in batch], dim=0)
    masks = [sample['masks'] for sample in batch]
    frame_ids = [sample['frame_ids'] for sample in batch]
    sequences = [sample['sequence'] for sample in batch]
    
    return {
        'video': videos,
        'masks': masks,
        'frame_ids': frame_ids,
        'sequences': sequences
    }


def collate_video_tracking(batch):
    """
    Collate function for video tracking data
    
    Args:
        batch (list): List of samples
        
    Returns:
        dict: Collated batch
    """
    videos = torch.stack([sample['video'] for sample in batch], dim=0)
    boxes = [sample['boxes'] for sample in batch]
    track_ids = [sample['track_ids'] for sample in batch]
    class_ids = [sample['class_ids'] for sample in batch]
    frame_ids = [sample['frame_ids'] for sample in batch]
    sequences = [sample['sequence'] for sample in batch]
    
    return {
        'video': videos,
        'boxes': boxes,
        'track_ids': track_ids,
        'class_ids': class_ids,
        'frame_ids': frame_ids,
        'sequences': sequences
    }


def get_video_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4, task_type='detection'):
    """
    Get dataloader for video dataset
    
    Args:
        dataset: Video dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker processes
        task_type (str): Task type ('detection', 'segmentation', or 'tracking')
        
    Returns:
        DataLoader: DataLoader for the dataset
    """
    # Choose appropriate collate function based on task type
    if task_type == 'detection':
        collate_fn = collate_video_detection
    elif task_type == 'segmentation':
        collate_fn = collate_video_segmentation
    elif task_type == 'tracking':
        collate_fn = collate_video_tracking
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


# Test code
if __name__ == "__main__":
    # Test DummyVideoDataset
    dummy_dataset = DummyVideoDataset(num_samples=10, clip_len=8)
    
    for i in range(2):
        sample = dummy_dataset[i]
        print(f"Sample {i}:")
        print(f"Video shape: {sample['video'].shape}")
        print(f"Number of frames: {len(sample['boxes'])}")
        print(f"First frame boxes: {sample['boxes'][0]}")
        print(f"First frame class IDs: {sample['class_ids'][0]}")
        print()
    
    # Test dataloader
    dataloader = get_video_dataloader(dummy_dataset, batch_size=2, task_type='detection')
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Videos shape: {batch['video'].shape}")
        print(f"Number of samples: {len(batch['boxes'])}")
        print(f"First sample, first frame boxes: {batch['boxes'][0][0]}")
        print()
        break 