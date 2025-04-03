import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm


class MOTImageSequenceDataset(Dataset):
    """Dataset for loading MOT16 image sequences."""
    
    def __init__(self, dataset_path, time_steps=16, transform=None, split='train', frame_stride=1):
        """
        Initialize the MOTImageSequenceDataset.
        
        Args:
            dataset_path: Path to the MOT16 dataset directory
            time_steps: Number of frames in each sequence
            transform: Optional transform to apply to the frames
            split: 'train' or 'test'
            frame_stride: Stride for frame sampling
        """
        self.dataset_path = Path(dataset_path)
        self.time_steps = time_steps
        self.transform = transform
        self.frame_stride = frame_stride
        self.split = split
        
        # Find all sequence directories
        split_path = self.dataset_path / split
        self.sequence_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        
        if len(self.sequence_dirs) == 0:
            raise ValueError(f"No sequence directories found in {split_path}")
        
        print(f"Found {len(self.sequence_dirs)} sequences in {split_path}")
        
        # Create a list to store all sequences
        self.sequences = []
        
        # Process each sequence directory
        for seq_dir in tqdm(self.sequence_dirs, desc="Loading sequences"):
            self._process_sequence(seq_dir)
        
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences could be created from the images in {split_path}. Check that the sequences have at least {time_steps} frames.")
        
        print(f"Created {len(self.sequences)} sequences from {len(self.sequence_dirs)} MOT directories")
    
    def _process_sequence(self, sequence_dir):
        """Process a single MOT sequence directory."""
        # Find the img1 directory which contains all frames
        img_dir = sequence_dir / 'img1'
        if not img_dir.exists():
            print(f"Warning: img1 directory not found in {sequence_dir}")
            return
        
        # Get all image files in sorted order
        image_files = sorted(list(img_dir.glob('*.jpg')))
        
        if len(image_files) == 0:
            print(f"Warning: No image files found in {img_dir}")
            return
        
        # Load all images in the sequence
        frames = []
        for img_file in image_files:
            # Read the image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Could not read image {img_file}")
                continue
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to a fixed size for consistency
            img = cv2.resize(img, (224, 224))
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            frames.append(img)
        
        # Check if we have enough frames to create sequences
        if len(frames) < self.time_steps:
            print(f"Warning: Sequence {sequence_dir.name} has only {len(frames)} frames, which is less than the required {self.time_steps}")
            return
        
        # Create sequences with stride
        for i in range(0, len(frames) - self.time_steps + 1, self.frame_stride):
            sequence = frames[i:i + self.time_steps]
            self.sequences.append(sequence)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Convert to tensor
        sequence = np.array(sequence)
        sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Apply transform if specified
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence


def test_mot_dataset():
    """Test function to verify the MOT dataset loading."""
    from torch.utils.data import DataLoader
    
    # Set the dataset path
    dataset_path = "D:/NCS/propose/dataset/MOT16"
    
    # Create the dataset
    dataset = MOTImageSequenceDataset(
        dataset_path=dataset_path,
        time_steps=16,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # Print batch information
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}: Shape = {batch.shape}")
        if i >= 2:  # Just print a few batches
            break
    
    print("MOT16 dataset test completed successfully!")


if __name__ == "__main__":
    test_mot_dataset() 