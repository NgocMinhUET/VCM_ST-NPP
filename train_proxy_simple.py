import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
from tqdm import tqdm

class VideoDataset(Dataset):
    """Dataset for loading video sequences."""
    
    def __init__(self, dataset_path, time_steps=16, transform=None, max_videos=None, frame_stride=4):
        """
        Initialize the VideoDataset.
        
        Args:
            dataset_path: Path to the directory containing video files
            time_steps: Number of frames in each sequence
            transform: Optional transform to apply to the frames
            max_videos: Maximum number of videos to load (for debugging)
            frame_stride: Stride for frame sampling
        """
        self.dataset_path = Path(dataset_path)
        self.time_steps = time_steps
        self.transform = transform
        self.frame_stride = frame_stride
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        self.video_files = []
        for ext in video_extensions:
            self.video_files.extend(list(self.dataset_path.glob(f'**/*{ext}')))
        
        # Check if any video files were found
        if len(self.video_files) == 0:
            raise ValueError(f"No video files found in {dataset_path}. Make sure the path exists and contains video files with extensions: {video_extensions}")
        
        # Limit the number of videos if specified
        if max_videos is not None:
            self.video_files = self.video_files[:max_videos]
        
        # Extract frames from videos and create sequences
        self.sequences = []
        for video_file in tqdm(self.video_files, desc="Loading videos"):
            self._extract_sequences(video_file)
        
        # Check if any sequences were created
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences could be created from the videos in {dataset_path}. Check that the videos have at least {time_steps} frames and are readable.")
    
    def _extract_sequences(self, video_file):
        """Extract frame sequences from a video file."""
        cap = cv2.VideoCapture(str(video_file))
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to a fixed size for consistency
            frame = cv2.resize(frame, (224, 224))
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
        
        cap.release()
        
        # Create sequences with stride
        if len(frames) >= self.time_steps:
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

def parse_args():
    parser = argparse.ArgumentParser(description="Test Video Dataset Loading")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the video dataset")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames in each sequence")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Print device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = VideoDataset(
        dataset_path=args.dataset_path,
        time_steps=args.time_steps
    )
    
    print(f"Dataset loaded with {len(dataset)} sequences")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # Print sequence information
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}: Shape = {batch.shape}")
        if i >= 2:  # Just print a few batches
            break
    
    print("Video loading test completed successfully!")

if __name__ == "__main__":
    main() 