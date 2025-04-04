#!/usr/bin/env python
"""
Training script for the Differentiable Proxy Network.

This script trains the proxy network to approximate the rate-distortion behavior
of the HEVC codec in a differentiable manner.
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import cv2
import subprocess
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.proxy_network import ProxyNetwork
from models.stnpp import STNPP

# Import our MOT dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mot_dataset import MOTImageSequenceDataset


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


def hevc_encode_decode(frames, qp, temp_dir=None):
    """
    Encode and decode frames using HEVC codec.
    
    Args:
        frames: Tensor of frames (T, C, H, W) in range [0, 1]
        qp: Quantization Parameter for HEVC
        temp_dir: Directory for temporary files
    
    Returns:
        Decoded frames tensor (T, C, H, W)
    """
    if temp_dir is None:
        temp_dir = Path('temp_codec')
    
    os.makedirs(temp_dir, exist_ok=True)
    
    # Convert frames to numpy and back to [0, 255] range
    frames_np = frames.permute(0, 2, 3, 1).cpu().numpy() * 255.0
    frames_np = frames_np.astype(np.uint8)
    
    # Save frames as PNG files (lossless)
    frame_paths = []
    for i, frame in enumerate(frames_np):
        frame_path = temp_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
    
    # Create YUV file from frames
    yuv_path = temp_dir / "temp.yuv"
    with open(yuv_path, 'wb') as yuv_file:
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path))
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            yuv_file.write(yuv.tobytes())
    
    # Encode with HEVC
    height, width = frames_np.shape[1:3]
    encoded_path = temp_dir / "encoded.hevc"
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}", "-i", str(yuv_path),
        "-c:v", "libx265", "-preset", "medium",
        "-x265-params", f"qp={qp}", str(encoded_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Decode back to YUV
    decoded_yuv_path = temp_dir / "decoded.yuv"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(encoded_path),
        "-c:v", "rawvideo", "-pix_fmt", "yuv420p",
        str(decoded_yuv_path)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Read decoded YUV back to tensors
    frame_size = width * height * 3 // 2  # YUV420 uses 1.5 bytes per pixel
    decoded_frames = []
    with open(decoded_yuv_path, 'rb') as f:
        for _ in range(len(frames_np)):
            yuv_bytes = f.read(frame_size)
            if not yuv_bytes:
                break
            
            # Create a YUV frame and convert to RGB
            yuv_np = np.frombuffer(yuv_bytes, dtype=np.uint8)
            yuv_np = yuv_np.reshape((height * 3 // 2, width))
            rgb = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_I420)
            
            # Convert to tensor and normalize to [0, 1]
            rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            decoded_frames.append(rgb_tensor)
    
    # Stack frames
    decoded_frames = torch.stack(decoded_frames, dim=0)
    
    # Clean up
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.remove(yuv_path)
    os.remove(encoded_path)
    os.remove(decoded_yuv_path)
    
    return decoded_frames


def train(args):
    """Main training function."""
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create temporary directory for codec operations
    temp_dir = os.path.join(args.output_dir, "temp_codec")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize ST-NPP model (for feature extraction)
    print("Initializing ST-NPP model...")
    stnpp = STNPP(
        input_channels=3,
        output_channels=args.feature_channels,
        spatial_backbone=args.backbone,
        temporal_model=args.temporal_model,
        fusion_type=args.fusion_type,
        pretrained=True
    ).to(device)
    stnpp.eval()  # Set to evaluation mode for feature extraction
    
    # Initialize proxy network
    print("Initializing Proxy Network...")
    proxy_net = ProxyNetwork(
        input_channels=args.feature_channels,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(proxy_net.parameters(), lr=args.lr)
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if args.amp else None
    
    # Load dataset
    print("Loading dataset...")
    if args.use_mot_dataset:
        # Use MOT16 dataset format
        dataset = MOTImageSequenceDataset(
            dataset_path=args.dataset_path,
            time_steps=args.time_steps,
            split=args.split,
            frame_stride=args.frame_stride
        )
    else:
        # Use video files
        dataset = VideoDataset(
            dataset_path=args.dataset_path,
            time_steps=args.time_steps,
            transform=None,
            max_videos=args.max_videos,
            frame_stride=args.frame_stride
        )
    
    print(f"Dataset loaded with {len(dataset)} sequences")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0.0
        total_rate = 0.0
        total_distortion = 0.0
        
        # Training phase
        proxy_net.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = batch.to(device)  # (B, T, C, H, W)
            batch = batch.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # Extract features using ST-NPP
            with torch.no_grad():
                preprocessed_features = stnpp(batch)
            
            # Get ground truth by running HEVC codec on a sample from batch
            if batch_idx % args.codec_interval == 0:
                with torch.no_grad():
                    # Use only the first item in batch to reduce processing time
                    sample_frames = batch[0].permute(1, 0, 2, 3)  # (T, C, H, W)
                    decoded_frames = hevc_encode_decode(
                        sample_frames, args.qp, temp_dir
                    ).to(device)
                    
                    # Extract features from decoded frames
                    decoded_frames = decoded_frames.unsqueeze(0)  # (1, T, C, H, W)
                    decoded_frames = decoded_frames.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
                    hevc_features = stnpp(decoded_frames)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass through proxy network with mixed precision
            if scaler is not None:
                with autocast():
                    reconstructed, latent = proxy_net(preprocessed_features)
                    rd_loss, rate, distortion = proxy_net.calculate_rd_loss(
                        preprocessed_features, reconstructed, latent,
                        lambda_value=args.lambda_value,
                        use_ssim=args.use_ssim
                    )
                
                # Backward pass and optimizer step with gradient scaling
                scaler.scale(rd_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstructed, latent = proxy_net(preprocessed_features)
                rd_loss, rate, distortion = proxy_net.calculate_rd_loss(
                    preprocessed_features, reconstructed, latent,
                    lambda_value=args.lambda_value,
                    use_ssim=args.use_ssim
                )
                
                # Backward pass and optimizer step
                rd_loss.backward()
                optimizer.step()
            
            # Update statistics
            total_loss += rd_loss.item()
            total_rate += rate.item()
            total_distortion += distortion.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{rd_loss.item():.4f}",
                'rate': f"{rate.item():.4f}",
                'distortion': f"{distortion.item():.4f}"
            })
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        avg_rate = total_rate / len(dataloader)
        avg_distortion = total_distortion / len(dataloader)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {avg_loss:.4f}, Rate: {avg_rate:.4f}, "
              f"Distortion: {avg_distortion:.4f}, "
              f"Time: {elapsed_time:.2f}s")
        
        # Save checkpoint if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.output_dir, "proxy_network_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': proxy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'rate': avg_rate,
                'distortion': avg_distortion
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"proxy_network_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': proxy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'rate': avg_rate,
                'distortion': avg_distortion
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "proxy_network_final.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': proxy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'rate': avg_rate,
        'distortion': avg_distortion
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    # Clean up
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    
    print("Training completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Proxy Network")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the video dataset or MOT16 dataset directory")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames in each sequence")
    parser.add_argument("--frame_stride", type=int, default=4,
                        help="Stride for frame sampling")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum number of videos to load (for debugging)")
    parser.add_argument("--use_mot_dataset", action="store_true",
                        help="Use MOT16 dataset format instead of video files")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use when using MOT16 (train or test)")
    
    # Model parameters
    parser.add_argument("--lambda_value", type=float, default=0.1,
                        help="Weight for the distortion term in the proxy loss")
    parser.add_argument("--use_ssim", action="store_true",
                        help="Use SSIM instead of MSE for distortion measurement")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate for optimizer")
    parser.add_argument("--codec_interval", type=int, default=10,
                        help="Interval for running the actual codec (slower)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="trained_models",
                        help="Directory to save the trained model")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Epoch interval for saving checkpoints")
    
    # HEVC parameters
    parser.add_argument("--qp", type=int, default=27,
                        help="Quantization Parameter for HEVC codec")
    
    # Additional model parameters
    parser.add_argument("--backbone", type=str, default="resnet34",
                        help="Backbone network for ST-NPP (resnet50, resnet34, efficientnet_b4)")
    parser.add_argument("--temporal_model", type=str, default="3dcnn",
                        help="Temporal model for ST-NPP (3dcnn, convlstm)")
    parser.add_argument("--fusion_type", type=str, default="concatenation",
                        help="Fusion type for ST-NPP (concatenation, attention)")
    parser.add_argument("--feature_channels", type=int, default=128,
                        help="Number of feature channels for ST-NPP and proxy network")
    parser.add_argument("--base_channels", type=int, default=64,
                        help="Base channels for the proxy network")
    parser.add_argument("--latent_channels", type=int, default=32,
                        help="Latent channels for the proxy network")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision training")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args) 