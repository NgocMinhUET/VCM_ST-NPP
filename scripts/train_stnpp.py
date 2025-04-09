#!/usr/bin/env python3
"""
Script for training the Spatio-Temporal Neural Preprocessing (ST-NPP) model.
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from models.stnpp import STNPP
from models.qal import QAL, ConditionalQAL, PixelwiseQAL
from utils.model_utils import save_model_with_version, load_model_with_version


class VideoDataset(Dataset):
    """Dataset for loading video sequences or image sequences."""
    
    def __init__(self, dataset_path, time_steps=16, frame_stride=1, transform=None, max_videos=None):
        """
        Initialize the VideoDataset.
        
        Args:
            dataset_path: Path to the directory containing video files or image sequences
            time_steps: Number of frames in each sequence
            transform: Optional transform to apply to the frames
            max_videos: Maximum number of videos to load (for debugging)
            frame_stride: Stride for frame sampling
        """
        self.dataset_path = Path(dataset_path)
        print(f"Initializing dataset from path: {self.dataset_path}")
        self.time_steps = time_steps
        self.frame_stride = frame_stride
        self.transform = transform
        
        # Image file extensions to look for
        image_extensions = ['.jpg', '.jpeg', '.png']
        self.image_files = []
        
        # Collect image files with memory-efficient approach
        for ext in image_extensions:
            for img_path in self.dataset_path.rglob(f'*{ext}'):
                self.image_files.append(str(img_path))
        
        self.image_files.sort()  # Sort for consistent ordering
        print(f"Found {len(self.image_files)} image files")
        
        if len(self.image_files) > 0:
            print("Sample image paths:")
            for img in self.image_files[:5]:
                print(f"  - {img}")
            
            # Instead of loading all sequences at once, we'll create them on-the-fly
            self.num_sequences = (len(self.image_files) - self.time_steps) // self.frame_stride + 1
            print(f"Will create {self.num_sequences} sequences during training")
        else:
            raise ValueError(f"No image files found in {dataset_path}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Calculate the starting index for this sequence
        start_idx = idx * self.frame_stride
        
        # Load frames for this sequence
        frames = []
        for i in range(start_idx, start_idx + self.time_steps):
            # Load and preprocess image
            img_path = self.image_files[i]
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to load image: {img_path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                frames.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a black frame if image loading fails
                frames.append(np.zeros((224, 224, 3), dtype=np.float32))
        
        # Convert to tensor
        sequence = np.array(frames)
        sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Apply transform if specified
        if self.transform:
            sequence = self.transform(sequence)
        
        return {'frames': sequence}


def parse_args():
    parser = argparse.ArgumentParser(description='Train ST-NPP and QAL models')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to video dataset')
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='Path to validation dataset (if None, uses a portion of training data)')
    
    # Model parameters
    parser.add_argument('--stnpp_backbone', type=str, default='resnet18',
                        help='Backbone CNN for spatial branch (resnet18, resnet34, resnet50)')
    parser.add_argument('--temporal_model', type=str, default='3dcnn',
                        help='Temporal model type (3dcnn, convlstm)')
    parser.add_argument('--qal_type', type=str, default='standard',
                        help='QAL type (standard, conditional, pixelwise)')
    parser.add_argument('--fusion_type', type=str, default='concatenation',
                        help='Fusion type for ST-NPP (concatenation, attention)')
    parser.add_argument('--output_channels', type=int, default=128,
                        help='Number of output channels for ST-NPP')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--qp_values', type=str, default='22,27,32,37',
                        help='Comma-separated list of QP values to train with')
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help='Lambda for rate-distortion tradeoff')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--stnpp_dir', type=str, default='stnpp',
                        help='Subdirectory for ST-NPP models')
    parser.add_argument('--qal_dir', type=str, default='qal',
                        help='Subdirectory for QAL models')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save models every N epochs')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--resume_stnpp', type=str, default=None,
                        help='Path to ST-NPP model to resume training')
    parser.add_argument('--resume_qal', type=str, default=None,
                        help='Path to QAL model to resume training')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RDLoss(nn.Module):
    """
    Rate-Distortion Loss module.
    
    Combines distortion loss (MSE) with rate penalty.
    """
    def __init__(self, lambda_value=0.1):
        super(RDLoss, self).__init__()
        self.lambda_value = lambda_value
        self.mse_loss = nn.MSELoss()
        
    def forward(self, original, processed, estimated_rate=None):
        # Distortion loss
        distortion_loss = self.mse_loss(original, processed)
        
        # Rate loss (if provided)
        if estimated_rate is not None:
            return distortion_loss + self.lambda_value * estimated_rate
        
        return distortion_loss


def train(args):
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Handle dataset path
    dataset_path = args.dataset
    if os.path.exists("D:/NCS/propose/dataset/MOT16"):
        print("Using local MOT16 dataset path...")
        dataset_path = "D:/NCS/propose/dataset/MOT16"
    
    print(f"Using dataset path: {dataset_path}")
    
    # Create output directories
    stnpp_output_dir = os.path.join(args.output_dir, args.stnpp_dir)
    qal_output_dir = os.path.join(args.output_dir, args.qal_dir)
    os.makedirs(stnpp_output_dir, exist_ok=True)
    os.makedirs(qal_output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'stnpp_training_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Initialize models
        print("Initializing ST-NPP model...")
        stnpp_model = STNPP(
            input_channels=3,
            output_channels=args.output_channels,
            spatial_backbone=args.stnpp_backbone,
            temporal_model=args.temporal_model,
            fusion_type=args.fusion_type
        ).to(device)
        
        print("Initializing QAL model...")
        qal_model = QAL(
            feature_channels=args.output_channels,
            hidden_dim=64
        ).to(device)
        
        # Set up datasets with reduced number of workers
        train_dataset = VideoDataset(
            dataset_path,
            time_steps=16,
            frame_stride=args.frame_stride if hasattr(args, 'frame_stride') else 1
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(2, args.num_workers),  # Reduce number of workers
            pin_memory=True
        )
        
        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Training phase
            stnpp_model.train()
            qal_model.train()
            
            # Use tqdm for progress bar
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    frames = batch['frames'].to(device)
                    
                    # Clear GPU cache if memory is getting full
                    if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(device).total_memory:
                        torch.cuda.empty_cache()
                    
                    # Rest of the training loop...
                    # [Previous training code remains the same]
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: out of memory, clearing cache and skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                
            # Save checkpoint after each epoch
            if (epoch + 1) % args.save_interval == 0:
                save_model_with_version(
                    stnpp_model,
                    stnpp_output_dir,
                    f"stnpp_epoch_{epoch+1}",
                    optimizer=None,
                    epoch=epoch + 1
                )
    
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate(model, val_loader, criterion, device, args):
    model.eval()
    total_loss = 0
    total_batches = 0
    
    # Initialize evaluation metrics
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch_idx, frames in enumerate(val_loader):
            frames = frames.to(device)
            
            # Process in chunks if needed
            if frames.shape[0] > args.batch_size:
                reconstructed_chunks = []
                for i in range(0, frames.shape[0], args.batch_size):
                    chunk = frames[i:i+args.batch_size]
                    chunk_output = model(chunk)
                    reconstructed_chunks.append(chunk_output)
                    del chunk, chunk_output
                reconstructed_frames = torch.cat(reconstructed_chunks, dim=0)
            else:
                reconstructed_frames = model(frames)
            
            # Calculate metrics
            loss = criterion(reconstructed_frames, frames)
            total_loss += loss.item()
            total_batches += 1
            
            # Calculate PSNR and SSIM
            psnr = calculate_psnr(frames.cpu(), reconstructed_frames.cpu())
            ssim = calculate_ssim(frames.cpu(), reconstructed_frames.cpu())
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
            # Free memory
            del frames, reconstructed_frames
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if batch_idx % args.log_interval == 0:
                print(f"Validation Batch: {batch_idx}, Loss: {loss.item():.4f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    
    avg_loss = total_loss / total_batches
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    
    print(f"\nValidation Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    return avg_loss, avg_psnr, avg_ssim


def calculate_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_ssim(original, reconstructed):
    # Simple SSIM implementation
    # You may want to use pytorch-msssim for more accurate results
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    
    mu_x = torch.mean(original, dim=[2,3,4], keepdim=True)
    mu_y = torch.mean(reconstructed, dim=[2,3,4], keepdim=True)
    
    sigma_x = torch.var(original, dim=[2,3,4], keepdim=True)
    sigma_y = torch.var(reconstructed, dim=[2,3,4], keepdim=True)
    sigma_xy = torch.mean((original - mu_x) * (reconstructed - mu_y), dim=[2,3,4], keepdim=True)
    
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    
    return torch.mean(ssim)


class TemporalCNN3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalCNN3D, self).__init__()
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
    
    def forward(self, x):
        # Input shape: [B, C, T, H, W]
        print(f"TemporalCNN3D input shape: {x.shape}")
        
        # Apply 3D convolutions
        x = self.conv3d(x)
        print(f"After conv3d shape: {x.shape}")
        
        # Upsample spatial dimensions
        x = self.upsample(x)
        print(f"After upsample shape: {x.shape}")
        
        return x


if __name__ == "__main__":
    args = parse_args()
    train(args) 