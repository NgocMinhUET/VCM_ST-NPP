#!/usr/bin/env python
"""
Training script for the Spatio-Temporal Neural Preprocessing (ST-NPP) module
with Quantization Adaptation Layer (QAL).

This script trains the ST-NPP module to reduce spatial-temporal redundancy
in videos before feeding them into a video codec.
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

from models.stnpp import STNPP
from models.qal import QAL
from models.proxy_network import ProxyNetwork


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
        
        # Limit the number of videos if specified
        if max_videos is not None:
            self.video_files = self.video_files[:max_videos]
        
        # Extract frames from videos and create sequences
        self.sequences = []
        for video_file in tqdm(self.video_files, desc="Loading videos"):
            self._extract_sequences(video_file)
    
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


def train(args):
    """Main training function."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize ST-NPP model
    print("Initializing ST-NPP model...")
    stnpp = STNPP(
        input_channels=3,
        output_channels=128,
        spatial_backbone=args.spatial_backbone,
        temporal_model=args.temporal_model,
        fusion_type=args.fusion_type,
        pretrained=True
    ).to(device)
    
    # Initialize QAL model
    print("Initializing QAL model...")
    qal = QAL(feature_channels=128, hidden_dim=64).to(device)
    
    # Load pre-trained Proxy Network
    print("Loading pre-trained Proxy Network...")
    proxy_net = ProxyNetwork(
        input_channels=128,
        base_channels=64,
        latent_channels=32
    ).to(device)
    
    # Load the trained proxy network
    if args.proxy_model_path:
        print(f"Loading proxy network from {args.proxy_model_path}")
        checkpoint = torch.load(args.proxy_model_path, map_location=device)
        proxy_net.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: No pre-trained proxy network provided")
    
    # Set proxy network to evaluation mode (frozen)
    proxy_net.eval()
    for param in proxy_net.parameters():
        param.requires_grad = False
    
    # Define optimizer for ST-NPP and QAL
    parameters = list(stnpp.parameters()) + list(qal.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    
    # Use mixed precision training if available
    scaler = GradScaler() if device.type == "cuda" else None
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = VideoDataset(
        dataset_path=args.dataset_path,
        time_steps=args.time_steps,
        frame_stride=args.frame_stride,
        max_videos=args.max_videos
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"Dataset loaded with {len(dataset)} sequences")
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Define loss function for reconstruction
    reconstruction_loss_fn = nn.MSELoss()
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    for epoch in range(args.epochs):
        start_time = time.time()
        total_loss = 0.0
        total_rd_loss = 0.0
        total_recon_loss = 0.0
        
        # Training phase
        stnpp.train()
        qal.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = batch.to(device)  # (B, T, C, H, W)
            
            # Create random QPs for this batch
            qp_values = torch.randint(
                low=args.min_qp, 
                high=args.max_qp + 1, 
                size=(batch.size(0),),
                device=device
            ).float()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    # Process through ST-NPP
                    batch_permuted = batch.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
                    stnpp_features = stnpp(batch_permuted)
                    
                    # Apply QAL
                    qal_features = qal(qp_values, stnpp_features)
                    
                    # Process through proxy network
                    reconstructed, latent = proxy_net(qal_features)
                    
                    # Calculate rate-distortion loss
                    rd_loss, rate, distortion = proxy_net.calculate_rd_loss(
                        qal_features, reconstructed, latent,
                        lambda_value=args.lambda_value,
                        use_ssim=args.use_ssim
                    )
                    
                    # Calculate reconstruction loss
                    recon_loss = reconstruction_loss_fn(reconstructed, qal_features)
                    
                    # Combined loss
                    loss = rd_loss + args.recon_weight * recon_loss
                
                # Backward pass and optimizer step with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Process through ST-NPP
                batch_permuted = batch.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
                stnpp_features = stnpp(batch_permuted)
                
                # Apply QAL
                qal_features = qal(qp_values, stnpp_features)
                
                # Process through proxy network
                reconstructed, latent = proxy_net(qal_features)
                
                # Calculate rate-distortion loss
                rd_loss, rate, distortion = proxy_net.calculate_rd_loss(
                    qal_features, reconstructed, latent,
                    lambda_value=args.lambda_value,
                    use_ssim=args.use_ssim
                )
                
                # Calculate reconstruction loss
                recon_loss = reconstruction_loss_fn(reconstructed, qal_features)
                
                # Combined loss
                loss = rd_loss + args.recon_weight * recon_loss
                
                # Backward pass and optimizer step
                loss.backward()
                optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_rd_loss += rd_loss.item()
            total_recon_loss += recon_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rd_loss': f"{rd_loss.item():.4f}",
                'recon_loss': f"{recon_loss.item():.4f}"
            })
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        avg_rd_loss = total_rd_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {avg_loss:.4f}, RD Loss: {avg_rd_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, "
              f"Time: {elapsed_time:.2f}s")
        
        # Save checkpoint if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Save ST-NPP model
            stnpp_path = os.path.join(args.output_dir, "stnpp_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': stnpp.state_dict(),
                'loss': best_loss
            }, stnpp_path)
            
            # Save QAL model
            qal_path = os.path.join(args.output_dir, "qal_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': qal.state_dict(),
                'loss': best_loss
            }, qal_path)
            
            print(f"Saved best model checkpoints to {stnpp_path} and {qal_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            # Save ST-NPP model
            stnpp_path = os.path.join(args.output_dir, f"stnpp_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': stnpp.state_dict(),
                'loss': avg_loss
            }, stnpp_path)
            
            # Save QAL model
            qal_path = os.path.join(args.output_dir, f"qal_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': qal.state_dict(),
                'loss': avg_loss
            }, qal_path)
            
            print(f"Saved checkpoints to {stnpp_path} and {qal_path}")
    
    # Save final models
    stnpp_final_path = os.path.join(args.output_dir, "stnpp_final.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': stnpp.state_dict(),
        'loss': avg_loss
    }, stnpp_final_path)
    
    qal_final_path = os.path.join(args.output_dir, "qal_final.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': qal.state_dict(),
        'loss': avg_loss
    }, qal_final_path)
    
    # Save combined model
    combined_path = os.path.join(args.output_dir, "stnpp_qal_model.pt")
    torch.save({
        'epoch': args.epochs,
        'stnpp_state_dict': stnpp.state_dict(),
        'qal_state_dict': qal.state_dict(),
        'loss': avg_loss
    }, combined_path)
    
    print(f"Saved final models to {stnpp_final_path}, {qal_final_path}, and {combined_path}")
    print("Training completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-NPP module with QAL")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the video dataset")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames in each sequence")
    parser.add_argument("--frame_stride", type=int, default=4,
                        help="Stride for frame sampling")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum number of videos to load (for debugging)")
    
    # Model parameters
    parser.add_argument("--spatial_backbone", type=str, default="resnet50",
                        choices=["resnet34", "resnet50", "efficientnet_b4"],
                        help="Backbone for the spatial branch")
    parser.add_argument("--temporal_model", type=str, default="3dcnn",
                        choices=["3dcnn", "convlstm"],
                        help="Model for the temporal branch")
    parser.add_argument("--fusion_type", type=str, default="concatenation",
                        choices=["concatenation", "attention"],
                        help="Type of fusion for spatial and temporal features")
    parser.add_argument("--proxy_model_path", type=str, default=None,
                        help="Path to the pre-trained proxy network model")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.00005,
                        help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Loss parameters
    parser.add_argument("--lambda_value", type=float, default=0.1,
                        help="Weight for the distortion term in the proxy loss")
    parser.add_argument("--recon_weight", type=float, default=0.5,
                        help="Weight for the reconstruction loss")
    parser.add_argument("--use_ssim", action="store_true",
                        help="Use SSIM instead of MSE for distortion measurement")
    parser.add_argument("--min_qp", type=int, default=22,
                        help="Minimum QP value for training")
    parser.add_argument("--max_qp", type=int, default=37,
                        help="Maximum QP value for training")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="trained_models",
                        help="Directory to save the trained models")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Epoch interval for saving checkpoints")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args) 