#!/usr/bin/env python
"""
Training script for the Differentiable Proxy Network using MOT16 dataset.
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
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import cv2
import subprocess
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom MOT dataset
from mot_dataset import MOTImageSequenceDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train Proxy Network on MOT16 Dataset")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="D:/NCS/propose/dataset/MOT16",
                        help="Path to the MOT16 dataset directory")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use (train or test)")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames in each sequence")
    parser.add_argument("--frame_stride", type=int, default=4,
                        help="Stride for frame sampling")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="trained_models/proxy_network",
                        help="Directory to save the trained model")
    
    return parser.parse_args()

class SimpleProxyNetwork(nn.Module):
    """A simplified proxy network for testing."""
    
    def __init__(self, input_channels=3, time_steps=16):
        super(SimpleProxyNetwork, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, input_channels, kernel_size=2, stride=2),
            nn.Sigmoid()  # Output in range [0, 1]
        )
    
    def forward(self, x):
        # x shape: (B, C, T, H, W)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def calculate_loss(self, x, reconstructed):
        """Calculate mean squared error loss."""
        mse_loss = nn.MSELoss()
        return mse_loss(reconstructed, x)

def train(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing Proxy Network...")
    model = SimpleProxyNetwork(
        input_channels=3,
        time_steps=args.time_steps
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Load dataset
    print("Loading MOT16 dataset...")
    dataset = MOTImageSequenceDataset(
        dataset_path=args.dataset_path,
        time_steps=args.time_steps,
        split=args.split,
        frame_stride=args.frame_stride
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    print(f"Starting training for {args.epochs} epochs...")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, frames in enumerate(progress_bar):
            # Move data to device
            frames = frames.to(device)  # (B, T, C, H, W)
            frames = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, _ = model(frames)
            
            # Calculate loss
            loss = model.calculate_loss(frames, reconstructed)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")
        
        # Save checkpoint if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.output_dir, "proxy_network_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "proxy_network_final.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    args = parse_args()
    train(args) 