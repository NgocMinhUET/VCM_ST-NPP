#!/usr/bin/env python
"""
Training script for the improved autoencoder with vector quantization for video compression.
This script trains the model on the MOT16 dataset with enhanced compression capabilities.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm

# Import our dataset and model
from mot_dataset import MOTImageSequenceDataset
from improved_autoencoder import ImprovedAutoencoder

def parse_args():
    parser = argparse.ArgumentParser(description="Train improved autoencoder with VQ on MOT16 dataset")
    
    # Dataset parameters
    parser.add_argument("--dataset_path", type=str, default="D:/NCS/propose/dataset/MOT16",
                        help="Path to the MOT16 dataset directory")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (train or test)")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames in each sequence")
    parser.add_argument("--frame_stride", type=int, default=4,
                        help="Stride for frame sampling")
    
    # Model parameters
    parser.add_argument("--latent_channels", type=int, default=8,
                        help="Number of channels in latent space")
    parser.add_argument("--time_reduction", type=int, default=2,
                        help="Temporal reduction factor")
    parser.add_argument("--num_embeddings", type=int, default=512,
                        help="Number of embeddings for vector quantization")
    parser.add_argument("--commitment_cost", type=float, default=0.25,
                        help="Commitment cost for VQ-VAE loss")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="trained_models/improved_autoencoder",
                        help="Directory to save the trained model")
    
    return parser.parse_args()

def train(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device and print detailed info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print CUDA info if available
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(torch.cuda.current_device())
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing improved autoencoder...")
    model = ImprovedAutoencoder(
        input_channels=3,
        latent_channels=args.latent_channels,
        time_reduction=args.time_reduction
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Initialize loss function for reconstruction
    mse_criterion = nn.MSELoss()
    
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
        running_recon_loss = 0.0
        running_vq_loss = 0.0
        start_time = time.time()
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, frames in enumerate(progress_bar):
            # Move data to device and prepare input
            frames = frames.to(device)  # (B, T, C, H, W)
            frames = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, vq_loss, latent = model(frames)
            
            # Calculate reconstruction loss
            recon_loss = mse_criterion(reconstructed, frames)
            
            # Total loss is reconstruction loss plus VQ loss
            loss = recon_loss + vq_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_vq_loss += vq_loss.item()
            
            # Calculate and display current BPP
            with torch.no_grad():
                bpp = model.calculate_bitrate(latent)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "vq": f"{vq_loss.item():.4f}",
                "bpp": f"{bpp:.4f}"
            })
        
        # Calculate average losses for the epoch
        avg_loss = running_loss / len(dataloader)
        avg_recon_loss = running_recon_loss / len(dataloader)
        avg_vq_loss = running_vq_loss / len(dataloader)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon_loss:.4f}, VQ: {avg_vq_loss:.4f}), "
              f"Time: {elapsed_time:.2f}s")
        
        # Save checkpoint if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.output_dir, "autoencoder_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save latest model
        checkpoint_path = os.path.join(args.output_dir, "autoencoder_latest.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)

if __name__ == "__main__":
    args = parse_args()
    train(args) 