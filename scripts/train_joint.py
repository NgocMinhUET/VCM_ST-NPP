#!/usr/bin/env python
"""
Joint fine-tuning script for ST-NPP and QAL models.

This script implements joint training of previously trained ST-NPP and QAL models
for end-to-end optimization. It sets up TensorBoard logging and implements
a model versioning system.
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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stnpp import STNPP
from models.qal import QAL
from models.proxy_network import ProxyNetwork
from utils.model_utils import save_model_with_version, load_model_with_version
from utils.codec_utils import HevcCodec
from scripts.train_stnpp import VideoDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Joint fine-tuning of ST-NPP and QAL models')
    
    # Model paths
    parser.add_argument('--stnpp_model', type=str, required=True,
                        help='Path to pretrained ST-NPP model')
    parser.add_argument('--qal_model', type=str, required=True,
                        help='Path to pretrained QAL model')
    parser.add_argument('--proxy_model', type=str, required=True,
                        help='Path to trained Proxy Network model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to video dataset')
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='Path to validation dataset (if None, uses a portion of training data)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--qp_values', type=str, default='22,27,32,37',
                        help='Comma-separated list of QP values to train with')
    parser.add_argument('--lambda_distortion', type=float, default=1.0,
                        help='Weight for distortion loss component')
    parser.add_argument('--lambda_rate', type=float, default=0.1,
                        help='Weight for rate loss component')
    parser.add_argument('--lambda_perception', type=float, default=0.01,
                        help='Weight for perceptual loss component')
    parser.add_argument('--use_real_codec', action='store_true',
                        help='Use real HEVC codec instead of proxy network')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./trained_models/joint',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs/joint',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save model every N epochs')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda/cpu)')
    
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


class JointRDLoss(nn.Module):
    """
    Joint Rate-Distortion-Perception Loss for training ST-NPP + QAL.
    
    This loss combines:
    1. Distortion: Reconstruction quality (MSE or PSNR)
    2. Rate: Bitrate estimation from proxy network or real codec
    3. Perception: Feature preservation for downstream tasks
    """
    def __init__(self, lambda_distortion=1.0, lambda_rate=0.1, lambda_perception=0.01):
        super(JointRDLoss, self).__init__()
        self.lambda_distortion = lambda_distortion
        self.lambda_rate = lambda_rate
        self.lambda_perception = lambda_perception
        self.mse_loss = nn.MSELoss()
        
    def forward(self, original, preprocessed, estimated_rate, perceptual_loss=None):
        # Distortion loss (MSE between original and preprocessed)
        distortion_loss = self.mse_loss(original, preprocessed)
        
        # Rate loss (directly from rate estimator)
        # Handle both tensor and scalar rate values
        if isinstance(estimated_rate, torch.Tensor):
            rate_loss = estimated_rate.mean()
        else:
            rate_loss = estimated_rate
        
        # Total loss
        total_loss = (self.lambda_distortion * distortion_loss + 
                      self.lambda_rate * rate_loss)
        
        # Add perceptual loss if provided
        if perceptual_loss is not None:
            total_loss += self.lambda_perception * perceptual_loss
            
        return total_loss, {
            'distortion_loss': distortion_loss.item(),
            'rate_loss': rate_loss.item() if isinstance(rate_loss, torch.Tensor) else rate_loss,
            'perceptual_loss': perceptual_loss.item() if perceptual_loss is not None else 0,
            'total_loss': total_loss.item()
        }


def train_joint(args):
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'joint_training_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load ST-NPP model
    print(f"Loading ST-NPP model from {args.stnpp_model}")
    stnpp_model = STNPP(
        input_channels=3,
        output_channels=128,  # Default value, will be overridden by loaded model
        spatial_backbone='resnet50',  # Default value, will be overridden by loaded model
        temporal_model='3dcnn',  # Default value, will be overridden by loaded model
        fusion_type='concatenation',  # Default value, will be overridden by loaded model
        pretrained=False  # We're loading weights, so no need for pretrained
    )
    stnpp_model, _ = load_model_with_version(stnpp_model, args.stnpp_model, device)
    
    # Load QAL model
    print(f"Loading QAL model from {args.qal_model}")
    qal_model = QAL(
        feature_channels=128,  # Default value, will be overridden by loaded model
        hidden_dim=64  # Default value, will be overridden by loaded model
    )
    qal_model, _ = load_model_with_version(qal_model, args.qal_model, device)
    
    # Load Proxy Network or initialize codec
    if args.use_real_codec:
        print("Using real HEVC codec")
        codec = HevcCodec()
    else:
        print(f"Loading Proxy Network model from {args.proxy_model}")
        try:
            # Initialize with minimal params and let the model loading handle the rest
            proxy_model = ProxyNetwork(
                input_channels=3  # Only provide the required parameters
            )
            proxy_model, _ = load_model_with_version(proxy_model, args.proxy_model, device)
            proxy_model.eval()  # Set to evaluation mode since we don't train the proxy
            
            # Check if proxy model has QP conditioning
            has_qp_condition = hasattr(proxy_model, 'use_qp_condition') and proxy_model.use_qp_condition
            print(f"Proxy Network QP conditioning: {'Enabled' if has_qp_condition else 'Disabled'}")
            
        except Exception as e:
            print(f"Error loading Proxy Network model: {e}")
            raise
    
    # Set up datasets and dataloaders
    train_dataset = VideoDataset(args.dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    if args.val_dataset:
        val_dataset = VideoDataset(args.val_dataset)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        # Use a portion of training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Parse QP values
    qp_values = [int(qp) for qp in args.qp_values.split(',')]
    
    # Set up optimizer
    # We're training both ST-NPP and QAL jointly
    joint_params = list(stnpp_model.parameters()) + list(qal_model.parameters())
    optimizer = optim.Adam(joint_params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Set up loss function
    criterion = JointRDLoss(
        lambda_distortion=args.lambda_distortion,
        lambda_rate=args.lambda_rate,
        lambda_perception=args.lambda_perception
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        stnpp_model.train()
        qal_model.train()
        train_losses = []
        
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)
            batch_size = frames.size(0)
            
            # Randomly select QP for this batch
            qp = random.choice(qp_values)
            
            # Forward pass through ST-NPP
            preprocessed = stnpp_model(frames)
            
            # Convert QP to tensor with proper shape for QAL model
            qp_tensor = torch.full((frames.size(0),), qp, dtype=torch.float32, device=device)
            
            # Forward pass through QAL
            qal_output = qal_model(preprocessed, qp_tensor)
            
            # Estimate rate (using proxy network or real codec)
            if args.use_real_codec:
                # This would be a placeholder for using a real codec
                # In practice, you would need to implement a differentiable
                # approximation or use straight-through estimator
                raise NotImplementedError("Real codec training not implemented yet")
            else:
                # Use proxy network for rate estimation
                # Check if ProxyNetwork expects QP parameter
                if hasattr(proxy_model, 'use_qp_condition') and proxy_model.use_qp_condition:
                    estimated_rate = proxy_model(qal_output, qp_tensor)
                else:
                    estimated_rate = proxy_model(qal_output)
            
            # Calculate loss
            loss, loss_components = criterion(frames, qal_output, estimated_rate)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track training statistics
            train_losses.append(loss.item())
            
            # Log to TensorBoard every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
                global_step = epoch * len(train_loader) + batch_idx
                
                # Log loss components
                for name, value in loss_components.items():
                    writer.add_scalar(f'train/{name}', value, global_step)
                
                # Log learning rate
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                # Log sample images (original vs preprocessed)
                if batch_idx % 50 == 0:
                    # Only log the first image of the batch
                    writer.add_image('train/original', frames[0].cpu(), global_step)
                    writer.add_image('train/preprocessed', preprocessed[0].cpu(), global_step)
                    writer.add_image('train/qal_output', qal_output[0].cpu(), global_step)
        
        train_time = time.time() - start_time
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Training Loss: {avg_train_loss:.6f}, Time: {train_time:.2f}s")
        
        # Validation phase
        stnpp_model.eval()
        qal_model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                frames = batch['frames'].to(device)
                
                # We'll validate across all QP values
                qp_results = []
                for qp in qp_values:
                    # Forward pass through ST-NPP
                    preprocessed = stnpp_model(frames)
                    
                    # Convert QP to tensor with proper shape for QAL model
                    qp_tensor = torch.full((frames.size(0),), qp, dtype=torch.float32, device=device)
                    
                    # Forward pass through QAL
                    qal_output = qal_model(preprocessed, qp_tensor)
                    
                    # Estimate rate
                    if args.use_real_codec:
                        raise NotImplementedError("Real codec validation not implemented yet")
                    else:
                        # Check if ProxyNetwork expects QP parameter
                        if hasattr(proxy_model, 'use_qp_condition') and proxy_model.use_qp_condition:
                            estimated_rate = proxy_model(qal_output, qp_tensor)
                        else:
                            estimated_rate = proxy_model(qal_output)
                    
                    # Calculate loss
                    loss, _ = criterion(frames, qal_output, estimated_rate)
                    qp_results.append(loss.item())
                
                # Average loss across QP values
                val_losses.append(sum(qp_results) / len(qp_results))
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # Log validation loss to TensorBoard
        writer.add_scalar('validation/avg_loss', avg_val_loss, epoch)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save model periodically
        if (epoch + 1) % args.save_interval == 0:
            # Save ST-NPP model
            stnpp_path = save_model_with_version(
                stnpp_model,
                args.output_dir,
                "stnpp_joint",
                optimizer=None,  # We don't save optimizer state for intermediate saves
                epoch=epoch + 1,
                metrics={"val_loss": avg_val_loss},
                version=f"{timestamp}_e{epoch+1}"
            )
            
            # Save QAL model
            qal_path = save_model_with_version(
                qal_model,
                args.output_dir,
                "qal_joint",
                optimizer=None,
                epoch=epoch + 1,
                metrics={"val_loss": avg_val_loss},
                version=f"{timestamp}_e{epoch+1}"
            )
            
            print(f"Saved models to {stnpp_path} and {qal_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save best ST-NPP model
            best_stnpp_path = save_model_with_version(
                stnpp_model,
                args.output_dir,
                "stnpp_joint_best",
                optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": best_val_loss},
                version=timestamp
            )
            
            # Save best QAL model
            best_qal_path = save_model_with_version(
                qal_model,
                args.output_dir,
                "qal_joint_best",
                optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": best_val_loss},
                version=timestamp
            )
            
            print(f"New best model saved to {best_stnpp_path} and {best_qal_path}")
    
    # Final save at the end of training
    final_stnpp_path = save_model_with_version(
        stnpp_model,
        args.output_dir,
        "stnpp_joint_final",
        optimizer,
        epoch=args.epochs,
        metrics={"val_loss": avg_val_loss},
        version=timestamp
    )
    
    final_qal_path = save_model_with_version(
        qal_model,
        args.output_dir,
        "qal_joint_final",
        optimizer,
        epoch=args.epochs,
        metrics={"val_loss": avg_val_loss},
        version=timestamp
    )
    
    print(f"Training completed. Final models saved to {final_stnpp_path} and {final_qal_path}")
    writer.close()
    
    return {
        "best_stnpp_model": best_stnpp_path if 'best_stnpp_path' in locals() else None,
        "best_qal_model": best_qal_path if 'best_qal_path' in locals() else None,
        "final_stnpp_model": final_stnpp_path,
        "final_qal_model": final_qal_path,
        "best_val_loss": best_val_loss,
        "final_val_loss": avg_val_loss,
        "log_dir": log_dir
    }


if __name__ == "__main__":
    args = parse_args()
    train_joint(args) 