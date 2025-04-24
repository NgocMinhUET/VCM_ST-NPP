#!/usr/bin/env python
"""
Training script for task-aware video compression model.

This script trains a combined model that optimizes for both
compression quality and downstream task performance.
"""

# Import warning suppression first (before any other imports)
try:
    import fix_tf_warnings
except ImportError:
    print("Warning: fix_tf_warnings.py not found. TensorFlow warnings will not be suppressed.")

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import random
from pathlib import Path

# Import project modules
try:
    from models.combined_model import CombinedModel
    from utils.data_utils import get_dataloader, get_transforms
    from utils.loss_utils import compute_total_loss
    from utils.model_utils import save_model, load_model
    from utils.metric_utils import (
        compute_psnr, compute_ssim, compute_bpp,
        evaluate_detection, evaluate_segmentation, evaluate_tracking
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please make sure all required packages are installed.")
    print("Run: pip install -r requirements.txt")
    import sys
    sys.exit(1)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Train task-aware video compression model")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="dummy", 
                        help="Dataset name or path")
    parser.add_argument("--task_type", type=str, default="detection",
                        choices=["detection", "segmentation", "tracking"],
                        help="Type of downstream task")
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Number of frames in each sequence")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--qp", type=int, default=30,
                        help="Quantization parameter (if not random)")
    parser.add_argument("--random_qp", action="store_true",
                        help="Use random QP values")
    parser.add_argument("--qp_range", type=int, nargs=2, default=[0, 50],
                        help="Range of QP values if random")
    
    # Model parameters
    parser.add_argument("--model_checkpoint", type=str, default=None,
                        help="Path to load model checkpoint")
    parser.add_argument("--hidden_channels", type=int, default=128,
                        help="Number of hidden channels in model")
    
    # Loss parameters
    parser.add_argument("--task_weight", type=float, default=1.0,
                        help="Weight for task loss (λ1)")
    parser.add_argument("--recon_weight", type=float, default=1.0,
                        help="Weight for reconstruction loss (λ2)")
    parser.add_argument("--bitrate_weight", type=float, default=0.1,
                        help="Weight for bitrate loss (λ3)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval in iterations")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="Validation interval in epochs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only for 1 batch and 1 epoch for testing")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(args):
    """Create and initialize the model"""
    model = CombinedModel(
        task_type=args.task_type,
        hidden_channels=args.hidden_channels,
        seq_length=args.seq_length
    )
    
    # Load checkpoint if provided
    if args.model_checkpoint and os.path.exists(args.model_checkpoint):
        print(f"Loading model checkpoint from {args.model_checkpoint}")
        load_model(model, args.model_checkpoint)
    
    return model.to(args.device)


def log_gpu_memory(writer, step, device=None):
    """Log GPU memory usage to TensorBoard"""
    if torch.cuda.is_available():
        if device is None:
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        else:
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        
        writer.add_scalar('Memory/Allocated (MB)', memory_allocated, step)
        writer.add_scalar('Memory/Reserved (MB)', memory_reserved, step)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader containing training data
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        device: Device to use for training
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter
        
    Returns:
        Average loss, PSNR, and SSIM for the epoch
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_bpp = 0.0
    processed_batches = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    # Track start time
    start_time = time.time()
    
    for i, batch in enumerate(pbar):
        try:
            # Get inputs and targets from batch
            if isinstance(batch, dict):
                frames = batch.get('frames', None)
                labels = batch.get('labels', None)
                qp = batch.get('qp', None)
            else:
                # Assuming batch is a list or tuple
                if len(batch) >= 3:
                    frames, labels, qp = batch[:3]
                else:
                    print(f"Warning: Batch format unexpected: {type(batch)}, {len(batch) if hasattr(batch, '__len__') else 'no length'}")
                    frames = batch[0] if len(batch) > 0 else None
                    labels = batch[1] if len(batch) > 1 else None
                    qp = None
            
            # Print debug info for the first batch
            if i == 0:
                print(f"Input frames shape: {frames.shape if frames is not None else 'None'}")
                print(f"Target keys: {labels.keys() if isinstance(labels, dict) else 'Not a dict'}")
                print(f"QP: {qp}")

            # Move data to device
            if frames is not None:
                frames = frames.to(device)
            if labels is not None and isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            elif labels is not None and isinstance(labels, dict):
                labels = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in labels.items()}
            if qp is not None and isinstance(qp, torch.Tensor):
                qp = qp.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                outputs = model(frames, qp)
                
                # Extract components from the outputs dictionary
                reconstructed = outputs['reconstructed']
                task_output = outputs['task_output']
                bitrate = outputs['bitrate']
                
                # Calculate loss
                loss = criterion(task_output, labels, reconstructed, frames, bitrate)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    try:
                        # Calculate PSNR and SSIM
                        psnr = compute_psnr(reconstructed, frames)
                        ssim = compute_ssim(reconstructed, frames)
                        bpp = bitrate.mean().item() if isinstance(bitrate, torch.Tensor) else bitrate
                        
                        # Update running totals
                        total_loss += loss.item()
                        total_psnr += psnr
                        total_ssim += ssim
                        total_bpp += bpp
                        processed_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'psnr': f"{psnr:.2f}",
                            'bpp': f"{bpp:.4f}"
                        })
                        
                        # Log to TensorBoard
                        if writer is not None and i % 10 == 0:  # Log every 10 batches
                            step = epoch * len(dataloader) + i
                            writer.add_scalar('train/loss', loss.item(), step)
                            writer.add_scalar('train/psnr', psnr, step)
                            writer.add_scalar('train/ssim', ssim, step)
                            writer.add_scalar('train/bpp', bpp, step)
                    
                    except Exception as e:
                        print(f"Error calculating metrics: {e}")
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"CUDA OOM in batch {i}. Skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Runtime error in forward pass: {e}")
                    continue
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue
    
    # Calculate averages
    if processed_batches > 0:
        avg_loss = total_loss / processed_batches
        avg_psnr = total_psnr / processed_batches
        avg_ssim = total_ssim / processed_batches
        avg_bpp = total_bpp / processed_batches
    else:
        avg_loss = float('inf')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_bpp = 0.0
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print epoch summary
    print(f"Epoch {epoch} [Train] - Avg Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, BPP: {avg_bpp:.4f}, Time: {elapsed_time:.2f}s")
    
    return avg_loss, avg_psnr, avg_ssim, avg_bpp


def validate(model, dataloader, criterion, device, epoch, writer=None):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader containing validation data
        criterion: Loss function
        device: Device to use for validation
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter
        
    Returns:
        Average loss, PSNR, and SSIM for the validation set
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_bpp = 0.0
    processed_batches = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            try:
                # Get inputs and targets from batch
                if isinstance(batch, dict):
                    frames = batch.get('frames', None)
                    labels = batch.get('labels', None)
                    qp = batch.get('qp', None)
                else:
                    # Assuming batch is a list or tuple
                    if len(batch) >= 3:
                        frames, labels, qp = batch[:3]
                    else:
                        print(f"Warning: Batch format unexpected: {type(batch)}, {len(batch) if hasattr(batch, '__len__') else 'no length'}")
                        frames = batch[0] if len(batch) > 0 else None
                        labels = batch[1] if len(batch) > 1 else None
                        qp = None

                # Move data to device
                if frames is not None:
                    frames = frames.to(device)
                if labels is not None and isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                elif labels is not None and isinstance(labels, dict):
                    labels = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in labels.items()}
                if qp is not None and isinstance(qp, torch.Tensor):
                    qp = qp.to(device)
                
                # Forward pass
                try:
                    outputs = model(frames, qp)
                    
                    # Extract components from the outputs dictionary
                    reconstructed = outputs['reconstructed']
                    task_output = outputs['task_output']
                    bitrate = outputs['bitrate']
                    
                    # Calculate loss
                    loss = criterion(task_output, labels, reconstructed, frames, bitrate)
                    
                    # Calculate metrics
                    try:
                        # Calculate PSNR and SSIM
                        psnr = compute_psnr(reconstructed, frames)
                        ssim = compute_ssim(reconstructed, frames)
                        bpp = bitrate.mean().item() if isinstance(bitrate, torch.Tensor) else bitrate
                        
                        # Update running totals
                        total_loss += loss.item()
                        total_psnr += psnr
                        total_ssim += ssim
                        total_bpp += bpp
                        processed_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'psnr': f"{psnr:.2f}",
                            'bpp': f"{bpp:.4f}"
                        })
                    
                    except Exception as e:
                        print(f"Error calculating metrics: {e}")
                        continue
                
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    continue
            
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
    
    # Calculate averages
    if processed_batches > 0:
        avg_loss = total_loss / processed_batches
        avg_psnr = total_psnr / processed_batches
        avg_ssim = total_ssim / processed_batches
        avg_bpp = total_bpp / processed_batches
    else:
        avg_loss = float('inf')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_bpp = 0.0
    
    # Print validation summary
    print(f"Epoch {epoch} [Val] - Avg Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, BPP: {avg_bpp:.4f}")
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/psnr', avg_psnr, epoch)
        writer.add_scalar('val/ssim', avg_ssim, epoch)
        writer.add_scalar('val/bpp', avg_bpp, epoch)
    
    return avg_loss, avg_psnr, avg_ssim, avg_bpp


def main(args):
    """Main training function"""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(args)
    print(f"Created {args.task_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create transforms
    train_transform, val_transform = get_transforms(
        task_type=args.task_type,
        resolution=(256, 256),  # Example resolution
        augment=True
    )
    
    # Create dataloaders
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        task_type=args.task_type,
        split='train',
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        random_qp=args.random_qp,
        qp_range=args.qp_range,
        shuffle=True,
        transform=train_transform
    )
    
    val_loader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        task_type=args.task_type,
        split='val',
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        random_qp=False,  # Use fixed QP for validation
        qp_range=(args.qp, args.qp),  # Use single QP value
        shuffle=False,
        transform=val_transform
    )
    
    print(f"Created dataloaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    # Create TensorBoard writer
    log_dir = Path(args.output_dir) / "logs" / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    
    # Apply dry-run modifications if specified
    if args.dry_run:
        print("Running in dry-run mode: 1 batch, 1 epoch")
        args.epochs = 1
        # Create subset of the dataloaders for dry run (just 1 batch)
        train_subset = torch.utils.data.Subset(train_loader.dataset, range(min(args.batch_size, len(train_loader.dataset))))
        val_subset = torch.utils.data.Subset(val_loader.dataset, range(min(args.batch_size, len(val_loader.dataset))))
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_loader.collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_loader.collate_fn,
            pin_memory=True
        )
        print(f"Reduced to {len(train_loader)} training batches and {len(val_loader)} validation batches for dry run")
    
    # Training loop
    best_val_loss = float('inf')
    best_val_psnr = 0.0
    
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        try:
            # Train for one epoch
            train_loss, train_psnr, train_ssim, train_bpp = train_epoch(model, train_loader, compute_total_loss, optimizer, device, epoch, writer)
            
            # Validate if it's time
            if (epoch + 1) % args.val_interval == 0:
                val_loss, val_psnr, val_ssim, val_bpp = validate(model, val_loader, compute_total_loss, device, epoch, writer)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = Path(args.output_dir) / f"best_model_loss.pth"
                    save_model(model, str(checkpoint_path), optimizer, epoch, {'loss': val_loss, 'psnr': val_psnr, 'ssim': val_ssim, 'bpp': val_bpp})
                    print(f"Saved best model (by loss) to {checkpoint_path}")
                
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    checkpoint_path = Path(args.output_dir) / f"best_model_psnr.pth"
                    save_model(model, str(checkpoint_path), optimizer, epoch, {'loss': val_loss, 'psnr': val_psnr, 'ssim': val_ssim, 'bpp': val_bpp})
                    print(f"Saved best model (by PSNR) to {checkpoint_path}")
            
            # Save last model at regular intervals
            if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                checkpoint_path = Path(args.output_dir) / f"model_epoch_{epoch+1}.pth"
                save_model(model, str(checkpoint_path), optimizer, epoch, {'loss': train_loss, 'psnr': train_psnr, 'ssim': train_ssim, 'bpp': train_bpp})
                print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
        
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {str(e)}")
            # Save model before potential crash
            checkpoint_path = Path(args.output_dir) / f"model_epoch_{epoch+1}_crash.pth"
            save_model(model, str(checkpoint_path), optimizer, epoch, {})
            print(f"Saved crash checkpoint at epoch {epoch+1} to {checkpoint_path}")
    
    # Save final model
    final_checkpoint_path = Path(args.output_dir) / "final_model.pth"
    save_model(model, str(final_checkpoint_path), optimizer, args.epochs - 1, {})
    print(f"Saved final model to {final_checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    print("Training completed")


if __name__ == "__main__":
    args = parse_args()
    main(args) 