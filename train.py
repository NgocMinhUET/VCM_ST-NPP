#!/usr/bin/env python
"""
Training script for task-aware video compression model.

This script trains a combined model that optimizes for both
compression quality and downstream task performance.
"""

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
from models.combined_model import CombinedModel
from utils.data_utils import get_dataloader, get_transforms
from utils.loss_utils import compute_total_loss
from utils.model_utils import save_model, load_model
from utils.metric_utils import (
    compute_psnr, compute_ssim, compute_bpp,
    evaluate_detection, evaluate_segmentation, evaluate_tracking
)


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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """
    Train the model for one epoch
    
    Args:
        model: The model to train
        dataloader: Data loader for training data
        criterion: The loss function (compute_total_loss)
        optimizer: The optimizer
        device: Device to use for tensors
        epoch: Current epoch number
        writer: TensorBoard writer
    
    Returns:
        Dict containing average losses and metrics
    """
    model.train()
    
    # Initialize metrics accumulators
    total_loss = 0.0
    total_task_loss = 0.0
    total_recon_loss = 0.0
    total_bitrate_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_bpp = 0.0
    
    # Counter for number of batches
    num_batches = 0
    
    # Create progress bar
    pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}")
    start_time = time.time()
    
    try:
        for batch_idx, batch in enumerate(dataloader):
            # Extract data from batch
            frames = batch['frames'].to(device)
            labels = batch['labels']
            qp = batch['qp'].to(device)
            
            # Print shape information in the first batch
            if batch_idx == 0:
                print(f"Input shape: {frames.shape}, Device: {frames.device}")
                print(f"QP shape: {qp.shape}, Device: {qp.device}")
                print(f"Labels type: {type(labels)}, Length: {len(labels)}")
            
            try:
                # Forward pass
                output = model(frames, qp)
                
                # Get output components
                reconstructed = output['reconstructed']
                task_output = output['task_output']
                bitrate = output['bitrate']
                
                # Convert labels to appropriate format based on task type
                if args.task_type == 'detection':
                    # Process detection labels (convert list of tensors to appropriate format)
                    task_labels = torch.zeros_like(task_output)
                    # This is simplified - in a real implementation, you would convert 
                    # bounding boxes to the format expected by your model
                
                elif args.task_type == 'segmentation':
                    # Process segmentation labels
                    task_labels = torch.zeros_like(task_output)
                    # This is simplified - in a real implementation, you would convert 
                    # segmentation masks to the format expected by your model
                
                elif args.task_type == 'tracking':
                    # Process tracking labels
                    task_labels = torch.zeros_like(task_output)
                    # This is simplified - in a real implementation, you would convert 
                    # tracking annotations to the format expected by your model
                
                # Compute loss
                loss = compute_total_loss(
                    task_out=task_output,
                    labels=task_labels,
                    recon=reconstructed,
                    raw=frames,
                    bitrate=bitrate,
                    task_weight=args.task_weight,
                    recon_weight=args.recon_weight,
                    bitrate_weight=args.bitrate_weight
                )
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    # Task-specific metrics would be calculated here
                    # For now, we'll just use basic metrics
                    psnr = compute_psnr(frames, reconstructed)
                    ssim = compute_ssim(frames, reconstructed)
                    bpp = compute_bpp(bitrate.mean().item(), frames.size(3), frames.size(4), frames.size(2))
                
                # Accumulate metrics
                total_loss += loss.item()
                total_psnr += psnr.item() if isinstance(psnr, torch.Tensor) else psnr
                total_ssim += ssim.item() if isinstance(ssim, torch.Tensor) else ssim
                total_bpp += bpp
                num_batches += 1
                
                # Log to TensorBoard
                global_step = epoch * len(dataloader) + batch_idx
                if batch_idx % args.log_interval == 0:
                    writer.add_scalar('Train/Loss', loss.item(), global_step)
                    writer.add_scalar('Train/PSNR', psnr.item() if isinstance(psnr, torch.Tensor) else psnr, global_step)
                    writer.add_scalar('Train/SSIM', ssim.item() if isinstance(ssim, torch.Tensor) else ssim, global_step)
                    writer.add_scalar('Train/BPP', bpp, global_step)
                    log_gpu_memory(writer, global_step, device)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'psnr': f"{psnr.item() if isinstance(psnr, torch.Tensor) else psnr:.2f}",
                    'bpp': f"{bpp:.4f}"
                })
            
            except RuntimeError as e:
                # Handle CUDA out-of-memory errors
                if "CUDA out of memory" in str(e):
                    print(f"CUDA OOM error. Batch size might be too large. Skipping batch {batch_idx}.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Runtime error in batch {batch_idx}: {str(e)}")
                    continue
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
    
    finally:
        pbar.close()
    
    # Calculate averages
    avg_loss = total_loss / max(num_batches, 1)
    avg_psnr = total_psnr / max(num_batches, 1)
    avg_ssim = total_ssim / max(num_batches, 1)
    avg_bpp = total_bpp / max(num_batches, 1)
    
    # Calculate elapsed time
    elapsed = time.time() - start_time
    
    # Log final metrics for the epoch
    writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Epoch_PSNR', avg_psnr, epoch)
    writer.add_scalar('Train/Epoch_SSIM', avg_ssim, epoch)
    writer.add_scalar('Train/Epoch_BPP', avg_bpp, epoch)
    
    print(f"Epoch {epoch+1} completed in {elapsed:.2f}s. Metrics: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}, BPP={avg_bpp:.4f}")
    
    return {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'bpp': avg_bpp
    }


def validate(model, dataloader, criterion, device, epoch, writer):
    """
    Validate the model on validation data
    
    Args:
        model: The model to validate
        dataloader: Data loader for validation data
        criterion: The loss function (compute_total_loss)
        device: Device to use for tensors
        epoch: Current epoch number
        writer: TensorBoard writer
    
    Returns:
        Dict containing average losses and metrics
    """
    model.eval()
    
    # Initialize metrics accumulators
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_bpp = 0.0
    
    # Task-specific metrics
    task_metrics = {}
    
    # Counter for number of batches
    num_batches = 0
    successful_batches = 0
    error_batches = 0
    
    # Create progress bar
    pbar = tqdm(total=len(dataloader), desc=f"Validation {epoch+1}")
    start_time = time.time()
    
    with torch.no_grad():
        try:
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Extract data from batch
                    frames = batch['frames'].to(device)
                    labels = batch['labels']
                    qp = batch['qp'].to(device)
                    
                    # Print shape information for the first batch
                    if batch_idx == 0:
                        print(f"Validation input shape: {frames.shape}, Device: {frames.device}")
                        print(f"Validation QP shape: {qp.shape}, Device: {qp.device}")
                        print(f"Validation labels type: {type(labels)}, Length: {len(labels)}")
                    
                    # Forward pass
                    try:
                        output = model(frames, qp)
                        
                        # Get output components
                        reconstructed = output['reconstructed']
                        task_output = output['task_output']
                        bitrate = output['bitrate']
                        
                        # Convert labels to appropriate format based on task type
                        if args.task_type == 'detection':
                            # Process detection labels
                            task_labels = torch.zeros_like(task_output)
                            # This is simplified
                        
                        elif args.task_type == 'segmentation':
                            # Process segmentation labels
                            task_labels = torch.zeros_like(task_output)
                            # This is simplified
                        
                        elif args.task_type == 'tracking':
                            # Process tracking labels
                            task_labels = torch.zeros_like(task_output)
                            # This is simplified
                        
                        # Compute loss
                        try:
                            loss = compute_total_loss(
                                task_out=task_output,
                                labels=task_labels,
                                recon=reconstructed,
                                raw=frames,
                                bitrate=bitrate,
                                task_weight=args.task_weight,
                                recon_weight=args.recon_weight,
                                bitrate_weight=args.bitrate_weight
                            )
                        except Exception as e:
                            print(f"Error computing validation loss for batch {batch_idx}: {str(e)}")
                            loss = torch.tensor(float('nan'), device=device)
                        
                        # Calculate metrics
                        try:
                            psnr = compute_psnr(frames, reconstructed)
                            ssim = compute_ssim(frames, reconstructed)
                            bpp = compute_bpp(bitrate.mean().item(), frames.size(3), frames.size(4), frames.size(2))
                            
                            # Ensure metrics are valid numbers
                            if torch.isnan(psnr).any() or torch.isinf(psnr).any():
                                print(f"Warning: Invalid PSNR values in batch {batch_idx}")
                                psnr = torch.tensor(0.0, device=device)
                            
                            if torch.isnan(ssim).any() or torch.isinf(ssim).any():
                                print(f"Warning: Invalid SSIM values in batch {batch_idx}")
                                ssim = torch.tensor(0.0, device=device)
                                
                            if np.isnan(bpp) or np.isinf(bpp):
                                print(f"Warning: Invalid BPP value in batch {batch_idx}")
                                bpp = 0.0
                        except Exception as e:
                            print(f"Error computing metrics for batch {batch_idx}: {str(e)}")
                            psnr = torch.tensor(0.0, device=device)
                            ssim = torch.tensor(0.0, device=device)
                            bpp = 0.0
                        
                        # Calculate task-specific metrics
                        try:
                            if args.task_type == 'detection':
                                batch_task_metrics = evaluate_detection(task_output, task_labels)
                            elif args.task_type == 'segmentation':
                                batch_task_metrics = evaluate_segmentation(task_output, task_labels)
                            elif args.task_type == 'tracking':
                                batch_task_metrics = evaluate_tracking(task_output, task_labels)
                            else:
                                batch_task_metrics = {}
                                
                            # Update task metrics
                            for k, v in batch_task_metrics.items():
                                if k not in task_metrics:
                                    task_metrics[k] = 0.0
                                # Ensure metric value is valid
                                if not (np.isnan(v) or np.isinf(v)):
                                    task_metrics[k] += v
                                else:
                                    print(f"Warning: Invalid {k} value in batch {batch_idx}")
                        except Exception as e:
                            print(f"Error computing task metrics for batch {batch_idx}: {str(e)}")
                            batch_task_metrics = {}
                        
                        # Accumulate metrics (only if loss is valid)
                        if not torch.isnan(loss).any() and not torch.isinf(loss).any():
                            total_loss += loss.item()
                            total_psnr += psnr.item() if isinstance(psnr, torch.Tensor) else psnr
                            total_ssim += ssim.item() if isinstance(ssim, torch.Tensor) else ssim
                            total_bpp += bpp
                            successful_batches += 1
                        
                        num_batches += 1
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}" if not torch.isnan(loss).any() and not torch.isinf(loss).any() else "N/A",
                            'psnr': f"{psnr.item() if isinstance(psnr, torch.Tensor) else psnr:.2f}",
                            'success': f"{successful_batches}/{num_batches}"
                        })
                    
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"CUDA OOM in validation batch {batch_idx}. Skipping...")
                            torch.cuda.empty_cache()
                        else:
                            print(f"Runtime error in validation batch {batch_idx}: {str(e)}")
                        error_batches += 1
                        pbar.update(1)
                        continue
                        
                except Exception as e:
                    print(f"Error processing validation batch {batch_idx}: {str(e)}")
                    error_batches += 1
                    pbar.update(1)
                    continue
                
        except Exception as e:
            print(f"Error during validation: {str(e)}")
        
        finally:
            pbar.close()
    
    # Calculate averages
    elapsed_time = time.time() - start_time
    avg_loss = total_loss / max(successful_batches, 1)
    avg_psnr = total_psnr / max(successful_batches, 1)
    avg_ssim = total_ssim / max(successful_batches, 1)
    avg_bpp = total_bpp / max(successful_batches, 1)
    
    # Average task metrics
    avg_task_metrics = {k: v / max(successful_batches, 1) for k, v in task_metrics.items()}
    
    # Log metrics
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/PSNR', avg_psnr, epoch)
    writer.add_scalar('Val/SSIM', avg_ssim, epoch)
    writer.add_scalar('Val/BPP', avg_bpp, epoch)
    writer.add_scalar('Val/ErrorRate', error_batches / max(num_batches, 1), epoch)
    
    # Log task-specific metrics
    for k, v in avg_task_metrics.items():
        writer.add_scalar(f'Val/{k}', v, epoch)
    
    # Print validation results
    print(f"\nValidation Epoch {epoch+1} completed in {elapsed_time:.2f}s - Processed {num_batches} batches ({successful_batches} successful, {error_batches} errors)")
    print(f"Metrics: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.4f}, BPP={avg_bpp:.4f}")
    
    # Print task-specific metrics
    if args.task_type == 'detection':
        print(f"Detection mAP: {avg_task_metrics.get('mAP', 0.0):.4f}")
    elif args.task_type == 'segmentation':
        print(f"Segmentation mIoU: {avg_task_metrics.get('mean_iou', 0.0):.4f}")
    elif args.task_type == 'tracking':
        print(f"Tracking MOTA: {avg_task_metrics.get('mota', 0.0):.4f}")
    
    return {
        'loss': avg_loss,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'bpp': avg_bpp,
        'error_rate': error_batches / max(num_batches, 1),
        **avg_task_metrics
    }


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
            train_metrics = train_epoch(model, train_loader, compute_total_loss, optimizer, device, epoch, writer)
            
            # Validate if it's time
            if (epoch + 1) % args.val_interval == 0:
                val_metrics = validate(model, val_loader, compute_total_loss, device, epoch, writer)
                
                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    checkpoint_path = Path(args.output_dir) / f"best_model_loss.pth"
                    save_model(model, str(checkpoint_path), optimizer, epoch, val_metrics)
                    print(f"Saved best model (by loss) to {checkpoint_path}")
                
                if val_metrics['psnr'] > best_val_psnr:
                    best_val_psnr = val_metrics['psnr']
                    checkpoint_path = Path(args.output_dir) / f"best_model_psnr.pth"
                    save_model(model, str(checkpoint_path), optimizer, epoch, val_metrics)
                    print(f"Saved best model (by PSNR) to {checkpoint_path}")
            
            # Save last model at regular intervals
            if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                checkpoint_path = Path(args.output_dir) / f"model_epoch_{epoch+1}.pth"
                save_model(model, str(checkpoint_path), optimizer, epoch, train_metrics)
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