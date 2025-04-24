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
    from utils.data_utils import get_dataloader, get_transforms, MOT16DataAdapter
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
        
        return memory_allocated, memory_reserved
    return 0, 0


def manage_gpu_memory(threshold_usage=0.9):
    """Monitor and cleanup GPU memory when needed"""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        current_usage = current_memory / max_memory if max_memory > 0 else 0
        
        if current_usage > threshold_usage:
            print(f"High GPU memory usage detected: {current_usage:.2f}. Cleaning up...")
            torch.cuda.empty_cache()
            return True
    return False


def process_batch_with_fallback(model, batch, device, reduce_batch_on_oom=True):
    """Process a batch with OOM fallback by splitting it if needed"""
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
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
        elif isinstance(labels, dict):
            labels = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in labels.items()}
        elif isinstance(labels, list):
            # Convert list of tensors to device
            labels = [x.to(device) if isinstance(x, torch.Tensor) else x for x in labels]
    if qp is not None and isinstance(qp, torch.Tensor):
        qp = qp.to(device)
    
    try:
        # Forward pass
        outputs = model(frames, qp)
        return outputs, frames, labels, qp
        
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() and reduce_batch_on_oom and frames is not None:
            print(f"CUDA OOM detected. Attempting to reduce batch size and retry...")
            torch.cuda.empty_cache()
            
            # Try with half the batch size
            batch_size = frames.shape[0]
            half_size = batch_size // 2
            
            if half_size < 1:
                print("Cannot reduce batch size further (already at 1). Skipping batch.")
                return None, frames, labels, qp
            
            print(f"Trying with reduced batch size: {batch_size} -> {half_size}")
            
            # Process first half
            if isinstance(frames, torch.Tensor):
                frames_1 = frames[:half_size]
            else:
                frames_1 = frames
                
            if isinstance(labels, torch.Tensor):
                labels_1 = labels[:half_size]
            elif isinstance(labels, list):
                labels_1 = labels[:half_size] if len(labels) >= half_size else labels
            elif isinstance(labels, dict):
                # This is more complex, may need custom handling per dataset
                labels_1 = labels
            else:
                labels_1 = labels
                
            if isinstance(qp, torch.Tensor):
                qp_1 = qp[:half_size]
            else:
                qp_1 = qp
            
            try:
                outputs_1 = model(frames_1, qp_1)
                
                # Process second half
                if isinstance(frames, torch.Tensor):
                    frames_2 = frames[half_size:]
                else:
                    frames_2 = frames
                    
                if isinstance(labels, torch.Tensor):
                    labels_2 = labels[half_size:]
                elif isinstance(labels, list):
                    labels_2 = labels[half_size:] if len(labels) >= batch_size else labels
                elif isinstance(labels, dict):
                    # This is more complex, may need custom handling per dataset
                    labels_2 = labels
                else:
                    labels_2 = labels
                    
                if isinstance(qp, torch.Tensor):
                    qp_2 = qp[half_size:]
                else:
                    qp_2 = qp
                
                outputs_2 = model(frames_2, qp_2)
                
                # Combine results
                outputs = {}
                for key in outputs_1:
                    if isinstance(outputs_1[key], torch.Tensor) and isinstance(outputs_2[key], torch.Tensor):
                        # Cat the tensors along batch dimension
                        outputs[key] = torch.cat([outputs_1[key], outputs_2[key]], dim=0)
                    else:
                        # For non-tensor outputs, just use the first part (may need custom handling)
                        outputs[key] = outputs_1[key]
                
                print("Successfully processed batch by splitting it")
                return outputs, frames, labels, qp
                
            except RuntimeError as e2:
                print(f"Still encountered OOM after reducing batch size: {str(e2)}")
                torch.cuda.empty_cache()
                return None, frames, labels, qp
        else:
            # Not OOM or can't reduce further
            print(f"Error in forward pass: {str(e)}")
            return None, frames, labels, qp


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None, task_type=None):
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
        task_type: Type of task (detection, segmentation, tracking)
        
    Returns:
        Average loss, PSNR, SSIM, BPP, and loss components for the epoch
    """
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_bpp = 0.0
    total_task_loss = 0.0
    total_recon_loss = 0.0
    total_bitrate_loss = 0.0
    processed_batches = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    # Track start time
    start_time = time.time()
    
    for i, batch in enumerate(pbar):
        try:
            # Check and cleanup GPU memory periodically
            if i % 5 == 0 and torch.cuda.is_available():
                manage_gpu_memory(threshold_usage=0.8)
            
            # Forward pass with OOM handling
            optimizer.zero_grad()
            outputs, frames, labels, qp = process_batch_with_fallback(model, batch, device)
            
            # Skip this batch if processing failed
            if outputs is None:
                print(f"Skipping batch {i} due to processing failure")
                continue
                
            # Extract components from the outputs dictionary
            reconstructed = outputs['reconstructed']
            task_output = outputs['task_output']
            bitrate = outputs['bitrate']
            
            # Removed print shapes for debugging on the first batch
            
            # Calculate loss
            try:
                loss = criterion(task_output, labels, reconstructed, frames, bitrate, task_type=task_type)
                
                # Backward and optimize
                loss.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Calculate metrics with error handling
                with torch.no_grad():
                    try:
                        # Calculate PSNR and SSIM
                        psnr = compute_psnr(reconstructed, frames)
                        ssim = compute_ssim(reconstructed, frames)
                        bpp = compute_bpp(bitrate, frames)
                        
                        # Ensure metrics are finite
                        if not torch.isfinite(psnr):
                            print(f"Non-finite PSNR detected: {psnr.item() if isinstance(psnr, torch.Tensor) else psnr}")
                            psnr = torch.tensor(0.0, device=device)
                        
                        if not torch.isfinite(ssim):
                            print(f"Non-finite SSIM detected: {ssim.item() if isinstance(ssim, torch.Tensor) else ssim}")
                            ssim = torch.tensor(0.0, device=device)
                            
                        if not torch.isfinite(bpp):
                            print(f"Non-finite BPP detected: {bpp.item() if isinstance(bpp, torch.Tensor) else bpp}")
                            bpp = torch.tensor(0.1, device=device)
                        
                        # Get individual loss components (if available)
                        if hasattr(loss, 'get') and callable(getattr(loss, 'get', None)):
                            # Dictionary-style loss
                            task_loss = loss.get('task', torch.tensor(0.0, device=device))
                            recon_loss = loss.get('recon', torch.tensor(0.0, device=device))
                            bitrate_loss = loss.get('bitrate', torch.tensor(0.0, device=device))
                        else:
                            # For standard Python dict
                            if isinstance(loss, dict):
                                task_loss = loss.get('task', torch.tensor(0.0, device=device))
                                recon_loss = loss.get('recon', torch.tensor(0.0, device=device))
                                bitrate_loss = loss.get('bitrate', torch.tensor(0.0, device=device))
                            else:
                                # Scalar loss
                                task_loss = torch.tensor(0.0, device=device)
                                recon_loss = torch.tensor(0.0, device=device)
                                bitrate_loss = torch.tensor(0.0, device=device)
                        
                        # Update running totals
                        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                        total_loss += loss_val
                        total_psnr += psnr.item() if isinstance(psnr, torch.Tensor) else psnr
                        total_ssim += ssim.item() if isinstance(ssim, torch.Tensor) else ssim
                        total_bpp += bpp.item() if isinstance(bpp, torch.Tensor) else bpp
                        
                        # Properly accumulate loss components
                        total_task_loss += task_loss.item() if isinstance(task_loss, torch.Tensor) else (task_loss if isinstance(task_loss, (int, float)) else 0.0)
                        total_recon_loss += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else (recon_loss if isinstance(recon_loss, (int, float)) else 0.0)
                        total_bitrate_loss += bitrate_loss.item() if isinstance(bitrate_loss, torch.Tensor) else (bitrate_loss if isinstance(bitrate_loss, (int, float)) else 0.0)
                        
                        processed_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f"{loss_val:.4f}",
                            'psnr': f"{psnr.item() if isinstance(psnr, torch.Tensor) else psnr:.2f}",
                            'bpp': f"{bpp.item() if isinstance(bpp, torch.Tensor) else bpp:.4f}"
                        })
                        
                        # Log to TensorBoard
                        if writer is not None and i % 10 == 0:  # Log every 10 batches
                            step = epoch * len(dataloader) + i
                            writer.add_scalar('train/loss', loss_val, step)
                            writer.add_scalar('train/psnr', psnr.item() if isinstance(psnr, torch.Tensor) else psnr, step)
                            writer.add_scalar('train/ssim', ssim.item() if isinstance(ssim, torch.Tensor) else ssim, step)
                            writer.add_scalar('train/bpp', bpp.item() if isinstance(bpp, torch.Tensor) else bpp, step)
                            
                            # Log loss components
                            writer.add_scalar('train/task_loss', task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss, step)
                            writer.add_scalar('train/recon_loss', recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss, step)
                            writer.add_scalar('train/bitrate_loss', bitrate_loss.item() if isinstance(bitrate_loss, torch.Tensor) else bitrate_loss, step)
                            
                            # Log GPU memory
                            log_gpu_memory(writer, step, device)
                    
                    except Exception as e:
                        print(f"Error calculating metrics: {e}")
                        torch.cuda.empty_cache()  # Clean up memory just in case
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"CUDA OOM during loss/backward. Skipping batch {i}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Runtime error during loss/backward: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error in loss calculation or backward pass: {e}")
                continue
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Try to clean up any GPU memory in case this helps with next batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # Calculate averages
    if processed_batches > 0:
        avg_loss = total_loss / processed_batches
        avg_psnr = total_psnr / processed_batches
        avg_ssim = total_ssim / processed_batches
        avg_bpp = total_bpp / processed_batches
        avg_task_loss = total_task_loss / processed_batches
        avg_recon_loss = total_recon_loss / processed_batches
        avg_bitrate_loss = total_bitrate_loss / processed_batches
    else:
        avg_loss = float('inf')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_bpp = 0.0
        avg_task_loss = 0.0
        avg_recon_loss = 0.0
        avg_bitrate_loss = 0.0
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print epoch summary - keep important logs
    print(f"Epoch {epoch} [Train] - Avg Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, BPP: {avg_bpp:.4f}, Time: {elapsed_time:.2f}s")
    print(f"Processed {processed_batches}/{len(dataloader)} batches")
    print(f"Epoch {epoch} [Train] Loss Components Summary - Task: {avg_task_loss:.4f}, Recon: {avg_recon_loss:.4f}, Bitrate: {avg_bitrate_loss:.4f}")
    
    return avg_loss, avg_psnr, avg_ssim, avg_bpp, avg_task_loss, avg_recon_loss, avg_bitrate_loss


def validate(model, dataloader, criterion, device, epoch, writer=None, task_type=None):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader containing validation data
        criterion: Loss function
        device: Device to use for validation
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter
        task_type: Type of task (detection, segmentation, tracking)
        
    Returns:
        Average loss, PSNR, SSIM, BPP, and loss components for the validation set
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_bpp = 0.0
    total_task_loss = 0.0
    total_recon_loss = 0.0
    total_bitrate_loss = 0.0
    processed_batches = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            try:
                # Process batch with OOM handling
                outputs, frames, labels, qp = process_batch_with_fallback(model, batch, device)
                
                # Skip this batch if processing failed
                if outputs is None:
                    print(f"Skipping validation batch {i} due to processing failure")
                    continue
                
                # Extract components from the outputs dictionary
                reconstructed = outputs['reconstructed']
                task_output = outputs['task_output']
                bitrate = outputs['bitrate']
                
                # Calculate loss
                try:
                    loss = criterion(task_output, labels, reconstructed, frames, bitrate, task_type=task_type)
                    
                    # Calculate metrics
                    try:
                        # Calculate PSNR and SSIM
                        psnr = compute_psnr(reconstructed, frames)
                        ssim = compute_ssim(reconstructed, frames)
                        bpp = compute_bpp(bitrate, frames)
                        
                        # Ensure metrics are finite
                        if not torch.isfinite(psnr):
                            print(f"Non-finite PSNR detected: {psnr.item() if isinstance(psnr, torch.Tensor) else psnr}")
                            psnr = torch.tensor(0.0, device=device)
                        
                        if not torch.isfinite(ssim):
                            print(f"Non-finite SSIM detected: {ssim.item() if isinstance(ssim, torch.Tensor) else ssim}")
                            ssim = torch.tensor(0.0, device=device)
                            
                        if not torch.isfinite(bpp):
                            print(f"Non-finite BPP detected: {bpp.item() if isinstance(bpp, torch.Tensor) else bpp}")
                            bpp = torch.tensor(0.1, device=device)
                        
                        # Get individual loss components (if available)
                        if hasattr(loss, 'get') and callable(getattr(loss, 'get', None)):
                            # Dictionary-style loss
                            task_loss = loss.get('task', torch.tensor(0.0, device=device))
                            recon_loss = loss.get('recon', torch.tensor(0.0, device=device))
                            bitrate_loss = loss.get('bitrate', torch.tensor(0.0, device=device))
                        else:
                            # For standard Python dict
                            if isinstance(loss, dict):
                                task_loss = loss.get('task', torch.tensor(0.0, device=device))
                                recon_loss = loss.get('recon', torch.tensor(0.0, device=device))
                                bitrate_loss = loss.get('bitrate', torch.tensor(0.0, device=device))
                            else:
                                # Scalar loss
                                task_loss = torch.tensor(0.0, device=device)
                                recon_loss = torch.tensor(0.0, device=device)
                                bitrate_loss = torch.tensor(0.0, device=device)
                        
                        # Update running totals
                        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
                        total_loss += loss_val
                        total_psnr += psnr.item() if isinstance(psnr, torch.Tensor) else psnr
                        total_ssim += ssim.item() if isinstance(ssim, torch.Tensor) else ssim
                        total_bpp += bpp.item() if isinstance(bpp, torch.Tensor) else bpp
                        
                        # Properly accumulate loss components
                        total_task_loss += task_loss.item() if isinstance(task_loss, torch.Tensor) else (task_loss if isinstance(task_loss, (int, float)) else 0.0)
                        total_recon_loss += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else (recon_loss if isinstance(recon_loss, (int, float)) else 0.0)
                        total_bitrate_loss += bitrate_loss.item() if isinstance(bitrate_loss, torch.Tensor) else (bitrate_loss if isinstance(bitrate_loss, (int, float)) else 0.0)
                        
                        processed_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f"{loss_val:.4f}",
                            'psnr': f"{psnr.item() if isinstance(psnr, torch.Tensor) else psnr:.2f}",
                            'bpp': f"{bpp.item() if isinstance(bpp, torch.Tensor) else bpp:.4f}"
                        })
                    
                    except Exception as e:
                        print(f"Error calculating validation metrics: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                
                except Exception as e:
                    print(f"Error in validation loss calculation: {e}")
                    continue
            
            except Exception as e:
                print(f"Error processing validation batch {i}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
    # Calculate averages
    if processed_batches > 0:
        avg_loss = total_loss / processed_batches
        avg_psnr = total_psnr / processed_batches
        avg_ssim = total_ssim / processed_batches
        avg_bpp = total_bpp / processed_batches
        avg_task_loss = total_task_loss / processed_batches
        avg_recon_loss = total_recon_loss / processed_batches
        avg_bitrate_loss = total_bitrate_loss / processed_batches
    else:
        avg_loss = float('inf')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_bpp = 0.0
        avg_task_loss = 0.0
        avg_recon_loss = 0.0
        avg_bitrate_loss = 0.0
    
    # Print validation summary (keep these important logs)
    print(f"Epoch {epoch} [Val] - Avg Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, BPP: {avg_bpp:.4f}")
    print(f"Processed {processed_batches}/{len(dataloader)} validation batches")
    print(f"Epoch {epoch} [Val] Loss Components Summary - Task: {avg_task_loss:.4f}, Recon: {avg_recon_loss:.4f}, Bitrate: {avg_bitrate_loss:.4f}")
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/psnr', avg_psnr, epoch)
        writer.add_scalar('val/ssim', avg_ssim, epoch)
        writer.add_scalar('val/bpp', avg_bpp, epoch)
        
        # Log loss components
        writer.add_scalar('val/task_loss', avg_task_loss, epoch)
        writer.add_scalar('val/recon_loss', avg_recon_loss, epoch)
        writer.add_scalar('val/bitrate_loss', avg_bitrate_loss, epoch)
    
    return avg_loss, avg_psnr, avg_ssim, avg_bpp, avg_task_loss, avg_recon_loss, avg_bitrate_loss


def main(args):
    """Main training function"""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    print("Debug: Starting main function")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if dataset is MOT16 and needs conversion
    if args.dataset.endswith('MOT16') and args.task_type == 'tracking':
        # Check if the dataset has already been converted
        processed_dir = Path(args.dataset).parent / 'MOT16_processed'
        
        if not (processed_dir / 'tracking' / 'train').exists() or not (processed_dir / 'tracking' / 'test').exists():
            print(f"Raw MOT16 dataset detected at {args.dataset}")
            print(f"Converting to expected format at {processed_dir}")
            
            # Create the processed directory
            os.makedirs(processed_dir, exist_ok=True)
            
            # Convert train split
            train_adapter = MOT16DataAdapter(
                mot_root=args.dataset,
                output_root=processed_dir,
                seq_length=args.seq_length,
                split='train',
                stride=1
            )
            if train_adapter.sequences:
                train_adapter.convert()
                
            # Convert test split if needed
            test_adapter = MOT16DataAdapter(
                mot_root=args.dataset,
                output_root=processed_dir,
                seq_length=args.seq_length,
                split='test',
                stride=1
            )
            if test_adapter.sequences:
                test_adapter.convert()
                
            print(f"MOT16 dataset converted to expected format. Using: {processed_dir}")
            # Update the dataset path to the processed directory
            args.dataset = str(processed_dir)
        else:
            print(f"Using pre-processed MOT16 dataset at: {processed_dir}")
            args.dataset = str(processed_dir)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(args)
    print(f"Created {args.task_type} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create transforms
    print("Debug: Creating transforms")
    train_transform, val_transform = get_transforms(
        task_type=args.task_type,
        resolution=(256, 256),  # Example resolution
        augment=True
    )
    
    # Create dataloaders
    print(f"Debug: Creating training dataloader from {args.dataset}")
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
    
    print(f"Debug: Creating validation dataloader from {args.dataset}")
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
        print("Debug: Creating subset for dry run")
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
            train_loss, train_psnr, train_ssim, train_bpp, train_task_loss, train_recon_loss, train_bitrate_loss = train_epoch(
                model, train_loader, compute_total_loss, optimizer, device, epoch, writer, task_type=args.task_type
            )
            
            # Explicitly print training loss components
            print(f"Epoch {epoch} [Train] Loss Components Summary - Task: {train_task_loss:.4f}, Recon: {train_recon_loss:.4f}, Bitrate: {train_bitrate_loss:.4f}")
            
            # Log epoch-level metrics to TensorBoard
            if writer is not None:
                writer.add_scalar('train_epoch/loss', train_loss, epoch)
                writer.add_scalar('train_epoch/psnr', train_psnr, epoch)
                writer.add_scalar('train_epoch/ssim', train_ssim, epoch)
                writer.add_scalar('train_epoch/bpp', train_bpp, epoch)
                writer.add_scalar('train_epoch/task_loss', train_task_loss, epoch)
                writer.add_scalar('train_epoch/recon_loss', train_recon_loss, epoch)
                writer.add_scalar('train_epoch/bitrate_loss', train_bitrate_loss, epoch)
            
            # Validate if it's time
            if (epoch + 1) % args.val_interval == 0:
                val_loss, val_psnr, val_ssim, val_bpp, val_task_loss, val_recon_loss, val_bitrate_loss = validate(
                    model, val_loader, compute_total_loss, device, epoch, writer, task_type=args.task_type
                )
                
                # Explicitly print validation loss components
                print(f"Epoch {epoch} [Val] Loss Components Summary - Task: {val_task_loss:.4f}, Recon: {val_recon_loss:.4f}, Bitrate: {val_bitrate_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = Path(args.output_dir) / f"best_model_loss.pth"
                    save_model(model, str(checkpoint_path), optimizer, epoch, {
                        'loss': val_loss, 
                        'psnr': val_psnr, 
                        'ssim': val_ssim, 
                        'bpp': val_bpp,
                        'task_loss': val_task_loss,
                        'recon_loss': val_recon_loss,
                        'bitrate_loss': val_bitrate_loss
                    })
                    print(f"Saved best model (by loss) to {checkpoint_path}")
                
                if val_psnr > best_val_psnr:
                    best_val_psnr = val_psnr
                    checkpoint_path = Path(args.output_dir) / f"best_model_psnr.pth"
                    save_model(model, str(checkpoint_path), optimizer, epoch, {
                        'loss': val_loss, 
                        'psnr': val_psnr, 
                        'ssim': val_ssim, 
                        'bpp': val_bpp,
                        'task_loss': val_task_loss,
                        'recon_loss': val_recon_loss,
                        'bitrate_loss': val_bitrate_loss
                    })
                    print(f"Saved best model (by PSNR) to {checkpoint_path}")
            
            # Save last model at regular intervals
            if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                checkpoint_path = Path(args.output_dir) / f"model_epoch_{epoch+1}.pth"
                save_model(model, str(checkpoint_path), optimizer, epoch, {
                    'loss': train_loss, 
                    'psnr': train_psnr, 
                    'ssim': train_ssim, 
                    'bpp': train_bpp,
                    'task_loss': train_task_loss,
                    'recon_loss': train_recon_loss,
                    'bitrate_loss': train_bitrate_loss
                })
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