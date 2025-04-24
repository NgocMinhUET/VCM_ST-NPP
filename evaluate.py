#!/usr/bin/env python
"""
Evaluation script for task-aware video compression model.

This script evaluates a combined model on a dataset for
compression quality and downstream task performance.
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

# Import project modules
from models.combined_model import CombinedModel
from utils.data_utils import get_dataloader, get_transforms
from utils.model_utils import load_model
from utils.metric_utils import (
    compute_psnr, compute_ssim, compute_bpp,
    evaluate_detection, evaluate_segmentation, evaluate_tracking
)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate task-aware video compression model")
    
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Dataset name or path")
    parser.add_argument("--task", type=str, required=True,
                        choices=["detection", "segmentation", "tracking"],
                        help="Type of downstream task")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--qp", type=int, default=30,
                        help="Quantization parameter")
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Number of frames in each sequence")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save visualization of results")
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, task_type, device):
    """Load model from checkpoint"""
    # Create model
    model = CombinedModel(
        task_type=task_type,
        hidden_channels=128,  # Default value, should be in the checkpoint
        seq_length=5  # Default value, should be in the checkpoint
    )
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = load_model(model, checkpoint_path)
    
    # If the checkpoint contains model configuration, we could update it here
    
    return model.to(device)


def evaluate(model, dataloader, task_type, device, save_dir=None, save_visualizations=False):
    """
    Evaluate the model on the given dataloader
    
    Args:
        model: The model to evaluate
        dataloader: Data loader for evaluation data
        task_type: Type of task (detection, segmentation, tracking)
        device: Device to use for tensors
        save_dir: Directory to save results
        save_visualizations: Whether to save result visualizations
    
    Returns:
        Dict containing metrics
    """
    model.eval()
    
    # Initialize metrics accumulators
    total_psnr = 0.0
    total_ssim = 0.0
    total_bpp = 0.0
    
    # Task-specific metrics
    task_metrics = {}
    
    # Counter for number of batches
    num_batches = 0
    
    # Create progress bar
    pbar = tqdm(total=len(dataloader), desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Extract data from batch
                frames = batch['frames'].to(device)
                labels = batch['labels']
                qp = batch['qp'].to(device)
                
                # Forward pass
                output = model(frames, qp)
                
                # Get output components
                reconstructed = output['reconstructed']
                task_output = output['task_output']
                bitrate = output['bitrate']
                
                # Convert labels to appropriate format based on task type
                if task_type == 'detection':
                    # Process detection labels
                    task_labels = torch.zeros_like(task_output)
                    # This is simplified
                
                elif task_type == 'segmentation':
                    # Process segmentation labels
                    task_labels = torch.zeros_like(task_output)
                    # This is simplified
                
                elif task_type == 'tracking':
                    # Process tracking labels
                    task_labels = torch.zeros_like(task_output)
                    # This is simplified
                
                # Calculate metrics
                psnr = compute_psnr(frames, reconstructed)
                ssim = compute_ssim(frames, reconstructed)
                bpp = compute_bpp(bitrate.mean().item(), frames.size(3), frames.size(4), frames.size(2))
                
                # Calculate task-specific metrics
                if task_type == 'detection':
                    batch_task_metrics = evaluate_detection(task_output, task_labels)
                elif task_type == 'segmentation':
                    batch_task_metrics = evaluate_segmentation(task_output, task_labels)
                elif task_type == 'tracking':
                    batch_task_metrics = evaluate_tracking(task_output, task_labels)
                
                # Update task metrics
                for k, v in batch_task_metrics.items():
                    if k not in task_metrics:
                        task_metrics[k] = 0.0
                    task_metrics[k] += v
                
                # Accumulate metrics
                total_psnr += psnr.item() if isinstance(psnr, torch.Tensor) else psnr
                total_ssim += ssim.item() if isinstance(ssim, torch.Tensor) else ssim
                total_bpp += bpp
                num_batches += 1
                
                # Save visualizations if requested
                if save_visualizations and save_dir:
                    # Code to save visualizations would go here
                    pass
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'psnr': f"{psnr.item() if isinstance(psnr, torch.Tensor) else psnr:.2f}",
                    'bpp': f"{bpp:.4f}"
                })
            
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {str(e)}")
                continue
    
    pbar.close()
    
    # Calculate averages
    avg_psnr = total_psnr / max(num_batches, 1)
    avg_ssim = total_ssim / max(num_batches, 1)
    avg_bpp = total_bpp / max(num_batches, 1)
    
    # Average task metrics
    avg_task_metrics = {k: v / max(num_batches, 1) for k, v in task_metrics.items()}
    
    # Combined metrics dictionary
    metrics = {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'bpp': avg_bpp,
        **avg_task_metrics
    }
    
    return metrics


def main(args):
    """Main evaluation function"""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    if args.save_visualizations:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.task, device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create transforms
    _, val_transform = get_transforms(
        task_type=args.task,
        resolution=(256, 256),  # Example resolution
        augment=False
    )
    
    # Create dataloader
    dataloader = get_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        task_type=args.task,
        split='val',  # Use validation split for evaluation
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        random_qp=False,
        qp_range=(args.qp, args.qp),  # Use single QP value
        shuffle=False,
        transform=val_transform
    )
    
    print(f"Created dataloader with {len(dataloader)} evaluation batches")
    
    # Evaluate model
    start_time = time.time()
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        task_type=args.task,
        device=device,
        save_dir=args.output_dir if args.save_visualizations else None,
        save_visualizations=args.save_visualizations
    )
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "="*50)
    print(f"Evaluation Results ({args.task}):")
    print("="*50)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"BPP: {metrics['bpp']:.4f}")
    
    # Print task-specific metrics
    if args.task == 'detection':
        print(f"mAP: {metrics.get('mAP', 0.0):.4f}")
        print(f"Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"Recall: {metrics.get('recall', 0.0):.4f}")
    elif args.task == 'segmentation':
        print(f"mIoU: {metrics.get('mean_iou', 0.0):.4f}")
        print(f"Pixel Accuracy: {metrics.get('pixel_acc', 0.0):.4f}")
    elif args.task == 'tracking':
        print(f"MOTA: {metrics.get('mota', 0.0):.4f}")
        print(f"IDF1: {metrics.get('idf1', 0.0):.4f}")
    
    print("="*50)
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    
    # Save results to file if output directory is specified
    if args.output_dir:
        results_file = Path(args.output_dir) / f"{args.task}_results.txt"
        with open(results_file, "w") as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Task: {args.task}\n")
            f.write(f"QP: {args.qp}\n\n")
            f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
            f.write(f"SSIM: {metrics['ssim']:.4f}\n")
            f.write(f"BPP: {metrics['bpp']:.4f}\n\n")
            
            if args.task == 'detection':
                f.write(f"mAP: {metrics.get('mAP', 0.0):.4f}\n")
                f.write(f"Precision: {metrics.get('precision', 0.0):.4f}\n")
                f.write(f"Recall: {metrics.get('recall', 0.0):.4f}\n")
            elif args.task == 'segmentation':
                f.write(f"mIoU: {metrics.get('mean_iou', 0.0):.4f}\n")
                f.write(f"Pixel Accuracy: {metrics.get('pixel_acc', 0.0):.4f}\n")
            elif args.task == 'tracking':
                f.write(f"MOTA: {metrics.get('mota', 0.0):.4f}\n")
                f.write(f"IDF1: {metrics.get('idf1', 0.0):.4f}\n")
        
        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args) 