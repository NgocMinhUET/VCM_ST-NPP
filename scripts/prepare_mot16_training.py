#!/usr/bin/env python
"""
MOT16 Dataset Preparation and Training Script

This script prepares the MOT16 dataset and starts training the task-aware video
compression model with tracking task.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare MOT16 dataset and start training"
    )
    parser.add_argument("--mot_root", type=str, required=True,
                        help="Path to MOT16 dataset")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path to processed dataset output directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--qp", type=int, default=30,
                        help="Quantization parameter")
    parser.add_argument("--random_qp", action="store_true",
                        help="Use random QP values during training")
    parser.add_argument("--qp_range", type=str, default="22,37",
                        help="QP range for random QP (comma separated min,max)")
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Number of frames in each sequence")
    parser.add_argument("--task_weight", type=float, default=1.0,
                        help="Weight for tracking task loss")
    parser.add_argument("--recon_weight", type=float, default=1.0,
                        help="Weight for reconstruction loss")
    parser.add_argument("--bitrate_weight", type=float, default=0.1,
                        help="Weight for bitrate loss")
    parser.add_argument("--output_dir", type=str, default="checkpoints/mot16",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading")
    parser.add_argument("--skip_conversion", action="store_true",
                        help="Skip dataset conversion (use existing processed data)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run with minimal data for testing")
    
    return parser.parse_args()


def main():
    """
    Main function.
    """
    args = parse_args()
    
    # Create output directories
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Parse QP range
    if args.qp_range:
        qp_min, qp_max = map(int, args.qp_range.split(","))
    else:
        qp_min, qp_max = 22, 37
    
    # Step 1: Convert MOT16 dataset to the expected format
    if not args.skip_conversion:
        print("\n=== Step 1: Converting MOT16 dataset ===\n")
        
        # Run conversion script
        convert_cmd = [
            sys.executable,
            str(Path(__file__).parent / "convert_mot16.py"),
            "--mot_root", args.mot_root,
            "--output_root", str(output_root),
            "--seq_length", str(args.seq_length),
            "--stride", "1",
            "--verify"
        ]
        
        try:
            subprocess.run(convert_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during dataset conversion: {e}")
            return 1
    else:
        print("\n=== Step 1: Skipping dataset conversion ===\n")
    
    # Step 2: Start training
    print("\n=== Step 2: Starting training ===\n")
    
    # Prepare training command
    train_cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "train.py"),
        "--dataset", str(output_root),
        "--task_type", "tracking",
        "--seq_length", str(args.seq_length),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--qp", str(args.qp),
        "--task_weight", str(args.task_weight),
        "--recon_weight", str(args.recon_weight),
        "--bitrate_weight", str(args.bitrate_weight),
        "--output_dir", str(args.output_dir),
        "--num_workers", str(args.num_workers)
    ]
    
    if args.random_qp:
        train_cmd.append("--random_qp")
        train_cmd.extend(["--qp_range", f"{qp_min},{qp_max}"])
    
    if args.dry_run:
        train_cmd.append("--dry_run")
    
    # Run training
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return 1
    
    print("\n=== Training completed successfully ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 