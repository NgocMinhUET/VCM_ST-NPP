#!/usr/bin/env python
"""
MOT16 Dataset Converter

This script converts the MOT16 dataset to the format expected by the task-aware
video compression model. It extracts frames and annotations from the MOT16 dataset
and organizes them in the expected directory structure.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from PIL import Image
import cv2
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.data_utils import MOT16DataAdapter


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert MOT16 dataset to the format expected by the task-aware video compression model"
    )
    parser.add_argument("--mot_root", type=str, required=True,
                        help="Path to MOT16 dataset")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Number of frames in each sequence")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for frame sampling")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the converted dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        help="Splits to convert")
    
    return parser.parse_args()


def main():
    """
    Main function.
    """
    args = parse_args()
    
    # Verify that the MOT16 directory exists
    mot_root = Path(args.mot_root)
    if not mot_root.exists():
        print(f"Error: MOT16 directory not found at {mot_root}")
        return 1
    
    # Create output directory
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in args.splits:
        print(f"Processing {split} split...")
        
        # Create adapter
        adapter = MOT16DataAdapter(
            mot_root=mot_root,
            output_root=output_root,
            seq_length=args.seq_length,
            split=split,
            stride=args.stride
        )
        
        # Check if sequences were found
        if not adapter.sequences:
            print(f"No MOT16 sequences found in {split} split. Skipping.")
            continue
        
        # Convert dataset
        adapter.convert()
        
        # Verify dataset if requested
        if args.verify:
            print(f"Verifying {split} split...")
            success = adapter.verify()
            if not success:
                print(f"Verification failed for {split} split.")
            else:
                print(f"Verification successful for {split} split.")
    
    # Create validation split from training
    train_dir = output_root / "tracking" / "train"
    val_dir = output_root / "tracking" / "val"
    
    if train_dir.exists() and not val_dir.exists():
        print("Creating validation split from training data...")
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of training sequences
        train_seqs = [d for d in train_dir.glob("*") if d.is_dir()]
        
        # Use 20% of training sequences for validation
        val_seqs = train_seqs[:max(1, len(train_seqs) // 5)]
        
        for seq_dir in val_seqs:
            # Copy sequence to validation directory
            dst_dir = val_dir / seq_dir.name
            if not dst_dir.exists():  # Only copy if not already present
                print(f"Copying {seq_dir.name} to validation split...")
                shutil.copytree(seq_dir, dst_dir)
    
    print("\nConversion complete.")
    print(f"Processed dataset saved to {output_root}")
    print("\nTo use this dataset, run:")
    print(f"python train.py --dataset {output_root} --task_type tracking")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 