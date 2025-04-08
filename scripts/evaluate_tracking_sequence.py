#!/usr/bin/env python3
"""
Script to evaluate tracking performance on image sequences using the improved compression method.
Supports both MOT16 dataset format and custom image sequence directories.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import compression and tracking models
from models.improved_autoencoder import ImprovedAutoencoder
from models.tracking import TrackingModel
from utils.metrics import calculate_psnr, calculate_ssim, calculate_tracking_metrics
from utils.visualization import visualize_tracking_results

class ImageSequenceDataset:
    """Dataset class for loading image sequences."""
    def __init__(
        self,
        sequence_dir: str,
        gt_path: Optional[str] = None,
        frame_pattern: str = "*.jpg",
        time_steps: int = 16
    ):
        self.sequence_dir = Path(sequence_dir)
        self.gt_path = Path(gt_path) if gt_path else None
        self.frame_pattern = frame_pattern
        self.time_steps = time_steps
        
        # Load image paths
        self.image_paths = sorted(list(self.sequence_dir.glob(frame_pattern)))
        if not self.image_paths:
            raise ValueError(f"No images found in {sequence_dir} with pattern {frame_pattern}")
        
        # Load ground truth if available
        self.gt_data = None
        if self.gt_path and self.gt_path.exists():
            self.gt_data = self._load_ground_truth()
    
    def _load_ground_truth(self) -> Dict:
        """Load ground truth data from file."""
        try:
            with open(self.gt_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load ground truth data: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def get_sequence_batch(
        self,
        start_idx: int,
        batch_size: int
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Get a batch of consecutive frames and corresponding ground truth."""
        end_idx = min(start_idx + batch_size, len(self))
        frames = []
        
        for idx in range(start_idx, end_idx):
            img = cv2.imread(str(self.image_paths[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        
        # Pad sequence if needed
        while len(frames) < batch_size:
            frames.append(frames[-1].clone())
        
        batch = torch.stack(frames)
        
        # Get corresponding ground truth if available
        gt_batch = None
        if self.gt_data:
            frame_ids = [self.image_paths[i].stem for i in range(start_idx, end_idx)]
            gt_batch = {frame_id: self.gt_data[frame_id] for frame_id in frame_ids
                       if frame_id in self.gt_data}
        
        return batch, gt_batch

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate tracking performance on image sequences")
    
    # Input parameters
    parser.add_argument("--sequence_dir", type=str, required=True,
                      help="Directory containing image sequence")
    parser.add_argument("--gt_path", type=str, default=None,
                      help="Path to ground truth data file (optional)")
    parser.add_argument("--frame_pattern", type=str, default="*.jpg",
                      help="Pattern to match image files")
    parser.add_argument("--time_steps", type=int, default=16,
                      help="Number of frames to process at once")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to trained compression model")
    parser.add_argument("--tracking_model", type=str, default="deepsort",
                      help="Tracking model to use (deepsort, sort, etc.)")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run evaluation on (cuda or cpu)")
    
    # Compression parameters
    parser.add_argument("--qp_values", type=str, default="22,27,32,37",
                      help="Comma-separated QP values to evaluate")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results",
                      help="Directory to save evaluation results")
    parser.add_argument("--save_videos", action="store_true",
                      help="Save compressed and tracking visualization videos")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose output")
    
    return parser.parse_args()

def evaluate_sequence(
    dataset: ImageSequenceDataset,
    compression_model: ImprovedAutoencoder,
    tracking_model: TrackingModel,
    device: torch.device,
    args: argparse.Namespace
) -> Dict:
    """Evaluate tracking performance on compressed image sequence."""
    results = {
        "compression_metrics": [],
        "tracking_metrics": [],
        "summary": {}
    }
    
    # Process sequence in batches
    for start_idx in tqdm(range(0, len(dataset), args.time_steps)):
        # Get batch of frames
        frames, gt_batch = dataset.get_sequence_batch(start_idx, args.time_steps)
        frames = frames.to(device)
        
        # Compress and decompress frames
        with torch.no_grad():
            compressed = compression_model.compress(frames)
            decompressed = compression_model.decompress(compressed)
        
        # Calculate compression metrics
        psnr = calculate_psnr(frames, decompressed)
        ssim = calculate_ssim(frames, decompressed)
        bpp = compression_model.calculate_bpp(compressed)
        
        results["compression_metrics"].append({
            "frame_start": start_idx,
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "bpp": bpp.item()
        })
        
        # Run tracking on original and compressed frames
        orig_tracks = tracking_model.track(frames.cpu().numpy())
        comp_tracks = tracking_model.track(decompressed.cpu().numpy())
        
        # Calculate tracking metrics if ground truth available
        if gt_batch:
            tracking_metrics = calculate_tracking_metrics(
                gt_batch, orig_tracks, comp_tracks
            )
            results["tracking_metrics"].append(tracking_metrics)
        
        # Save visualization if requested
        if args.save_videos:
            output_path = os.path.join(
                args.output_dir,
                f"tracking_vis_{start_idx:06d}.mp4"
            )
            visualize_tracking_results(
                frames.cpu().numpy(),
                decompressed.cpu().numpy(),
                orig_tracks,
                comp_tracks,
                output_path
            )
    
    # Calculate summary statistics
    results["summary"] = {
        "avg_psnr": np.mean([m["psnr"] for m in results["compression_metrics"]]),
        "avg_ssim": np.mean([m["ssim"] for m in results["compression_metrics"]]),
        "avg_bpp": np.mean([m["bpp"] for m in results["compression_metrics"]])
    }
    
    if results["tracking_metrics"]:
        results["summary"].update({
            "avg_mota": np.mean([m["mota"] for m in results["tracking_metrics"]]),
            "avg_motp": np.mean([m["motp"] for m in results["tracking_metrics"]]),
            "avg_id_switches": np.mean([m["id_switches"] for m in results["tracking_metrics"]])
        })
    
    return results

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    compression_model = ImprovedAutoencoder.load_from_checkpoint(args.model_path).to(device)
    compression_model.eval()
    
    tracking_model = TrackingModel(model_type=args.tracking_model)
    
    # Create dataset
    print("Loading dataset...")
    dataset = ImageSequenceDataset(
        args.sequence_dir,
        args.gt_path,
        args.frame_pattern,
        args.time_steps
    )
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluate_sequence(dataset, compression_model, tracking_model, device, args)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed! Results saved to {results_path}")
    
    # Print summary
    print("\nSummary:")
    for metric, value in results["summary"].items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 