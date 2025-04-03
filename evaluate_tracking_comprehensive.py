#!/usr/bin/env python
"""
Script to evaluate tracking performance on videos compressed with different methods.
Compares tracking metrics between original video and various compression methods.
"""

import os
import argparse
import numpy as np
import cv2
import torch
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import motmetrics as mm
from motmetrics.metrics import MOTAccumulator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tracking performance on compressed videos")
    
    # Input parameters
    parser.add_argument("--sequence_path", type=str, required=True,
                        help="Path to MOT sequence directory")
    parser.add_argument("--compression_results", type=str, required=True,
                        help="Path to compression results directory from compare_compression_methods.py")
    
    # Evaluation parameters
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to process (None=all)")
    parser.add_argument("--output_dir", type=str, 
                        default="results/tracking_comparison",
                        help="Directory to save evaluation results")
    
    return parser.parse_args()

def load_mot_groundtruth(gt_path, max_frames=None):
    """Load MOT ground truth data."""
    gt_data = {}
    try:
        with open(gt_path, 'r') as f:
            for line in f:
                frame, track_id, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
                frame = int(frame)
                if max_frames is not None and frame > max_frames:
                    continue
                track_id = int(track_id)
                if frame not in gt_data:
                    gt_data[frame] = []
                gt_data[frame].append((track_id, (x, y, w, h)))
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {gt_path}")
        # Create dummy ground truth for testing
        frame_count = 600 if max_frames is None else max_frames
        for frame in range(1, frame_count + 1):
            gt_data[frame] = [(1, (100, 100, 50, 100))]
    return gt_data

def track_object_simple(frames, initial_box):
    """A simple tracking implementation using template matching."""
    results = [initial_box]
    x, y, w, h = initial_box
    
    # Check if initial box is valid
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > frames[0].shape[1] or y+h > frames[0].shape[0]:
        # Adjust to be a valid box
        x = max(0, min(x, frames[0].shape[1] - 10))
        y = max(0, min(y, frames[0].shape[0] - 10))
        w = max(10, min(w, frames[0].shape[1] - x))
        h = max(10, min(h, frames[0].shape[0] - y))
    
    prev_frame = frames[0]
    prev_box = (x, y, w, h)
    
    try:
        prev_roi = prev_frame[y:y+h, x:x+w]
    except IndexError:
        # If box is invalid, use a safe default
        x, y, w, h = 0, 0, min(100, frames[0].shape[1]//4), min(100, frames[0].shape[0]//4)
        prev_box = (x, y, w, h)
        prev_roi = prev_frame[y:y+h, x:x+w]
    
    for i in range(1, len(frames)):
        curr_frame = frames[i]
        
        # Define a search region
        search_x = max(0, prev_box[0] - 20)
        search_y = max(0, prev_box[1] - 20)
        search_w = min(prev_box[2] + 40, curr_frame.shape[1] - search_x)
        search_h = min(prev_box[3] + 40, curr_frame.shape[0] - search_y)
        
        if search_w <= 0 or search_h <= 0:
            # Invalid search region, just keep the previous box
            results.append(prev_box)
            continue
        
        # Find the object in the new frame using template matching
        try:
            search_region = curr_frame[search_y:search_y+search_h, search_x:search_x+search_w]
            
            # Resize the template if necessary
            if prev_roi.shape[0] > search_region.shape[0] or prev_roi.shape[1] > search_region.shape[1]:
                h_scale = min(1.0, float(search_region.shape[0]) / prev_roi.shape[0])
                w_scale = min(1.0, float(search_region.shape[1]) / prev_roi.shape[1])
                scale = min(h_scale, w_scale)
                if scale < 1.0:
                    new_h = max(1, int(prev_roi.shape[0] * scale))
                    new_w = max(1, int(prev_roi.shape[1] * scale))
                    prev_roi = cv2.resize(prev_roi, (new_w, new_h))
            
            if prev_roi.shape[0] > search_region.shape[0] or prev_roi.shape[1] > search_region.shape[1]:
                results.append(prev_box)
                continue
                
            # Use template matching
            try:
                result = cv2.matchTemplate(search_region, prev_roi, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(result)
                
                # Update box location
                new_x = search_x + max_loc[0]
                new_y = search_y + max_loc[1]
                new_box = (new_x, new_y, prev_roi.shape[1], prev_roi.shape[0])
                
                # Update for next frame
                prev_box = new_box
                y_end = min(new_y + prev_roi.shape[0], curr_frame.shape[0])
                x_end = min(new_x + prev_roi.shape[1], curr_frame.shape[1])
                if y_end > new_y and x_end > new_x:
                    prev_roi = curr_frame[new_y:y_end, new_x:x_end]
                
                results.append(new_box)
            except cv2.error:
                # OpenCV error in template matching
                results.append(prev_box)
                
        except Exception as e:
            # Any other error
            print(f"Tracking error: {e}")
            results.append(prev_box)
    
    return results

def evaluate_tracking(original_frames, compressed_frames, gt_data):
    """Evaluate tracking performance on original and compressed frames."""
    acc_original = MOTAccumulator(auto_id=True)
    acc_compressed = MOTAccumulator(auto_id=True)
    
    # Make sure we have ground truth for some frames
    if not gt_data:
        print("No ground truth data available")
        return None, None
    
    # Get the first frame with ground truth
    first_frame = min(gt_data.keys())
    
    # Get initial objects from ground truth
    gt_objects = gt_data[first_frame]
    gt_ids = [obj[0] for obj in gt_objects]
    gt_boxes = [obj[1] for obj in gt_objects]
    
    if not gt_boxes:
        print("No ground truth boxes available")
        return None, None
    
    print(f"Starting tracking with {len(gt_boxes)} objects...")
    
    # Track each object (limit to first 5 objects for speed)
    tracked_original = []
    tracked_compressed = []
    
    for gt_box in gt_boxes[:5]:  
        print(f"Tracking object at {gt_box}...")
        tracked_original.append(track_object_simple(original_frames, gt_box))
        tracked_compressed.append(track_object_simple(compressed_frames, gt_box))
    
    # Evaluate frame by frame
    for frame_idx, frame_num in enumerate(sorted(gt_data.keys())):
        if frame_idx >= len(original_frames):
            break
            
        # Get ground truth
        gt_objects = gt_data[frame_num]
        frame_gt_ids = [obj[0] for obj in gt_objects]
        frame_gt_boxes = [obj[1] for obj in gt_objects]
        
        # Skip if we don't have tracking data
        if not tracked_original or not tracked_compressed:
            continue
            
        frame_tracked_original = []
        frame_tracked_compressed = []
        
        # Get tracked positions for this frame
        for i, (track_orig, track_comp) in enumerate(zip(tracked_original, tracked_compressed)):
            if frame_idx < len(track_orig):
                frame_tracked_original.append(track_orig[frame_idx])
            if frame_idx < len(track_comp):
                frame_tracked_compressed.append(track_comp[frame_idx])
        
        # Skip if we don't have tracking results
        if not frame_tracked_original or not frame_tracked_compressed:
            continue
        
        # Build distance matrices
        distance_matrix_orig = []
        distance_matrix_comp = []
        
        for gt_box in frame_gt_boxes:
            row_orig = []
            row_comp = []
            for track_box in frame_tracked_original:
                iou = calculate_iou(gt_box, track_box)
                row_orig.append(1 - iou)  # distance = 1 - IoU
            for track_box in frame_tracked_compressed:
                iou = calculate_iou(gt_box, track_box)
                row_comp.append(1 - iou)  # distance = 1 - IoU
            
            if row_orig:
                distance_matrix_orig.append(row_orig)
            if row_comp:
                distance_matrix_comp.append(row_comp)
        
        # Update accumulators
        if distance_matrix_orig and frame_gt_ids:
            acc_original.update(
                frame_gt_ids,
                list(range(len(frame_tracked_original))),
                distance_matrix_orig
            )
        
        if distance_matrix_comp and frame_gt_ids:
            acc_compressed.update(
                frame_gt_ids,
                list(range(len(frame_tracked_compressed))),
                distance_matrix_comp
            )
    
    return acc_original, acc_compressed

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    # Calculate areas
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def load_compressed_frames(video_path, max_frames=None):
    """Load compressed frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        if max_frames is not None and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames

def create_comparison_plots(results, output_dir):
    """Create comparison plots for different compression methods."""
    # Extract data for plotting
    methods = list(results.keys())
    compression_ratio_values = []
    bpp_values = []
    psnr_values = []
    mota_values = []
    motp_values = []
    delta_mota_values = []
    
    for method in methods:
        compression_ratio_values.append(results[method]["compression_ratio"])
        bpp_values.append(results[method]["bpp"])
        psnr_values.append(results[method]["psnr"])
        mota_values.append(results[method]["mota"])
        motp_values.append(results[method]["motp"])
        delta_mota_values.append(results[method]["delta_mota"])
    
    # Group methods by codec
    codec_groups = {}
    for method in methods:
        if method == "our_method":
            codec = "Our Method"
        else:
            codec = method.split("_")[0]
            
        if codec not in codec_groups:
            codec_groups[codec] = []
        codec_groups[codec].append(method)
    
    # Set up colors and markers
    colors = {
        "Our Method": "red",
        "h264": "blue",
        "h265": "green",
        "vp9": "purple"
    }
    
    markers = {
        "Our Method": "o",
        "h264": "s",
        "h265": "^",
        "vp9": "D"
    }
    
    # Create BPP vs MOTA plot
    plt.figure(figsize=(10, 6))
    
    for codec, methods_in_group in codec_groups.items():
        x = [results[m]["bpp"] for m in methods_in_group]
        y = [results[m]["mota"] for m in methods_in_group]
        
        if codec == "Our Method":
            plt.scatter(x, y, color=colors[codec], marker=markers[codec], s=100, label=codec, zorder=10)
        else:
            plt.plot(x, y, color=colors[codec], marker=markers[codec], label=codec)
    
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel('MOTA')
    plt.title('Rate-Accuracy Curve (BPP vs MOTA)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "bpp_vs_mota.png"), dpi=300, bbox_inches='tight')
    
    # Create PSNR vs MOTA plot
    plt.figure(figsize=(10, 6))
    
    for codec, methods_in_group in codec_groups.items():
        x = [results[m]["psnr"] for m in methods_in_group]
        y = [results[m]["mota"] for m in methods_in_group]
        
        if codec == "Our Method":
            plt.scatter(x, y, color=colors[codec], marker=markers[codec], s=100, label=codec, zorder=10)
        else:
            plt.plot(x, y, color=colors[codec], marker=markers[codec], label=codec)
    
    plt.xlabel('PSNR (dB)')
    plt.ylabel('MOTA')
    plt.title('Quality-Accuracy Curve (PSNR vs MOTA)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "psnr_vs_mota.png"), dpi=300, bbox_inches='tight')
    
    # Create Compression Ratio vs Delta MOTA plot
    plt.figure(figsize=(10, 6))
    
    for codec, methods_in_group in codec_groups.items():
        x = [results[m]["compression_ratio"] for m in methods_in_group]
        y = [results[m]["delta_mota"] for m in methods_in_group]
        
        if codec == "Our Method":
            plt.scatter(x, y, color=colors[codec], marker=markers[codec], s=100, label=codec, zorder=10)
        else:
            plt.plot(x, y, color=colors[codec], marker=markers[codec], label=codec)
    
    plt.xlabel('Compression Ratio')
    plt.ylabel('Delta MOTA')
    plt.title('Compression Ratio vs Change in MOTA')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "ratio_vs_delta_mota.png"), dpi=300, bbox_inches='tight')
    
    print(f"Saved comparison plots to {output_dir}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load sequence frames and ground truth
    sequence_dir = Path(args.sequence_path)
    gt_path = sequence_dir / "gt" / "gt.txt"
    img_dir = sequence_dir / "img1"
    
    if not img_dir.exists():
        raise ValueError(f"Image directory not found at {img_dir}")
    
    # Load frames
    frame_files = sorted(img_dir.glob("*.jpg"))
    if not frame_files:
        raise ValueError(f"No image files found in {img_dir}")
    
    # Limit frames if specified
    if args.max_frames is not None:
        frame_files = frame_files[:args.max_frames]
    
    original_frames = []
    for frame_file in tqdm(frame_files, desc="Loading frames"):
        frame = cv2.imread(str(frame_file))
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
        original_frames.append(frame)
    
    print(f"Loaded {len(original_frames)} frames")
    
    # Load ground truth
    gt_data = load_mot_groundtruth(gt_path, args.max_frames)
    
    # Load compression results
    compression_results_path = os.path.join(args.compression_results, "compression_results.json")
    if not os.path.exists(compression_results_path):
        raise ValueError(f"Compression results not found at {compression_results_path}")
    
    with open(compression_results_path, 'r') as f:
        compression_results = json.load(f)
    
    # Evaluate tracking on original video
    print("Evaluating tracking on original video...")
    metrics_host = mm.metrics.create()
    metrics_names = ['mota', 'motp', 'precision', 'recall']
    
    # Initialize results dictionary with codec info
    tracking_results = {}
    
    # Load compressed videos for each method
    for method in tqdm(compression_results.keys(), desc="Evaluating methods"):
        print(f"\nEvaluating method: {method}")
        
        # Load compressed frames
        if method == "our_method":
            # For our method, we need to load the frames from sample directory
            compressed_frames = []
            for i in range(len(original_frames)):
                frame_path = os.path.join(args.compression_results, "samples", method, f"compressed_{i % 5}.png")
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    compressed_frames.append(frame)
                else:
                    # If file doesn't exist, use last frame or create blank frame
                    if compressed_frames:
                        compressed_frames.append(compressed_frames[-1])
                    else:
                        compressed_frames.append(np.zeros_like(original_frames[0]))
        else:
            # For standard codecs, load from the video file
            video_path = os.path.join(args.compression_results, f"temp_{method}.mp4")
            if os.path.exists(video_path):
                compressed_frames = load_compressed_frames(video_path, args.max_frames)
            else:
                print(f"Warning: Video file not found for method {method}. Skipping.")
                continue
        
        # Make sure we have enough frames
        if len(compressed_frames) < len(original_frames):
            print(f"Warning: Not enough compressed frames for method {method}. Padding with last frame.")
            last_frame = compressed_frames[-1] if compressed_frames else np.zeros_like(original_frames[0])
            compressed_frames.extend([last_frame] * (len(original_frames) - len(compressed_frames)))
        
        # Evaluate tracking
        print(f"Evaluating tracking for method {method}...")
        acc_original, acc_compressed = evaluate_tracking(original_frames, compressed_frames, gt_data)
        
        if acc_original is None or acc_compressed is None:
            print(f"Error: Tracking evaluation failed for method {method}")
            continue
        
        # Calculate metrics
        try:
            summary_original = metrics_host.compute(acc_original, metrics=metrics_names, name='Original')
            summary_compressed = metrics_host.compute(acc_compressed, metrics=metrics_names, name='Compressed')
            
            # Extract metrics
            mota_original = summary_original['mota']['Original'] if 'mota' in summary_original else 0
            motp_original = summary_original['motp']['Original'] if 'motp' in summary_original else 0
            precision_original = summary_original['precision']['Original'] if 'precision' in summary_original else 0
            recall_original = summary_original['recall']['Original'] if 'recall' in summary_original else 0
            
            mota_compressed = summary_compressed['mota']['Compressed'] if 'mota' in summary_compressed else 0
            motp_compressed = summary_compressed['motp']['Compressed'] if 'motp' in summary_compressed else 0
            precision_compressed = summary_compressed['precision']['Compressed'] if 'precision' in summary_compressed else 0
            recall_compressed = summary_compressed['recall']['Compressed'] if 'recall' in summary_compressed else 0
            
            # Calculate delta metrics
            delta_mota = mota_compressed - mota_original
            delta_motp = motp_compressed - motp_original
            delta_precision = precision_compressed - precision_original
            delta_recall = recall_compressed - recall_original
            
            # Store tracking results
            tracking_results[method] = {
                "compression_ratio": compression_results[method]["compression_ratio"],
                "bpp": compression_results[method]["bpp"],
                "psnr": compression_results[method]["psnr"],
                "ms_ssim": compression_results[method]["ms_ssim"],
                "mota_original": mota_original,
                "motp_original": motp_original,
                "precision_original": precision_original,
                "recall_original": recall_original,
                "mota": mota_compressed,
                "motp": motp_compressed,
                "precision": precision_compressed,
                "recall": recall_compressed,
                "delta_mota": delta_mota,
                "delta_motp": delta_motp,
                "delta_precision": delta_precision,
                "delta_recall": delta_recall
            }
            
        except Exception as e:
            print(f"Error calculating metrics for method {method}: {e}")
    
    # Save tracking results to JSON
    tracking_results_path = os.path.join(args.output_dir, "tracking_results.json")
    with open(tracking_results_path, 'w') as f:
        json.dump(tracking_results, f, indent=4)
    
    # Create and save comparison table
    table_path = os.path.join(args.output_dir, "tracking_comparison_table.txt")
    with open(table_path, 'w') as f:
        f.write("Tracking Performance on Compressed Videos\n")
        f.write("=======================================\n\n")
        f.write(f"{'Method':<15} | {'Comp.Ratio':>12} | {'BPP':>8} | {'PSNR':>8} | {'MOTA':>8} | {'MOTP':>8} | {'Δ MOTA':>8} | {'Δ MOTP':>8}\n")
        f.write("-" * 100 + "\n")
        
        # Sort methods by compression ratio
        sorted_methods = sorted(tracking_results.keys(), key=lambda x: tracking_results[x]["compression_ratio"], reverse=True)
        
        for method in sorted_methods:
            r = tracking_results[method]
            f.write(f"{method:<15} | {r['compression_ratio']:>12.2f} | {r['bpp']:>8.4f} | {r['psnr']:>8.2f} | ")
            f.write(f"{r['mota']:>8.4f} | {r['motp']:>8.4f} | {r['delta_mota']:>+8.4f} | {r['delta_motp']:>+8.4f}\n")
    
    # Create comparison plots
    create_comparison_plots(tracking_results, args.output_dir)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 