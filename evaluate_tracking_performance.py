#!/usr/bin/env python
"""
Script to evaluate tracking performance on image sequences compressed with the improved autoencoder.
Supports both MOT sequences and generic image directory formats.
"""

import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import motmetrics as mm
from motmetrics.metrics import MOTAccumulator
import glob
import json

from improved_autoencoder import ImprovedAutoencoder

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tracking performance on compressed image sequences")
    
    # Input parameters
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to image sequence directory. Can be either a MOT sequence dir or regular image dir")
    parser.add_argument("--input_format", type=str, default="auto", choices=["auto", "mot", "images"],
                        help="Format of input: 'mot' for MOT sequence, 'images' for image directory, 'auto' to detect")
    parser.add_argument("--image_pattern", type=str, default="*.jpg",
                        help="Pattern for image files when using regular image directory (e.g., '*.jpg', '*.png')")
    parser.add_argument("--model_path", type=str, 
                        default="trained_models/improved_autoencoder/autoencoder_best.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Starting frame index for processing (0-indexed)")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Process every Nth frame (1=all frames)")
    
    # Model parameters
    parser.add_argument("--latent_channels", type=int, default=8,
                        help="Number of channels in latent space")
    parser.add_argument("--time_reduction", type=int, default=2,
                        help="Temporal reduction factor")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames to process at once")
    
    # Tracking parameters
    parser.add_argument("--tracker_type", type=str, default="CSRT",
                        help="OpenCV tracker type (CSRT, KCF, etc.)")
    parser.add_argument("--gt_file", type=str, default=None,
                        help="Path to ground truth file for non-MOT sequences")
    
    # Evaluation parameters
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to process (None=all)")
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save intermediate results after processing each batch")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, 
                        default="results/tracking_evaluation",
                        help="Directory to save evaluation results")
    
    return parser.parse_args()

def load_mot_groundtruth(gt_path, max_frames=None, start_frame=0, frame_step=1):
    """Load MOT ground truth data."""
    gt_data = {}
    try:
        with open(gt_path, 'r') as f:
            for line in f:
                frame, track_id, x, y, w, h, conf, _, _, _ = map(float, line.strip().split(','))
                frame = int(frame)
                
                # Apply frame filtering
                if frame < start_frame + 1:  # MOT frames are 1-indexed
                    continue
                if (frame - 1) % frame_step != 0:  # Convert to 0-indexed for modulo
                    continue
                if max_frames is not None and frame > start_frame + (max_frames * frame_step):
                    continue
                    
                track_id = int(track_id)
                if frame not in gt_data:
                    gt_data[frame] = []
                gt_data[frame].append((track_id, (x, y, w, h)))
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {gt_path}")
        # Create dummy ground truth for testing
        frame_count = 600 if max_frames is None else max_frames
        for frame in range(start_frame + 1, start_frame + (frame_count * frame_step) + 1, frame_step):
            gt_data[frame] = [(1, (100, 100, 50, 100))]
    return gt_data

def load_custom_groundtruth(gt_path, max_frames=None, start_frame=0, frame_step=1):
    """Load custom ground truth data from file."""
    gt_data = {}
    if gt_path is None or not os.path.exists(gt_path):
        print("No valid ground truth file provided. Creating dummy ground truth.")
        frame_count = 600 if max_frames is None else max_frames
        for frame in range(start_frame + 1, start_frame + (frame_count * frame_step) + 1, frame_step):
            gt_data[frame] = [(1, (100, 100, 50, 100))]
        return gt_data
        
    try:
        # First check if it's a JSON format
        if gt_path.endswith('.json'):
            with open(gt_path, 'r') as f:
                json_data = json.load(f)
                
            # Determine format based on JSON structure
            if isinstance(json_data, dict) and 'frames' in json_data:
                # Assume format with 'frames' key containing frame data
                for frame_idx, frame_data in enumerate(json_data['frames']):
                    # Apply frame filtering
                    if frame_idx < start_frame:
                        continue
                    if (frame_idx - start_frame) % frame_step != 0:
                        continue
                    if max_frames is not None and frame_idx >= start_frame + (max_frames * frame_step):
                        continue
                    
                    frame_num = frame_idx + 1  # Convert to 1-indexed for consistency
                    gt_data[frame_num] = []
                    
                    if 'objects' in frame_data:
                        for obj_idx, obj in enumerate(frame_data['objects']):
                            if 'bbox' in obj:
                                bbox = obj['bbox']
                                # Assuming bbox format is [x, y, width, height]
                                gt_data[frame_num].append((obj.get('id', obj_idx + 1), tuple(bbox)))
            else:
                # Try other JSON formats
                for frame_str, objects in json_data.items():
                    if frame_str.isdigit():
                        frame = int(frame_str)
                        
                        # Apply frame filtering
                        if frame < start_frame + 1:
                            continue
                        if (frame - 1) % frame_step != 0:
                            continue
                        if max_frames is not None and frame > start_frame + (max_frames * frame_step):
                            continue
                            
                        gt_data[frame] = []
                        for obj_id, bbox in objects.items():
                            if isinstance(bbox, list) and len(bbox) >= 4:
                                gt_data[frame].append((int(obj_id), tuple(bbox[:4])))
        else:
            # Try to load generic CSV format: frame_id, track_id, x, y, w, h
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:  # Minimal format
                        frame = int(float(parts[0]))
                        
                        # Apply frame filtering
                        if frame < start_frame + 1:
                            continue
                        if (frame - 1) % frame_step != 0:
                            continue
                        if max_frames is not None and frame > start_frame + (max_frames * frame_step):
                            continue
                            
                        track_id = int(float(parts[1]))
                        x, y, w, h = map(float, parts[2:6])
                        if frame not in gt_data:
                            gt_data[frame] = []
                        gt_data[frame].append((track_id, (x, y, w, h)))
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        print("Creating dummy ground truth.")
        frame_count = 600 if max_frames is None else max_frames
        for frame in range(start_frame + 1, start_frame + (frame_count * frame_step) + 1, frame_step):
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
        
        # Calculate distances (1-IoU)
        distances_original = []
        distances_compressed = []
        
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

def calculate_psnr(original, compressed):
    """Calculate PSNR between original and compressed frames."""
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = ImprovedAutoencoder(
        input_channels=3,
        latent_channels=args.latent_channels,
        time_reduction=args.time_reduction
    ).to(device)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Warning: Model checkpoint not found at {args.model_path}")
        print("Proceeding with uninitialized model for testing purposes")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Proceeding with uninitialized model for testing purposes")
    
    # Determine input format and load sequence data
    input_path = Path(args.input_path)
    input_format = args.input_format
    
    # Auto-detect format if requested
    if input_format == "auto":
        # Check if it looks like a MOT sequence directory
        if (input_path / "img1").exists() and (input_path / "gt").exists():
            input_format = "mot"
        else:
            input_format = "images"
        print(f"Auto-detected input format: {input_format}")
    
    # Load frames based on format
    print("Loading sequence data...")
    
    if input_format == "mot":
        # MOT format: img1 directory with frames
        img_dir = input_path / "img1"
        gt_path = input_path / "gt" / "gt.txt"
        
        if not img_dir.exists():
            raise ValueError(f"Image directory not found at {img_dir}")
        
        # Load frames
        frame_files = sorted(img_dir.glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(img_dir.glob("*.png"))  # Try PNG if no JPG
        
        if not frame_files:
            raise ValueError(f"No image files found in {img_dir}")
    else:
        # Regular image directory
        frame_files = sorted(list(input_path.glob(args.image_pattern)))
        gt_path = args.gt_file
        
        if not frame_files:
            raise ValueError(f"No images matching pattern '{args.image_pattern}' found in {input_path}")
    
    # Apply frame filtering (start_frame, frame_step)
    frame_files = frame_files[args.start_frame::args.frame_step]
    
    # Limit frames if specified
    if args.max_frames is not None:
        frame_files = frame_files[:args.max_frames]
    
    print(f"Selected {len(frame_files)} frames for processing")
    
    # Load ground truth based on format
    if input_format == "mot":
        gt_data = load_mot_groundtruth(gt_path, args.max_frames, args.start_frame, args.frame_step)
    else:
        gt_data = load_custom_groundtruth(gt_path, args.max_frames, args.start_frame, args.frame_step)
    
    # Process frames in batches
    print("Processing frames in batches...")
    batch_size = min(args.time_steps, 100)  # Limit batch size for memory
    
    all_original_frames = []
    all_compressed_frames = []
    all_compression_sizes = []
    all_original_sizes = []
    
    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(frame_files), batch_size), desc="Batch processing"):
        batch_files = frame_files[batch_start:batch_start+batch_size]
        
        # Load this batch of frames
        batch_frames = []
        for frame_file in tqdm(batch_files, desc="Loading frames", leave=False):
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Warning: Could not read frame {frame_file}")
                continue
            batch_frames.append(frame)
        
        if not batch_frames:
            print(f"Warning: No frames loaded in batch starting at {batch_start}")
            continue
        
        # Save original size in bytes
        batch_original_sizes = []
        for frame in batch_frames:
            _, buffer = cv2.imencode('.jpg', frame)
            batch_original_sizes.append(len(buffer))
        
        # Process frames in sub-batches of time_steps for the model
        batch_compressed_frames = []
        batch_compression_sizes = []
        
        for i in range(0, len(batch_frames), args.time_steps):
            # Get sequence
            sequence = batch_frames[i:i+args.time_steps]
            if len(sequence) < args.time_steps:
                # Pad the sequence if necessary
                sequence = sequence + [sequence[-1]] * (args.time_steps - len(sequence))
            
            # Create tensor of shape [B, C, T, H, W]
            sequence_tensor = torch.zeros((1, 3, len(sequence), sequence[0].shape[0], sequence[0].shape[1]), 
                                         device=device)
            
            for t, frame in enumerate(sequence):
                # Convert BGR to RGB and normalize to [0, 1]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                sequence_tensor[0, :, t, :, :] = frame_tensor.to(device)
            
            # Compress and reconstruct
            with torch.no_grad():
                reconstructed, latent, _ = model(sequence_tensor)
                
                # Estimate compressed size in bytes
                latent_cpu = latent.cpu().numpy()
                batch_compression_sizes.append(latent_cpu.nbytes)
            
            # Convert back to numpy
            reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            
            # Convert from RGB back to BGR for OpenCV
            reconstructed_frames = []
            for t in range(reconstructed_np.shape[0]):
                # Convert normalized RGB [0,1] to BGR [0,255]
                frame_rgb = (reconstructed_np[t] * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                reconstructed_frames.append(frame_bgr)
            
            # Store actual frames (not padding)
            actual_frame_count = min(len(sequence), len(batch_frames) - i)
            batch_compressed_frames.extend(reconstructed_frames[:actual_frame_count])
        
        # Add to the overall lists
        all_original_frames.extend(batch_frames)
        all_compressed_frames.extend(batch_compressed_frames)
        all_compression_sizes.extend(batch_compression_sizes)
        all_original_sizes.extend(batch_original_sizes)
        
        # Save intermediate results if requested
        if args.save_intermediate and batch_start > 0:
            # Save sample images
            sample_dir = os.path.join(args.output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save the last few frames from this batch
            for i in range(max(0, len(batch_frames)-5), len(batch_frames)):
                rel_idx = i - max(0, len(batch_frames)-5)
                abs_idx = batch_start + i
                
                orig_path = os.path.join(sample_dir, f"original_batch{batch_start}_{rel_idx}.jpg")
                comp_path = os.path.join(sample_dir, f"compressed_batch{batch_start}_{rel_idx}.jpg")
                
                cv2.imwrite(orig_path, batch_frames[i])
                cv2.imwrite(comp_path, batch_compressed_frames[i])
            
            print(f"Saved intermediate results for batch starting at {batch_start}")
    
    # Ensure we have frames to process
    if not all_original_frames:
        raise ValueError("No frames could be loaded from the sequence")
    
    print(f"Processed {len(all_original_frames)} frames")
    
    # Calculate compression metrics
    total_original_size = sum(all_original_sizes)
    total_compressed_size = sum(all_compression_sizes)
    compression_ratio = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
    bpp = (total_compressed_size * 8) / (len(all_original_frames) * all_original_frames[0].shape[0] * all_original_frames[0].shape[1])
    
    # Calculate PSNR
    psnr_values = []
    for orig, comp in zip(all_original_frames, all_compressed_frames):
        psnr = calculate_psnr(orig, comp)
        if not np.isinf(psnr):
            psnr_values.append(psnr)
    
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    
    # Save sample images for visual inspection
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    for i in range(min(5, len(all_original_frames))):
        orig_path = os.path.join(sample_dir, f"original_{i}.jpg")
        comp_path = os.path.join(sample_dir, f"compressed_{i}.jpg")
        
        cv2.imwrite(orig_path, all_original_frames[i])
        cv2.imwrite(comp_path, all_compressed_frames[i])
        
        # Create side-by-side comparison
        h, w = all_original_frames[i].shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = all_original_frames[i]
        comparison[:, w:] = all_compressed_frames[i]
        comp_path = os.path.join(sample_dir, f"comparison_{i}.jpg")
        cv2.imwrite(comp_path, comparison)
    
    # Create comparison video
    video_path = os.path.join(args.output_dir, "comparison.mp4")
    fps = 25  # Typical MOT16 frame rate
    h, w = all_original_frames[0].shape[:2]
    video_writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w*2, h)
    )
    
    for orig, comp in zip(all_original_frames, all_compressed_frames):
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        comparison[:, :w] = orig
        comparison[:, w:] = comp
        video_writer.write(comparison)
    
    video_writer.release()
    
    # Evaluate tracking
    print("Evaluating tracking performance...")
    acc_original, acc_compressed = evaluate_tracking(
        all_original_frames, all_compressed_frames, gt_data
    )
    
    # Save compression results
    results_path = os.path.join(args.output_dir, "compression_results.txt")
    with open(results_path, 'w') as f:
        f.write("Compression Results\n")
        f.write("==================\n\n")
        f.write(f"Sequence: {input_path.name}\n")
        f.write(f"Frames: {len(all_original_frames)}\n")
        f.write(f"Resolution: {all_original_frames[0].shape[1]}x{all_original_frames[0].shape[0]}\n\n")
        f.write(f"Original Size: {total_original_size / 1024 / 1024:.2f} MB\n")
        f.write(f"Compressed Size: {total_compressed_size / 1024 / 1024:.2f} MB\n")
        f.write(f"Compression Ratio: {compression_ratio:.2f}x\n")
        f.write(f"Bits per Pixel: {bpp:.4f}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
    
    # Continue with tracking evaluation if successful
    if acc_original is None or acc_compressed is None:
        print("Error: Tracking evaluation failed")
        print("Compression results were saved.")
        return
    
    # Calculate tracking metrics
    try:
        metrics_names = ['mota', 'motp', 'precision', 'recall']
        metrics_host = mm.metrics.create()
        summary_original = metrics_host.compute(acc_original, metrics=metrics_names, name='Original')
        summary_compressed = metrics_host.compute(acc_compressed, metrics=metrics_names, name='Compressed')
        
        # Save results
        results_path = os.path.join(args.output_dir, "tracking_results.txt")
        with open(results_path, 'w') as f:
            f.write("Tracking and Compression Evaluation\n")
            f.write("================================\n\n")
            f.write(f"Sequence: {input_path.name}\n")
            f.write(f"Frames: {len(all_original_frames)}\n")
            f.write(f"Resolution: {all_original_frames[0].shape[1]}x{all_original_frames[0].shape[0]}\n\n")
            
            f.write("Compression Metrics:\n")
            f.write(f"Original Size: {total_original_size / 1024 / 1024:.2f} MB\n")
            f.write(f"Compressed Size: {total_compressed_size / 1024 / 1024:.2f} MB\n")
            f.write(f"Compression Ratio: {compression_ratio:.2f}x\n")
            f.write(f"Bits per Pixel: {bpp:.4f}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f} dB\n\n")
            
            f.write("Tracking Metrics:\n")
            f.write("Original Video:\n")
            for metric in metrics_names:
                if metric in summary_original:
                    f.write(f"{metric.upper()}: {summary_original[metric]['Original']:.4f}\n")
                else:
                    f.write(f"{metric.upper()}: N/A\n")
            f.write("\n")
            
            f.write("Compressed Video:\n")
            for metric in metrics_names:
                if metric in summary_compressed:
                    f.write(f"{metric.upper()}: {summary_compressed[metric]['Compressed']:.4f}\n")
                else:
                    f.write(f"{metric.upper()}: N/A\n")
            
            # Calculate impact on tracking
            f.write("\nImpact on Tracking Performance:\n")
            for metric in metrics_names:
                if metric in summary_original and metric in summary_compressed:
                    orig_val = summary_original[metric]['Original']
                    comp_val = summary_compressed[metric]['Compressed']
                    diff = comp_val - orig_val
                    f.write(f"{metric.upper()} Change: {diff:+.4f}\n")
        
        print(f"Saved results to {results_path}")
        
    except Exception as e:
        print(f"Error calculating tracking metrics: {e}")
        print("Compression results were saved successfully.")

if __name__ == "__main__":
    main() 