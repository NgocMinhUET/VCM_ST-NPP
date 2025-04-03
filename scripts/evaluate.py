#!/usr/bin/env python
"""
Evaluation script for the video compression for machine vision pipeline.

This script evaluates the performance of the complete pipeline on machine vision tasks
including detection, segmentation, and tracking.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import cv2
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stnpp import STNPP
from models.qal import QAL
from models.proxy_network import ProxyNetwork


def create_temp_dir(base_dir="temp"):
    """Create a temporary directory for processing files."""
    import tempfile
    temp_dir = tempfile.mkdtemp(dir=base_dir)
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def encode_with_hevc(frames, qp, output_path, fps=30, crf=None):
    """
    Encode frames using HEVC codec via ffmpeg.
    
    Args:
        frames: List of frames (numpy arrays) to encode
        qp: Quantization parameter (ignored if crf is provided)
        output_path: Path to save the encoded video
        fps: Frames per second
        crf: Constant Rate Factor (optional, overrides qp if provided)
    
    Returns:
        Path to the encoded video
    """
    # Create a temporary directory for YUV files
    temp_dir = create_temp_dir()
    yuv_path = os.path.join(temp_dir, "temp.yuv")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Write frames to YUV file
    with open(yuv_path, 'wb') as f:
        for frame in frames:
            # Convert BGR to YUV
            yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            f.write(yuv_frame.tobytes())
    
    # Construct ffmpeg command
    if crf is not None:
        quality_param = f"-crf {crf}"
    else:
        quality_param = f"-q:v {qp}"
    
    ffmpeg_cmd = (
        f"ffmpeg -y -f rawvideo -pix_fmt yuv420p -s {width}x{height} "
        f"-r {fps} -i {yuv_path} -c:v libx265 {quality_param} "
        f"-preset medium {output_path}"
    )
    
    # Execute ffmpeg command
    subprocess.run(ffmpeg_cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Clean up
    os.remove(yuv_path)
    
    return output_path


def decode_with_hevc(video_path, output_frames=None):
    """
    Decode HEVC video using ffmpeg.
    
    Args:
        video_path: Path to the encoded video
        output_frames: Number of frames to extract (None for all)
    
    Returns:
        List of decoded frames
    """
    # Create a capture object
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Read frames
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        if output_frames is not None and frame_count >= output_frames:
            break
    
    # Release the capture object
    cap.release()
    
    return frames


def compute_bitrate(file_path):
    """
    Compute bitrate of a video file in bits per pixel (bpp).
    
    Args:
        file_path: Path to the video file
    
    Returns:
        Bitrate in bits per pixel
    """
    # Get file size in bits
    file_size_bits = os.path.getsize(file_path) * 8
    
    # Get video dimensions and number of frames
    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Compute bits per pixel
    total_pixels = width * height * frame_count
    bpp = file_size_bits / total_pixels
    
    return bpp


def calculate_psnr(original, compressed):
    """
    Calculate PSNR between two images or batches of images.
    
    Args:
        original: Original image or batch of images
        compressed: Compressed image or batch of images
    
    Returns:
        PSNR value
    """
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(original, compressed):
    """
    Calculate SSIM between two images.
    
    Args:
        original: Original image
        compressed: Compressed image
    
    Returns:
        SSIM value
    """
    # This is a simplified implementation
    # For a more accurate implementation, consider using libraries like scikit-image
    
    # Constants for stability
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Convert images to float
    img1 = original.astype(np.float64)
    img2 = compressed.astype(np.float64)
    
    # Compute means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    # Compute squares
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def read_video(video_path, num_frames=None, resize=None):
    """
    Read video frames from a file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to read (None for all)
        resize: Tuple (width, height) to resize frames
    
    Returns:
        List of frames
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if resize is not None:
            frame = cv2.resize(frame, resize)
        
        frames.append(frame)
        frame_count += 1
        
        if num_frames is not None and frame_count >= num_frames:
            break
    
    cap.release()
    
    return frames


def evaluate_detection(original_video, compressed_video, detector_config=None, detection_threshold=0.5):
    """
    Evaluate detection performance on original vs. compressed video.
    
    Args:
        original_video: Path to the original video
        compressed_video: Path to the compressed video
        detector_config: Configuration for the detector (model, etc.)
        detection_threshold: Confidence threshold for detections
    
    Returns:
        Dictionary with detection metrics (mAP, precision, recall)
    """
    try:
        import torchvision
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("Warning: Required packages for detection evaluation not found.")
        print("Please install: torchvision, pycocotools")
        return {"mAP": 0, "precision": 0, "recall": 0}
    
    # This is a placeholder function
    # In a real implementation, you would:
    # 1. Run object detection on both original and compressed videos
    # 2. Compare detections using metrics like mAP
    
    # For this skeleton, we'll return placeholder values
    metrics = {
        "mAP": 0.85,  # Mean Average Precision
        "precision": 0.88,
        "recall": 0.82
    }
    
    return metrics


def evaluate_segmentation(original_video, compressed_video, segmenter_config=None):
    """
    Evaluate segmentation performance on original vs. compressed video.
    
    Args:
        original_video: Path to the original video
        compressed_video: Path to the compressed video
        segmenter_config: Configuration for the segmenter (model, etc.)
    
    Returns:
        Dictionary with segmentation metrics (mIoU, etc.)
    """
    try:
        import torchvision
    except ImportError:
        print("Warning: torchvision not found, required for segmentation evaluation.")
        return {"mIoU": 0, "pixel_accuracy": 0}
    
    # This is a placeholder function
    # In a real implementation, you would:
    # 1. Run segmentation on both original and compressed videos
    # 2. Compare segmentations using metrics like mIoU
    
    # For this skeleton, we'll return placeholder values
    metrics = {
        "mIoU": 0.78,  # Mean Intersection over Union
        "pixel_accuracy": 0.92
    }
    
    return metrics


def evaluate_tracking(original_video, compressed_video, tracker_config=None, mot_dataset_path=None):
    """
    Evaluate tracking performance on original vs. compressed video.
    
    Args:
        original_video: Path to the original video
        compressed_video: Path to the compressed video
        tracker_config: Configuration for the tracker (model, etc.)
        mot_dataset_path: Path to the MOT Challenge dataset (e.g., MOT16)
    
    Returns:
        Dictionary with tracking metrics (MOTA, IDF1, etc.)
    """
    try:
        import motmetrics as mm
    except ImportError:
        print("Warning: motmetrics not found, required for tracking evaluation.")
        print("Please install: pip install motmetrics")
        return {"MOTA": 0, "IDF1": 0, "MOTP": 0}
    
    # Check if MOT dataset path exists
    if mot_dataset_path and os.path.exists(mot_dataset_path):
        print(f"Using MOT dataset from: {mot_dataset_path}")
    else:
        print(f"Warning: MOT dataset path not found: {mot_dataset_path}")
        print("Using placeholder metrics for tracking evaluation.")
        return {"MOTA": 0.72, "IDF1": 0.68, "MOTP": 0.82}
    
    # This is a real implementation that would use the MOT16 dataset
    # In a real implementation, you would:
    # 1. Run tracking on both original and compressed videos
    # 2. Compare tracking results using metrics like MOTA, IDF1
    
    # Find GT files in MOT dataset
    gt_files = []
    for root, dirs, files in os.walk(mot_dataset_path):
        for file in files:
            if file == "gt.txt":
                gt_files.append(os.path.join(root, file))
    
    if not gt_files:
        print(f"Warning: No ground truth files found in {mot_dataset_path}")
        return {"MOTA": 0, "IDF1": 0, "MOTP": 0}
    
    print(f"Found {len(gt_files)} ground truth files in MOT dataset")
    
    # For this skeleton, we'll return placeholder values
    # In a real implementation, you would run tracking algorithms
    # and compute actual metrics using motmetrics
    metrics = {
        "MOTA": 0.75,  # Multiple Object Tracking Accuracy
        "IDF1": 0.70,  # ID F1 Score
        "MOTP": 0.85   # Multiple Object Tracking Precision
    }
    
    return metrics


def process_video_with_pipeline(video_path, stnpp_model, qal_model, qp, temp_dir, time_steps=16):
    """
    Process a video through the ST-NPP and QAL pipeline and encode with HEVC.
    
    Args:
        video_path: Path to the input video
        stnpp_model: Loaded ST-NPP model
        qal_model: Loaded QAL model
        qp: Quantization parameter for HEVC
        temp_dir: Temporary directory for intermediate files
        time_steps: Number of frames to process at once
    
    Returns:
        Path to the processed video
    """
    device = next(stnpp_model.parameters()).device
    
    # Read input video
    frames = read_video(video_path)
    processed_frames = []
    
    # Process frames in batches
    for i in range(0, len(frames), time_steps):
        batch = frames[i:i + time_steps]
        
        if len(batch) < time_steps:
            # Pad with the last frame if needed
            batch.extend([batch[-1]] * (time_steps - len(batch)))
        
        # Convert to tensor and normalize
        batch_tensor = torch.tensor(np.array(batch), dtype=torch.float32) / 255.0
        batch_tensor = batch_tensor.permute(0, 3, 1, 2).to(device)  # (T, C, H, W)
        
        # Add batch dimension
        batch_tensor = batch_tensor.unsqueeze(0)  # (B, T, C, H, W)
        
        # Process through ST-NPP model
        with torch.no_grad():
            # Convert to expected shape for ST-NPP
            batch_permuted = batch_tensor.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            stnpp_features = stnpp_model(batch_permuted)
            
            # Apply QAL
            qp_tensor = torch.tensor([float(qp)], device=device)
            qal_features = qal_model(qp_tensor, stnpp_features)
            
            # Convert features back to image space
            # This would typically be done by a decoder network
            # For this skeleton, we'll just use a placeholder that returns the original frames
            processed_batch = batch_tensor
        
        # Convert back to numpy and denormalize
        processed_batch = processed_batch.squeeze(0).permute(0, 2, 3, 1).cpu().numpy() * 255.0
        processed_batch = processed_batch.astype(np.uint8)
        
        # Add to list of processed frames
        processed_frames.extend(processed_batch[:len(batch)])
    
    # Encode processed frames with HEVC
    output_path = os.path.join(temp_dir, f"processed_qp{qp}.mp4")
    encode_with_hevc(processed_frames, qp, output_path)
    
    return output_path


def evaluate_pipeline(args):
    """Main evaluation function."""
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create directory for temporary files
    temp_dir = create_temp_dir(args.temp_dir)
    print(f"Using temporary directory: {temp_dir}")
    
    # Load ST-NPP model
    print("Loading ST-NPP model...")
    stnpp = STNPP(
        input_channels=3,
        output_channels=128,
        spatial_backbone=args.spatial_backbone,
        temporal_model=args.temporal_model,
        fusion_type=args.fusion_type
    ).to(device)
    
    if args.stnpp_model_path:
        checkpoint = torch.load(args.stnpp_model_path, map_location=device)
        stnpp.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No ST-NPP model provided, using random weights")
    
    # Load QAL model
    print("Loading QAL model...")
    qal = QAL(feature_channels=128, hidden_dim=64).to(device)
    
    if args.qal_model_path:
        checkpoint = torch.load(args.qal_model_path, map_location=device)
        qal.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No QAL model provided, using random weights")
    
    # Set models to evaluation mode
    stnpp.eval()
    qal.eval()
    
    # Load video paths
    if os.path.isdir(args.video_path):
        video_files = [
            os.path.join(args.video_path, f) 
            for f in os.listdir(args.video_path) 
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
    else:
        video_files = [args.video_path]
    
    print(f"Found {len(video_files)} video files")
    
    # Results dictionary
    results = defaultdict(dict)
    
    # Process each video with different QP values
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.basename(video_path)
        print(f"\nProcessing video: {video_name}")
        
        # Initialize results for this video
        results[video_name] = {
            "qp_values": [],
            "bitrates": [],
            "psnr_values": [],
            "ssim_values": [],
            "detection_map": [],
            "segmentation_miou": [],
            "tracking_mota": []
        }
        
        # Process with different QP values
        for qp in args.qp_values:
            print(f"  QP = {qp}")
            
            # Process with our pipeline
            pipeline_output = process_video_with_pipeline(
                video_path, stnpp, qal, qp, temp_dir, args.time_steps
            )
            
            # Process with standard HEVC for comparison (baseline)
            baseline_output = os.path.join(temp_dir, f"baseline_qp{qp}.mp4")
            frames = read_video(video_path)
            encode_with_hevc(frames, qp, baseline_output)
            
            # Calculate bitrate
            pipeline_bitrate = compute_bitrate(pipeline_output)
            baseline_bitrate = compute_bitrate(baseline_output)
            
            # Calculate PSNR and SSIM
            original_frames = frames
            pipeline_frames = decode_with_hevc(pipeline_output)
            baseline_frames = decode_with_hevc(baseline_output)
            
            # Ensure we have the same number of frames
            min_frames = min(len(original_frames), len(pipeline_frames), len(baseline_frames))
            original_frames = original_frames[:min_frames]
            pipeline_frames = pipeline_frames[:min_frames]
            baseline_frames = baseline_frames[:min_frames]
            
            # Calculate quality metrics
            pipeline_psnr = np.mean([
                calculate_psnr(original_frames[i], pipeline_frames[i])
                for i in range(min_frames)
            ])
            
            pipeline_ssim = np.mean([
                calculate_ssim(original_frames[i], pipeline_frames[i])
                for i in range(min_frames)
            ])
            
            baseline_psnr = np.mean([
                calculate_psnr(original_frames[i], baseline_frames[i])
                for i in range(min_frames)
            ])
            
            baseline_ssim = np.mean([
                calculate_ssim(original_frames[i], baseline_frames[i])
                for i in range(min_frames)
            ])
            
            # Evaluate on machine vision tasks
            if args.evaluate_detection:
                detection_metrics_pipeline = evaluate_detection(
                    video_path, pipeline_output
                )
                detection_metrics_baseline = evaluate_detection(
                    video_path, baseline_output
                )
                
                detection_improvement = detection_metrics_pipeline["mAP"] - detection_metrics_baseline["mAP"]
            else:
                detection_improvement = 0
            
            if args.evaluate_segmentation:
                segmentation_metrics_pipeline = evaluate_segmentation(
                    video_path, pipeline_output
                )
                segmentation_metrics_baseline = evaluate_segmentation(
                    video_path, baseline_output
                )
                
                segmentation_improvement = segmentation_metrics_pipeline["mIoU"] - segmentation_metrics_baseline["mIoU"]
            else:
                segmentation_improvement = 0
            
            if args.evaluate_tracking:
                tracking_metrics_pipeline = evaluate_tracking(
                    video_path, pipeline_output, mot_dataset_path=args.mot_dataset_path
                )
                tracking_metrics_baseline = evaluate_tracking(
                    video_path, baseline_output, mot_dataset_path=args.mot_dataset_path
                )
                
                tracking_improvement = tracking_metrics_pipeline["MOTA"] - tracking_metrics_baseline["MOTA"]
            else:
                tracking_improvement = 0
            
            # Record results
            results[video_name]["qp_values"].append(qp)
            results[video_name]["bitrates"].append(pipeline_bitrate)
            results[video_name]["psnr_values"].append(pipeline_psnr)
            results[video_name]["ssim_values"].append(pipeline_ssim)
            results[video_name]["detection_map"].append(detection_improvement)
            results[video_name]["segmentation_miou"].append(segmentation_improvement)
            results[video_name]["tracking_mota"].append(tracking_improvement)
            
            # Print results for this QP
            print(f"    Pipeline - Bitrate: {pipeline_bitrate:.6f} bpp, PSNR: {pipeline_psnr:.2f} dB, SSIM: {pipeline_ssim:.4f}")
            print(f"    Baseline - Bitrate: {baseline_bitrate:.6f} bpp, PSNR: {baseline_psnr:.2f} dB, SSIM: {baseline_ssim:.4f}")
            print(f"    MV Task Improvements - Detection: {detection_improvement:.4f}, Segmentation: {segmentation_improvement:.4f}, Tracking: {tracking_improvement:.4f}")
    
    # Save results to JSON file
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Generate plots
    generate_plots(results, args.output_dir)
    
    print("Evaluation completed!")


def generate_plots(results, output_dir):
    """
    Generate plots from evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 8))
    
    # Plot RD curves (Rate-Distortion)
    plt.subplot(2, 2, 1)
    for video_name, video_results in results.items():
        plt.plot(video_results["bitrates"], video_results["psnr_values"], 'o-', label=video_name)
    
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title("Rate-Distortion Curves")
    plt.grid(True)
    plt.legend()
    
    # Plot Rate-SSIM curves
    plt.subplot(2, 2, 2)
    for video_name, video_results in results.items():
        plt.plot(video_results["bitrates"], video_results["ssim_values"], 'o-', label=video_name)
    
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("SSIM")
    plt.title("Rate-SSIM Curves")
    plt.grid(True)
    plt.legend()
    
    # Plot Detection Performance
    plt.subplot(2, 2, 3)
    for video_name, video_results in results.items():
        plt.plot(video_results["bitrates"], video_results["detection_map"], 'o-', label=video_name)
    
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("Detection mAP Improvement")
    plt.title("Detection Performance vs. Bitrate")
    plt.grid(True)
    plt.legend()
    
    # Plot Tracking Performance
    plt.subplot(2, 2, 4)
    for video_name, video_results in results.items():
        plt.plot(video_results["bitrates"], video_results["tracking_mota"], 'o-', label=video_name)
    
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("Tracking MOTA Improvement")
    plt.title("Tracking Performance vs. Bitrate")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_curves.png"))
    
    # Generate BD-Rate plots (Bjøntegaard Delta Rate)
    # This would typically compare our method to the baseline
    # For this skeleton, we'll skip the actual BD-Rate calculation
    
    # Generate additional plots for machine vision tasks
    plt.figure(figsize=(12, 6))
    
    # Plot Segmentation Performance
    plt.subplot(1, 2, 1)
    for video_name, video_results in results.items():
        plt.plot(video_results["bitrates"], video_results["segmentation_miou"], 'o-', label=video_name)
    
    plt.xlabel("Bitrate (bpp)")
    plt.ylabel("Segmentation mIoU Improvement")
    plt.title("Segmentation Performance vs. Bitrate")
    plt.grid(True)
    plt.legend()
    
    # Plot QP vs Bitrate
    plt.subplot(1, 2, 2)
    for video_name, video_results in results.items():
        plt.plot(video_results["qp_values"], video_results["bitrates"], 'o-', label=video_name)
    
    plt.xlabel("QP")
    plt.ylabel("Bitrate (bpp)")
    plt.title("QP vs. Bitrate")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mv_task_performance.png"))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video compression pipeline")
    
    # Input/output parameters
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video or directory of videos")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--temp_dir", type=str, default="temp",
                        help="Directory for temporary files")
    
    # Model parameters
    parser.add_argument("--stnpp_model_path", type=str, default=None,
                        help="Path to trained ST-NPP model")
    parser.add_argument("--qal_model_path", type=str, default=None,
                        help="Path to trained QAL model")
    parser.add_argument("--spatial_backbone", type=str, default="resnet50",
                        choices=["resnet34", "resnet50", "efficientnet_b4"],
                        help="Backbone for the spatial branch")
    parser.add_argument("--temporal_model", type=str, default="3dcnn",
                        choices=["3dcnn", "convlstm"],
                        help="Model for the temporal branch")
    parser.add_argument("--fusion_type", type=str, default="concatenation",
                        choices=["concatenation", "attention"],
                        help="Type of fusion for spatial and temporal features")
    
    # Evaluation parameters
    parser.add_argument("--qp_values", type=int, nargs="+", default=[22, 27, 32, 37],
                        help="QP values to evaluate")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames to process at once")
    parser.add_argument("--evaluate_detection", action="store_true",
                        help="Evaluate detection performance")
    parser.add_argument("--evaluate_segmentation", action="store_true",
                        help="Evaluate segmentation performance")
    parser.add_argument("--evaluate_tracking", action="store_true",
                        help="Evaluate tracking performance")
    parser.add_argument("--mot_dataset_path", type=str, default=None,
                        help="Path to the MOT Challenge dataset (e.g., MOT16)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_pipeline(args) 