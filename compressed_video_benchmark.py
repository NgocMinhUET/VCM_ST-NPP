#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task-Aware Video Compression Benchmark
-------------------------------------
This script benchmarks our task-aware compression against traditional codecs
(H.264, H.265, VVC) on various datasets and metrics.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from models.combined_model import CombinedModel
from utils.data_utils import get_dataloader
from utils.metric_utils import calculate_psnr, calculate_ssim, calculate_msssim
from utils.metric_utils import compute_detection_metrics, compute_segmentation_metrics, compute_tracking_metrics
from utils.video_utils import encode_video_with_ffmpeg, decode_video_with_ffmpeg
from utils.common_utils import create_logger, setup_seed

# Supported codecs for comparison
CODECS = ['h264', 'h265', 'vvc', 'task_aware']

# QP values to test for rate-distortion curves
QP_VALUES = [20, 25, 30, 35, 40, 45]

def parse_args():
    parser = argparse.ArgumentParser(description='Task-Aware Video Compression Benchmark')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--task_type', type=str, required=True, choices=['detection', 'segmentation', 'tracking'], 
                        help='Task type to evaluate')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to use')
    parser.add_argument('--seq_length', type=int, default=5, help='Sequence length for video processing')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--codecs', nargs='+', default=CODECS, choices=CODECS, 
                      help='Codecs to benchmark against')
    parser.add_argument('--qp_values', nargs='+', type=int, default=QP_VALUES, 
                      help='QP values to test')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='benchmark_results', 
                      help='Directory to save results')
    parser.add_argument('--save_videos', action='store_true', 
                      help='Save compressed videos for visual comparison')
    parser.add_argument('--vis_results', action='store_true', 
                      help='Visualize and save result plots')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run the model on')
    parser.add_argument('--num_workers', type=int, default=4, 
                      help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=1, 
                      help='Batch size for evaluation')
    
    # Compression parameters
    parser.add_argument('--crf', type=int, default=None,
                      help='Use CRF instead of QP for traditional codecs')
    parser.add_argument('--preset', type=str, default='medium',
                      help='Encoding preset for traditional codecs')
    
    args = parser.parse_args()
    return args

def prepare_output_directory(args):
    """Create output directory structure for results"""
    base_dir = Path(args.output_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories for each codec
    for codec in args.codecs:
        codec_dir = base_dir / codec
        codec_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for compressed videos if requested
        if args.save_videos:
            video_dir = codec_dir / 'videos'
            video_dir.mkdir(exist_ok=True)
    
    # Create directory for plots
    if args.vis_results:
        plot_dir = base_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
    
    return base_dir

def load_task_aware_model(args):
    """Load the task-aware compression model from checkpoint"""
    print(f"Loading task-aware model from {args.checkpoint}")
    model = CombinedModel(
        task_type=args.task_type,
        load_path=args.checkpoint
    )
    model = model.to(args.device)
    model.eval()
    return model

def benchmark_traditional_codec(args, codec, qp, dataloader, output_dir):
    """Benchmark a traditional codec (H.264, H.265, VVC)"""
    results = {
        'psnr': [],
        'ssim': [],
        'msssim': [],
        'bitrate': [],
        'encoding_time': [],
        'decoding_time': []
    }
    
    # Add task-specific metrics
    if args.task_type == 'detection':
        results['mAP'] = []
        results['mAP50'] = []
        results['mAP75'] = []
    elif args.task_type == 'segmentation':
        results['mIoU'] = []
        results['pixelAcc'] = []
    elif args.task_type == 'tracking':
        results['MOTA'] = []
        results['IDF1'] = []
    
    codec_dir = output_dir / codec
    video_dir = codec_dir / 'videos' if args.save_videos else None
    
    for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {codec.upper()} (QP={qp})")):
        frames, labels = batch['frames'], batch['labels']
        sequence_name = batch.get('sequence_name', f"sequence_{i}")
        
        # Use original frames as ground truth
        original_frames = frames.cpu().numpy()
        
        # Compress and decompress with traditional codec
        temp_video_path = f"temp_{codec}_{qp}_{i}.mp4"
        output_video_path = video_dir / f"{sequence_name}_{qp}.mp4" if args.save_videos else None
        
        # Encode frames to video
        start_time = time.time()
        encode_video_with_ffmpeg(
            frames=original_frames,
            output_path=temp_video_path,
            codec=codec,
            qp=qp,
            crf=args.crf,
            preset=args.preset
        )
        encoding_time = time.time() - start_time
        
        # Calculate bitrate
        file_size_bytes = os.path.getsize(temp_video_path)
        duration_seconds = original_frames.shape[0] / 30  # Assuming 30 FPS
        bitrate_kbps = (file_size_bytes * 8) / (1000 * duration_seconds)
        
        # Decode video back to frames
        start_time = time.time()
        decoded_frames = decode_video_with_ffmpeg(temp_video_path)
        decoding_time = time.time() - start_time
        
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        # Save compressed video if requested
        if args.save_videos and output_video_path:
            os.rename(temp_video_path, output_video_path)
        
        # Calculate quality metrics
        psnr_vals = []
        ssim_vals = []
        msssim_vals = []
        
        for j in range(len(original_frames)):
            org_frame = original_frames[j]
            dec_frame = decoded_frames[j]
            
            # Calculate PSNR, SSIM, MS-SSIM
            psnr = calculate_psnr(org_frame, dec_frame)
            ssim = calculate_ssim(org_frame, dec_frame)
            msssim = calculate_msssim(org_frame, dec_frame)
            
            psnr_vals.append(psnr)
            ssim_vals.append(ssim)
            msssim_vals.append(msssim)
        
        results['psnr'].append(np.mean(psnr_vals))
        results['ssim'].append(np.mean(ssim_vals))
        results['msssim'].append(np.mean(msssim_vals))
        results['bitrate'].append(bitrate_kbps)
        results['encoding_time'].append(encoding_time)
        results['decoding_time'].append(decoding_time)
        
        # Calculate task-specific metrics
        if args.task_type == 'detection':
            metrics = compute_detection_metrics(decoded_frames, labels)
            results['mAP'].append(metrics['mAP'])
            results['mAP50'].append(metrics['mAP50'])
            results['mAP75'].append(metrics['mAP75'])
        elif args.task_type == 'segmentation':
            metrics = compute_segmentation_metrics(decoded_frames, labels)
            results['mIoU'].append(metrics['mIoU'])
            results['pixelAcc'].append(metrics['pixelAcc'])
        elif args.task_type == 'tracking':
            metrics = compute_tracking_metrics(decoded_frames, labels)
            results['MOTA'].append(metrics['MOTA'])
            results['IDF1'].append(metrics['IDF1'])
    
    # Compute averages
    avg_results = {key: np.mean(values) for key, values in results.items()}
    
    return avg_results

def benchmark_task_aware_codec(args, qp, dataloader, model, output_dir):
    """Benchmark our task-aware codec"""
    results = {
        'psnr': [],
        'ssim': [],
        'msssim': [],
        'bitrate': [],
        'encoding_time': [],
        'decoding_time': []
    }
    
    # Add task-specific metrics
    if args.task_type == 'detection':
        results['mAP'] = []
        results['mAP50'] = []
        results['mAP75'] = []
    elif args.task_type == 'segmentation':
        results['mIoU'] = []
        results['pixelAcc'] = []
    elif args.task_type == 'tracking':
        results['MOTA'] = []
        results['IDF1'] = []
    
    codec_dir = output_dir / 'task_aware'
    video_dir = codec_dir / 'videos' if args.save_videos else None
    
    for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating Task-Aware Codec (QP={qp})")):
        frames, labels = batch['frames'], batch['labels']
        sequence_name = batch.get('sequence_name', f"sequence_{i}")
        
        frames = frames.to(args.device)
        
        # Use original frames as ground truth
        original_frames = frames.cpu().numpy()
        
        # Process through our model (encode and decode)
        with torch.no_grad():
            start_time = time.time()
            outputs = model(frames, qp=qp)
            
            # Unpack model outputs
            reconstructed_frames = outputs['reconstructed']
            encoding_time = time.time() - start_time
            
            # Estimate time for decoding only (approximate)
            start_time = time.time()
            _ = model.decode(outputs['latents'], qp=qp)
            decoding_time = time.time() - start_time
        
        # Convert reconstructed frames to numpy
        reconstructed_np = reconstructed_frames.cpu().numpy()
        
        # Save compressed video if requested
        if args.save_videos:
            output_video_path = video_dir / f"{sequence_name}_{qp}.mp4"
            # Convert frames to uint8 format
            frames_uint8 = (reconstructed_np * 255).astype(np.uint8)
            # Save as video
            height, width = frames_uint8.shape[2:4]
            video_writer = cv2.VideoWriter(
                str(output_video_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,  # FPS
                (width, height)
            )
            for frame in frames_uint8:
                # OpenCV expects BGR format
                bgr_frame = frame.transpose(1, 2, 0)[..., ::-1]
                video_writer.write(bgr_frame)
            video_writer.release()
        
        # Calculate quality metrics
        psnr_vals = []
        ssim_vals = []
        msssim_vals = []
        
        for j in range(len(original_frames)):
            org_frame = original_frames[j]
            rec_frame = reconstructed_np[j]
            
            # Calculate PSNR, SSIM, MS-SSIM
            psnr = calculate_psnr(org_frame, rec_frame)
            ssim = calculate_ssim(org_frame, rec_frame)
            msssim = calculate_msssim(org_frame, rec_frame)
            
            psnr_vals.append(psnr)
            ssim_vals.append(ssim)
            msssim_vals.append(msssim)
        
        # Estimate bitrate from model (bits per pixel * resolution * framerate)
        bpp = outputs.get('bpp', 0.1)  # Default if not provided
        resolution = frames.shape[3] * frames.shape[4]  # width * height
        framerate = 30  # Assumed
        bitrate_kbps = (bpp * resolution * framerate) / 1000
        
        results['psnr'].append(np.mean(psnr_vals))
        results['ssim'].append(np.mean(ssim_vals))
        results['msssim'].append(np.mean(msssim_vals))
        results['bitrate'].append(bitrate_kbps)
        results['encoding_time'].append(encoding_time)
        results['decoding_time'].append(decoding_time)
        
        # Calculate task-specific metrics
        if args.task_type == 'detection':
            metrics = outputs.get('metrics', compute_detection_metrics(reconstructed_np, labels))
            results['mAP'].append(metrics['mAP'])
            results['mAP50'].append(metrics['mAP50'])
            results['mAP75'].append(metrics['mAP75'])
        elif args.task_type == 'segmentation':
            metrics = outputs.get('metrics', compute_segmentation_metrics(reconstructed_np, labels))
            results['mIoU'].append(metrics['mIoU'])
            results['pixelAcc'].append(metrics['pixelAcc'])
        elif args.task_type == 'tracking':
            metrics = outputs.get('metrics', compute_tracking_metrics(reconstructed_np, labels))
            results['MOTA'].append(metrics['MOTA'])
            results['IDF1'].append(metrics['IDF1'])
    
    # Compute averages
    avg_results = {key: np.mean(values) for key, values in results.items()}
    
    return avg_results

def visualize_results(args, all_results, output_dir):
    """Create visualizations of benchmark results"""
    if not args.vis_results:
        return
    
    plot_dir = output_dir / 'plots'
    
    # Prepare data for plotting
    codecs = list(all_results.keys())
    qp_values = list(all_results[codecs[0]].keys())
    
    # Rate-distortion curves (PSNR vs Bitrate)
    plt.figure(figsize=(10, 6))
    for codec in codecs:
        bitrates = [all_results[codec][qp]['bitrate'] for qp in qp_values]
        psnrs = [all_results[codec][qp]['psnr'] for qp in qp_values]
        plt.plot(bitrates, psnrs, 'o-', label=codec)
    
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve: PSNR vs Bitrate')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / 'rate_distortion_psnr.png')
    
    # Rate-distortion curves (SSIM vs Bitrate)
    plt.figure(figsize=(10, 6))
    for codec in codecs:
        bitrates = [all_results[codec][qp]['bitrate'] for qp in qp_values]
        ssims = [all_results[codec][qp]['ssim'] for qp in qp_values]
        plt.plot(bitrates, ssims, 'o-', label=codec)
    
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('SSIM')
    plt.title('Rate-Distortion Curve: SSIM vs Bitrate')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / 'rate_distortion_ssim.png')
    
    # Task-specific metric vs Bitrate
    plt.figure(figsize=(10, 6))
    metric_name = 'mAP' if args.task_type == 'detection' else 'mIoU' if args.task_type == 'segmentation' else 'MOTA'
    
    for codec in codecs:
        bitrates = [all_results[codec][qp]['bitrate'] for qp in qp_values]
        metric_values = [all_results[codec][qp][metric_name] for qp in qp_values]
        plt.plot(bitrates, metric_values, 'o-', label=codec)
    
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel(metric_name)
    plt.title(f'Rate-Performance Curve: {metric_name} vs Bitrate')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / f'rate_performance_{metric_name.lower()}.png')
    
    # Encoding time comparison
    plt.figure(figsize=(10, 6))
    for codec in codecs:
        qps = [int(qp) for qp in qp_values]
        times = [all_results[codec][qp]['encoding_time'] for qp in qp_values]
        plt.plot(qps, times, 'o-', label=codec)
    
    plt.xlabel('QP')
    plt.ylabel('Encoding Time (s)')
    plt.title('Encoding Time vs QP')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / 'encoding_time.png')
    
    # Decoding time comparison
    plt.figure(figsize=(10, 6))
    for codec in codecs:
        qps = [int(qp) for qp in qp_values]
        times = [all_results[codec][qp]['decoding_time'] for qp in qp_values]
        plt.plot(qps, times, 'o-', label=codec)
    
    plt.xlabel('QP')
    plt.ylabel('Decoding Time (s)')
    plt.title('Decoding Time vs QP')
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir / 'decoding_time.png')
    
    print(f"Plots saved to {plot_dir}")

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    setup_seed(42)
    
    # Prepare output directory
    output_dir = prepare_output_directory(args)
    
    # Create logger
    logger = create_logger(output_dir / 'benchmark.log')
    logger.info(f"Starting benchmark with args: {args}")
    
    # Load task-aware model if needed
    task_aware_model = None
    if 'task_aware' in args.codecs:
        task_aware_model = load_task_aware_model(args)
    
    # Load dataset
    dataloader = get_dataloader(
        dataset_name=args.dataset,
        split=args.split,
        task_type=args.task_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_length=args.seq_length
    )
    
    # Store all results
    all_results = {codec: {} for codec in args.codecs}
    
    # Run benchmarks for each codec and QP value
    for qp in args.qp_values:
        for codec in args.codecs:
            logger.info(f"Benchmarking {codec} with QP={qp}")
            
            if codec == 'task_aware':
                results = benchmark_task_aware_codec(args, qp, dataloader, task_aware_model, output_dir)
            else:
                results = benchmark_traditional_codec(args, codec, qp, dataloader, output_dir)
            
            # Store results
            all_results[codec][qp] = results
            
            # Log results
            logger.info(f"Results for {codec} (QP={qp}):")
            for key, value in results.items():
                logger.info(f"  {key}: {value}")
    
    # Save all results as JSON
    with open(output_dir / 'benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create visualizations
    visualize_results(args, all_results, output_dir)
    
    logger.info("Benchmark completed!")

if __name__ == '__main__':
    main() 