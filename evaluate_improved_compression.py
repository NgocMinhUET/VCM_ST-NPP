#!/usr/bin/env python
"""
Evaluation script for the improved autoencoder with vector quantization.
This script evaluates compression performance on sample videos and calculates metrics.
"""

import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from improved_autoencoder import ImprovedAutoencoder
from video_compression import extract_frames, create_video

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate improved autoencoder compression")
    
    # Input parameters
    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to input video file or image sequence (using format like %06d.jpg)")
    parser.add_argument("--model_path", type=str, 
                        default="trained_models/improved_autoencoder/autoencoder_best.pt",
                        help="Path to trained model checkpoint")
    
    # Model parameters
    parser.add_argument("--latent_channels", type=int, default=8,
                        help="Number of channels in latent space")
    parser.add_argument("--time_reduction", type=int, default=2,
                        help="Temporal reduction factor")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames to process at once")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, 
                        default="results/improved_compression",
                        help="Directory to save evaluation results")
    
    return parser.parse_args()

def calculate_psnr(original, reconstructed):
    """Calculate PSNR between original and reconstructed frames."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def compress_and_evaluate(model, frames, time_steps, device):
    """Compress video frames and evaluate performance."""
    total_frames = len(frames)
    compressed_data = []
    reconstructed_frames = []
    bpp_values = []
    psnr_values = []
    
    # Process frames in sequences
    for i in tqdm(range(0, total_frames, time_steps), desc="Processing frames"):
        # Get sequence of frames
        if i + time_steps > total_frames:
            sequence = frames[i:] + frames[i:i+time_steps-len(frames[i:])]
        else:
            sequence = frames[i:i+time_steps]
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).permute(3, 0, 1, 2).unsqueeze(0)
        sequence_tensor = sequence_tensor.to(device)
        
        # Compress and reconstruct
        with torch.no_grad():
            reconstructed, _, latent = model(sequence_tensor)
            bpp = model.calculate_bitrate(latent)
        
        # Store compressed data and calculate metrics
        compressed_data.append(latent.cpu().numpy())
        reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        
        # Only store the actual frames (not padding)
        if i + time_steps > total_frames:
            reconstructed_frames.extend(reconstructed_np[:total_frames-i])
            # Calculate metrics for actual frames
            for j in range(total_frames-i):
                psnr = calculate_psnr(sequence[j], reconstructed_np[j])
                psnr_values.append(psnr)
                bpp_values.append(bpp)
        else:
            reconstructed_frames.extend(reconstructed_np)
            # Calculate metrics for all frames
            for j in range(time_steps):
                psnr = calculate_psnr(sequence[j], reconstructed_np[j])
                psnr_values.append(psnr)
                bpp_values.append(bpp)
    
    return compressed_data, reconstructed_frames, np.mean(psnr_values), np.mean(bpp_values)

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
    
    # Sử dụng weights_only=True để tránh cảnh báo bảo mật
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract frames from video
    print("Extracting frames...")
    frames_result = extract_frames(args.input_video)
    
    # Xử lý kết quả trả về từ extract_frames, giờ hàm này trả về tuple
    if isinstance(frames_result, tuple) and len(frames_result) == 5:
        frames, fps, width, height, frame_count = frames_result
    else:
        # Trường hợp phù hợp với interface cũ (nếu chưa cập nhật video_compression.py)
        frames = frames_result
        
    print(f"Extracted {len(frames)} frames")
    
    # Compress and evaluate
    print("Compressing and evaluating...")
    compressed_data, reconstructed_frames, avg_psnr, avg_bpp = compress_and_evaluate(
        model, frames, args.time_steps, device
    )
    
    # Calculate compression ratio
    original_size = sum(frame.nbytes for frame in frames)
    compressed_size = sum(data.nbytes for data in compressed_data)
    compression_ratio = original_size / compressed_size
    
    # Save results
    results = {
        'compression_ratio': compression_ratio,
        'average_psnr': avg_psnr,
        'average_bpp': avg_bpp,
        'original_size_mb': original_size / (1024 * 1024),
        'compressed_size_mb': compressed_size / (1024 * 1024)
    }
    
    # Save metrics to file
    results_path = os.path.join(args.output_dir, "compression_results.txt")
    with open(results_path, 'w') as f:
        f.write("Improved Autoencoder Compression Results\n")
        f.write("=====================================\n\n")
        f.write(f"Compression Ratio: {compression_ratio:.2f}x\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average BPP: {avg_bpp:.4f}\n")
        f.write(f"Original Size: {results['original_size_mb']:.2f} MB\n")
        f.write(f"Compressed Size: {results['compressed_size_mb']:.2f} MB\n")
    
    print(f"Saved results to {results_path}")
    
    # Create reconstructed video
    output_video_path = os.path.join(args.output_dir, "reconstructed.mp4")
    
    # Nếu có thông tin về fps, width, height từ extract_frames
    if isinstance(frames_result, tuple) and len(frames_result) >= 4:
        create_video(reconstructed_frames, output_video_path, fps, width, height)
    else:
        # Sử dụng phiên bản create_video không cần fps, width, height (nếu hàm create_video có hỗ trợ)
        create_video(reconstructed_frames, output_video_path)
        
    print(f"Saved reconstructed video to {output_video_path}")
    
    # Create comparison video
    comparison_frames = []
    for orig, recon in zip(frames, reconstructed_frames):
        # Convert to uint8 for video writing
        orig_uint8 = (orig * 255).astype(np.uint8)
        recon_uint8 = (recon * 255).astype(np.uint8)
        # Create side-by-side comparison
        comparison = np.hstack((orig_uint8, recon_uint8))
        comparison_frames.append(comparison)
    
    comparison_video_path = os.path.join(args.output_dir, "comparison.mp4")
    
    # Xử lý tương tự với comparison video
    if isinstance(frames_result, tuple) and len(frames_result) >= 4:
        # Chiều rộng gấp đôi vì là video so sánh side-by-side
        create_video(comparison_frames, comparison_video_path, fps, width*2, height)
    else:
        create_video(comparison_frames, comparison_video_path)
        
    print(f"Saved comparison video to {comparison_video_path}")

if __name__ == "__main__":
    main() 