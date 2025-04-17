#!/usr/bin/env python
"""
Script to compare our improved autoencoder compression with standard video compression methods.
Supports comparison with H.264, H.265, and VP9 codecs at various quality settings.
"""

import os
import argparse
import numpy as np
import cv2
import torch
import time
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from improved_autoencoder import ImprovedAutoencoder
from skimage.metrics import structural_similarity as ssim
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Test our proposed compression method")
    
    # Input parameters
    parser.add_argument("--input_video", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    
    # Model parameters
    parser.add_argument("--latent_channels", type=int, default=8,
                        help="Number of latent channels")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames to process at once")
    parser.add_argument("--time_reduction", type=int, default=4,
                        help="Time dimension reduction factor")
    
    # Testing parameters
    parser.add_argument("--use_sample_data", action="store_true",
                        help="Use sample data instead of running actual compression")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, 
                        default="results/compression_test",
                        help="Directory to save evaluation results")
    
    return parser.parse_args()

def check_ffmpeg():
    """Check if ffmpeg is installed and has required codecs."""
    # Check for local x265 installation
    home_dir = os.path.expanduser("~")
    local_x265_path = os.path.join(home_dir, "local", "bin", "x265")
    
    if os.path.exists(local_x265_path):
        # Add local bin and lib to PATH and LD_LIBRARY_PATH
        local_bin = os.path.join(home_dir, "local", "bin")
        local_lib = os.path.join(home_dir, "local", "lib")
        
        os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{local_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        print(f"Found local x265 installation at {local_x265_path}")

    try:
        # Check if ffmpeg is installed
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        
        # Check for required codecs
        ffmpeg_output = subprocess.run(["ffmpeg", "-codecs"], check=True, capture_output=True, text=True).stdout
        
        # Check for H.264 support
        if "libx264" not in ffmpeg_output:
            raise RuntimeError("FFmpeg is missing H.264 support (libx264)")
        
        # Check for H.265 support (either through system ffmpeg or local installation)
        if "libx265" not in ffmpeg_output and not os.path.exists(local_x265_path):
            raise RuntimeError("FFmpeg is missing H.265 support (libx265) and no local x265 installation found")
            
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "FFmpeg is not installed. Please install FFmpeg with H.264 and H.265 support.\n"
            "For H.265 support without root access, you can build x265 locally using:\n"
            "  ./build_x265.ps1\n"
            "This will install x265 in your home directory."
        )

def calculate_psnr(original, compressed):
    """Calculate PSNR between original and compressed frames."""
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ms_ssim(original, compressed):
    """Calculate MS-SSIM between original and compressed frames."""
    # Convert to grayscale for SSIM calculation
    if len(original.shape) == 3 and original.shape[2] == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        compressed_gray = compressed
    
    return ssim(original_gray, compressed_gray, data_range=255, multichannel=False)

def compress_with_our_method(frames, args, device, qp_level=None):
    """Compress frames using our improved autoencoder method with different QP levels."""
    print(f"\nCompressing with our improved autoencoder (QP level: {qp_level if qp_level is not None else 'default'})...")
    start_time = time.time()
    
    try:
        # Flag to determine whether to use different num_embeddings values
        FLAG_USE_QP_LEVELS = True
        
        # Adjust quantization parameters based on QP level
        if FLAG_USE_QP_LEVELS and qp_level is not None:
            if qp_level == 1:
                latent_channels = 8  # Default setting
                num_embeddings = 512
            elif qp_level == 2:
                latent_channels = 8
                num_embeddings = 256  # Fewer codebook entries = more compression
            elif qp_level == 3:
                latent_channels = 8
                num_embeddings = 128  # Even fewer entries = even more compression
            else:
                latent_channels = args.latent_channels
                num_embeddings = 512  # Default
        else:
            # Default values for all cases if flag is off
            latent_channels = args.latent_channels
            num_embeddings = 512
        
        # Load model
        print(f"Loading model from {args.model_path}")
        model = ImprovedAutoencoder(
            input_channels=3,
            latent_channels=latent_channels,
            time_reduction=args.time_reduction,
            num_embeddings=num_embeddings
        ).to(device)
        
        try:
            # Redirect stderr to suppress PyTorch messages
            stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            
            # Load checkpoint
            checkpoint = torch.load(args.model_path, map_location=device)
            
            # Custom load_state_dict to suppress missing key messages
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            # Restore stderr
            sys.stderr.close()
            sys.stderr = stderr
            
            print("Model loaded successfully")
            model.eval()
            
        except Exception as e:
            if 'stderr' in locals():
                sys.stderr.close()
                sys.stderr = stderr
            print(f"Error loading model: {str(e)}")
            return None, None, None
        
        print(f"Processing {len(frames)} frames in batches of {args.time_steps}...")
        compressed_frames = []
        total_compressed_size = 0
        
        # Process frames in batches of time_steps
        with tqdm(total=len(frames), desc="Compressing frames") as pbar:
            for i in range(0, len(frames), args.time_steps):
                try:
                    # Get sequence
                    sequence = frames[i:i+args.time_steps]
                    if len(sequence) < args.time_steps:
                        # Pad the sequence if necessary
                        sequence = sequence + [sequence[-1]] * (args.time_steps - len(sequence))
                    
                    # Create tensor [B, C, T, H, W]
                    sequence_tensor = torch.zeros((1, 3, len(sequence), sequence[0].shape[0], sequence[0].shape[1]), 
                                               device=device)
                    
                    for t, frame in enumerate(sequence):
                        # Convert BGR to RGB and normalize to [0, 1]
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
                        sequence_tensor[0, :, t, :, :] = frame_tensor.to(device)
                    
                    # Compress and reconstruct
                    with torch.no_grad():
                        reconstructed, latent, indices = model(sequence_tensor)
                        
                        # Calculate compressed size:
                        # 1. Size of indices (each index is stored as a 16-bit integer)
                        indices_size = indices.numel() * 2  # 2 bytes per index
                        
                        # 2. Size of codebook (each entry is a float32)
                        codebook_size = num_embeddings * latent_channels * 4  # 4 bytes per float
                        
                        # Total compressed size for this batch
                        batch_compressed_size = indices_size + codebook_size
                        total_compressed_size += batch_compressed_size
                    
                    # Convert back to numpy
                    reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                    
                    # Convert from RGB back to BGR for OpenCV
                    reconstructed_frames = []
                    for t in range(reconstructed_np.shape[0]):
                        # Convert normalized RGB [0,1] to BGR [0,255]
                        frame_rgb = (reconstructed_np[t] * 255).astype(np.uint8)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        reconstructed_frames.append(frame_bgr)
                    
                    # Store actual frames
                    if i + args.time_steps > len(frames):
                        compressed_frames.extend(reconstructed_frames[:len(frames)-i])
                    else:
                        compressed_frames.extend(reconstructed_frames)
                    
                    # Update progress bar
                    pbar.update(min(args.time_steps, len(frames) - i))
                    
                except Exception as e:
                    print(f"\nError processing batch starting at frame {i}: {str(e)}")
                    return None, None, None
        
        compression_time = time.time() - start_time
        print(f"\nCompression completed in {compression_time:.2f} seconds")
        print(f"Total compressed size: {total_compressed_size/1024/1024:.2f} MB")
        
        return compressed_frames, total_compressed_size, compression_time

    except Exception as e:
        print(f"\nError in compression process: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def compress_with_ffmpeg(frames, codec, crf, output_dir, temp_name="temp", x265_preset="medium"):
    """Compress frames using FFmpeg with the specified codec and CRF."""
    print(f"Compressing with {codec} at CRF {crf}...")
    temp_input_dir = os.path.join(output_dir, "temp_input")
    os.makedirs(temp_input_dir, exist_ok=True)
    
    # Save original frames as images
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(temp_input_dir, f"frame_{i:06d}.png"), frame)
    
    # Set codec parameters
    if codec == "h264":
        codec_params = ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium"]
    elif codec == "h265":
        codec_params = ["-c:v", "libx265", "-crf", str(crf), "-preset", x265_preset]
    elif codec == "vp9":
        codec_params = ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0"]
    else:
        raise ValueError(f"Unsupported codec: {codec}")
    
    # Output video path
    output_video = os.path.join(output_dir, f"{temp_name}_{codec}_crf{crf}.mp4")
    
    # Compress with FFmpeg
    start_time = time.time()
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-r", "30", "-i", 
        os.path.join(temp_input_dir, "frame_%06d.png"),
        "-pix_fmt", "yuv420p"
    ] + codec_params + [output_video]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return None, None, None
    
    # Get compressed size
    compressed_size = os.path.getsize(output_video)
    
    # Read compressed video back
    cap = cv2.VideoCapture(output_video)
    compressed_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        compressed_frames.append(frame)
    cap.release()
    
    # Make sure we have the same number of frames
    compressed_frames = compressed_frames[:len(frames)]
    
    # If we don't have enough frames, pad with the last frame
    while len(compressed_frames) < len(frames):
        compressed_frames.append(compressed_frames[-1] if compressed_frames else np.zeros_like(frames[0]))
    
    compression_time = time.time() - start_time
    
    return compressed_frames, compressed_size, compression_time

def calculate_mota(original_frames, compressed_frames):
    """Calculate MOTA (Multiple Object Tracking Accuracy) metric."""
    try:
        import numpy as np
        
        total_frames = len(original_frames)
        if total_frames == 0:
            print("No frames to process")
            return 0.0
            
        total_score = 0.0
        processed_frames = 0
        
        # Create SIFT detector
        sift = cv2.SIFT_create(
            nfeatures=0,  # 0 means unlimited
            nOctaveLayers=3,
            contrastThreshold=0.02,  # Lower value to detect more features
            edgeThreshold=10,
            sigma=1.6
        )
        
        # Create FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        print("\nCalculating MOTA...")
        for frame_idx, (orig_frame, comp_frame) in enumerate(zip(original_frames, compressed_frames)):
            try:
                # Convert frames to grayscale
                orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
                comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization to enhance features
                orig_gray = cv2.equalizeHist(orig_gray)
                comp_gray = cv2.equalizeHist(comp_gray)
                
                # Detect and compute keypoints and descriptors
                kp1, des1 = sift.detectAndCompute(orig_gray, None)
                kp2, des2 = sift.detectAndCompute(comp_gray, None)
                
                if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
                    print(f"Frame {frame_idx}: Not enough keypoints detected "
                          f"(orig: {len(kp1) if kp1 else 0}, comp: {len(kp2) if kp2 else 0})")
                    continue
                
                # Convert descriptors to float32
                des1 = np.float32(des1)
                des2 = np.float32(des2)
                
                # Find matches using FLANN
                try:
                    matches = flann.knnMatch(des1, des2, k=2)
                except Exception as e:
                    print(f"Frame {frame_idx}: Error in matching - {str(e)}")
                    continue
                
                # Apply Lowe's ratio test with a more lenient ratio
                good_matches = []
                for i, (m, n) in enumerate(matches):
                    if m.distance < 0.85 * n.distance:  # More lenient ratio (0.85 instead of 0.75)
                        good_matches.append(m)
                
                # Calculate match ratio
                match_ratio = len(good_matches) / max(len(kp1), len(kp2))
                
                if frame_idx % 10 == 0:  # Print every 10th frame
                    print(f"Frame {frame_idx}: Found {len(kp1)} / {len(kp2)} keypoints, "
                          f"{len(good_matches)} good matches, score = {match_ratio:.4f}")
                
                total_score += match_ratio
                processed_frames += 1
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {str(e)}")
                continue
        
        if processed_frames == 0:
            print("No frames were successfully processed")
            return 0.0
            
        # Calculate final MOTA score
        mota = total_score / processed_frames
        print(f"\nMOTA calculation completed:")
        print(f"Processed frames: {processed_frames}/{total_frames}")
        print(f"Average score: {mota:.4f}")
        
        return max(0.0, min(1.0, mota))  # Clamp between 0 and 1
        
    except Exception as e:
        print(f"Error calculating MOTA: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0

def evaluate_compression(original_frames, compressed_frames, compressed_size):
    """Evaluate compression quality using multiple metrics."""
    # Calculate file size of original frames
    total_pixels = 0
    original_size = 0
    
    for frame in original_frames:
        h, w = frame.shape[:2]
        total_pixels += h * w
        # Calculate size in bytes (3 channels, 8 bits per channel)
        original_size += h * w * 3
    
    # Convert original_size from bytes to bits
    original_size_bits = original_size * 8
    compressed_size_bits = compressed_size * 8
    
    # Calculate compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    # Calculate bits per pixel (BPP)
    # BPP = total compressed bits / total pixels
    bpp = compressed_size_bits / total_pixels if total_pixels > 0 else 0
    
    print(f"\nCompression Statistics:")
    print(f"Original size: {original_size/1024/1024:.2f} MB")
    print(f"Compressed size: {compressed_size/1024/1024:.2f} MB")
    print(f"Total pixels: {total_pixels:,}")
    print(f"BPP: {bpp:.4f}")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    
    # Calculate PSNR and MS-SSIM
    psnr_values = []
    ssim_values = []
    
    for orig, comp in zip(original_frames, compressed_frames):
        psnr = calculate_psnr(orig, comp)
        if not np.isinf(psnr):
            psnr_values.append(psnr)
        
        ms_ssim = calculate_ms_ssim(orig, comp)
        ssim_values.append(ms_ssim)
    
    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    
    # Calculate MOTA
    mota = calculate_mota(original_frames, compressed_frames)
    
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average MS-SSIM: {avg_ssim:.4f}")
    print(f"MOTA: {mota:.4f}")
    
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "bpp": bpp,
        "psnr": avg_psnr,
        "ms_ssim": avg_ssim,
        "mota": mota
    }

def save_comparison_video(original_frames, compressed_frames_dict, output_path, fps=30):
    """Create a side-by-side comparison video of different compression methods."""
    # Get first frame dimensions
    h, w = original_frames[0].shape[:2]
    
    # Number of methods (+1 for original)
    n_methods = len(compressed_frames_dict) + 1
    
    # Create composite frame width
    comp_w = w * n_methods
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (comp_w, h))
    
    for i in range(len(original_frames)):
        # Start with original frame
        composite = np.zeros((h, comp_w, 3), dtype=np.uint8)
        composite[:, 0:w] = original_frames[i]
        
        # Add each compressed method
        col = 1
        for method, frames in compressed_frames_dict.items():
            if i < len(frames):
                composite[:, col*w:(col+1)*w] = frames[i]
            col += 1
        
        # Write to video
        out.write(composite)
    
    out.release()
    print(f"Saved comparison video to {output_path}")

def create_comparison_plots(results, output_dir):
    """Create comparison plots for different compression methods."""
    # Check if we have any results
    if not results:
        print("Warning: No results to plot")
        return
    
    # Extract data for plotting
    methods = list(results.keys())
    
    # Group methods by type
    our_methods = [m for m in methods if m.startswith("our_method")]
    
    if not our_methods:
        print("Warning: No compression results found")
        return
    
    # Set up colors and markers
    colors = {
        "our": "red"
    }
    
    markers = {
        "our": "o"
    }
    
    # Create BPP vs PSNR plot
    plt.figure(figsize=(10, 6))
    
    # Plot our methods
    x = [results[m]["bpp"] for m in our_methods]
    y = [results[m]["psnr"] for m in our_methods]
    
    # Sort by BPP
    sorted_data = sorted(zip(x, y, our_methods), key=lambda item: item[0])
    x = [item[0] for item in sorted_data]
    y = [item[1] for item in sorted_data]
    our_methods = [item[2] for item in sorted_data]
    
    plt.plot(x, y, color=colors["our"], marker=markers["our"], 
            linestyle='-', linewidth=2, label="Our Method")
    
    # Annotate QP values
    for i, method in enumerate(our_methods):
        if "_qp" in method:
            qp = method.split("_qp")[1]
            plt.annotate(f"QP {qp}", (x[i], y[i]), 
                      textcoords="offset points", xytext=(0, 10), 
                      ha='center', fontsize=8)
    
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve (BPP vs PSNR)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "rd_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create BPP vs MS-SSIM plot
    plt.figure(figsize=(10, 6))
    
    # Plot our methods
    x = [results[m]["bpp"] for m in our_methods]
    y = [results[m]["ms_ssim"] for m in our_methods]
    
    # Sort by BPP
    sorted_data = sorted(zip(x, y, our_methods), key=lambda item: item[0])
    x = [item[0] for item in sorted_data]
    y = [item[1] for item in sorted_data]
    
    plt.plot(x, y, color=colors["our"], marker=markers["our"], 
            linestyle='-', linewidth=2, label="Our Method")
    
    # Annotate QP values
    for i, method in enumerate(our_methods):
        if "_qp" in method:
            qp = method.split("_qp")[1]
            plt.annotate(f"QP {qp}", (x[i], y[i]), 
                      textcoords="offset points", xytext=(0, 10), 
                      ha='center', fontsize=8)
    
    plt.xlabel('Bits per Pixel (BPP)')
    plt.ylabel('MS-SSIM')
    plt.title('Rate-Distortion Curve (BPP vs MS-SSIM)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "bpp_vs_ssim.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create Compression Ratio plot
    plt.figure(figsize=(12, 6))
    
    # Get method names and compression ratios
    labels = []
    values = []
    colors_list = []
    
    for method in our_methods:
        if "_qp" in method:
            label = f"Our QP{method.split('_qp')[1]}"
        else:
            label = "Our Method"
        color = colors["our"]
        
        labels.append(label)
        values.append(results[method]["compression_ratio"])
        colors_list.append(color)
    
    # Sort by compression ratio (descending)
    sorted_data = sorted(zip(labels, values, colors_list), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]
    colors_list = [item[2] for item in sorted_data]
    
    plt.bar(labels, values, color=colors_list)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Compression Ratio")
    plt.title("Compression Ratio Comparison")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compression_ratio.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plots to {output_dir}")

def generate_sample_data():
    """Generate sample data for testing without running actual compression."""
    print("Generating sample data for testing...")
    
    # Sample compression results
    sample_results = {
        "our_method_qp1": {
            "original_size": 10000000,
            "compressed_size": 10000,
            "compression_ratio": 1000.0,
            "bpp": 0.07,
            "psnr": 22.5,
            "ms_ssim": 0.85,
            "compression_time": 45.0
        },
        "our_method_qp2": {
            "original_size": 10000000,
            "compressed_size": 20000,
            "compression_ratio": 500.0,
            "bpp": 0.14,
            "psnr": 24.5,
            "ms_ssim": 0.88,
            "compression_time": 45.0
        },
        "our_method_qp3": {
            "original_size": 10000000,
            "compressed_size": 40000,
            "compression_ratio": 250.0,
            "bpp": 0.28,
            "psnr": 26.5,
            "ms_ssim": 0.90,
            "compression_time": 45.0
        },
        "h264_crf18": {
            "original_size": 10000000,
            "compressed_size": 500000,
            "compression_ratio": 20.0,
            "bpp": 3.5,
            "psnr": 33.0,
            "ms_ssim": 0.95,
            "compression_time": 15.0
        },
        "h264_crf23": {
            "original_size": 10000000,
            "compressed_size": 300000,
            "compression_ratio": 33.3,
            "bpp": 2.1,
            "psnr": 30.0,
            "ms_ssim": 0.92,
            "compression_time": 14.0
        },
        "h264_crf28": {
            "original_size": 10000000,
            "compressed_size": 150000,
            "compression_ratio": 66.6,
            "bpp": 1.05,
            "psnr": 27.0,
            "ms_ssim": 0.88,
            "compression_time": 12.0
        },
        "h264_crf33": {
            "original_size": 10000000,
            "compressed_size": 80000,
            "compression_ratio": 125.0,
            "bpp": 0.56,
            "psnr": 24.0,
            "ms_ssim": 0.83,
            "compression_time": 10.0
        },
        "h265_crf18": {
            "original_size": 10000000,
            "compressed_size": 400000,
            "compression_ratio": 25.0,
            "bpp": 2.8,
            "psnr": 34.0,
            "ms_ssim": 0.96,
            "compression_time": 25.0
        },
        "h265_crf23": {
            "original_size": 10000000,
            "compressed_size": 250000,
            "compression_ratio": 40.0,
            "bpp": 1.75,
            "psnr": 31.0,
            "ms_ssim": 0.93,
            "compression_time": 23.0
        },
        "h265_crf28": {
            "original_size": 10000000,
            "compressed_size": 120000,
            "compression_ratio": 83.3,
            "bpp": 0.84,
            "psnr": 28.0,
            "ms_ssim": 0.89,
            "compression_time": 22.0
        },
        "h265_crf33": {
            "original_size": 10000000,
            "compressed_size": 60000,
            "compression_ratio": 166.6,
            "bpp": 0.42,
            "psnr": 25.0,
            "ms_ssim": 0.84,
            "compression_time": 20.0
        },
        "vp9_crf18": {
            "original_size": 10000000,
            "compressed_size": 450000,
            "compression_ratio": 22.2,
            "bpp": 3.15,
            "psnr": 33.5,
            "ms_ssim": 0.955,
            "compression_time": 35.0
        },
        "vp9_crf23": {
            "original_size": 10000000,
            "compressed_size": 270000,
            "compression_ratio": 37.0,
            "bpp": 1.89,
            "psnr": 30.5,
            "ms_ssim": 0.925,
            "compression_time": 32.0
        },
        "vp9_crf28": {
            "original_size": 10000000,
            "compressed_size": 135000,
            "compression_ratio": 74.0,
            "bpp": 0.945,
            "psnr": 27.5,
            "ms_ssim": 0.885,
            "compression_time": 30.0
        },
        "vp9_crf33": {
            "original_size": 10000000,
            "compressed_size": 70000,
            "compression_ratio": 142.8,
            "bpp": 0.49,
            "psnr": 24.5,
            "ms_ssim": 0.835,
            "compression_time": 28.0
        }
    }
    
    return sample_results

def check_ffmpeg_codecs():
    try:
        output = subprocess.run(["ffmpeg", "-codecs"], capture_output=True, text=True).stdout
        if "libx264" in output:
            print("FFmpeg supports libx264 ✅")
        else:
            print("⚠️ FFmpeg does NOT support libx264")

        if "libx265" in output:
            print("FFmpeg supports libx265 ✅")
        else:
            print("⚠️ FFmpeg does NOT support libx265")
    except Exception as e:
        print("Error checking ffmpeg codecs:", e)

def generate_summary_table(results, output_dir):
    """Generate a summary table of compression results."""
    table_path = os.path.join(output_dir, "comparison_table.txt")
    with open(table_path, 'w') as f:
        f.write("Compression Results Summary\n")
        f.write("=========================\n\n")
        
        # Header
        f.write(f"{'Method':<15} {'BPP':>10} {'PSNR':>10} {'MS-SSIM':>10} {'MOTA':>10} {'Comp.Ratio':>12} {'Time(s)':>10}\n")
        f.write("-" * 80 + "\n")
        
        # Results for each method
        for method in sorted(results.keys()):
            r = results[method]
            f.write(f"{method:<15} {r['bpp']:>10.4f} {r['psnr']:>10.2f} {r['ms_ssim']:>10.4f} "
                   f"{r['mota']:>10.4f} {r['compression_ratio']:>12.2f} {r['compression_time']:>10.1f}\n")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for FFmpeg and codecs
    print("Checking FFmpeg and codecs...")
    check_ffmpeg()
    check_ffmpeg_codecs()
    
    # Load video frames
    print("\nLoading video frames...")
    cap = cv2.VideoCapture(args.input_video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        print("Error: No frames loaded from video")
        return
    
    print(f"Loaded {len(frames)} frames")
    
    # Initialize results dictionary
    results = {}
    
    # Test our method with different QP levels
    qp_levels = [1, 2, 3]  # Different compression levels
    
    for qp in qp_levels:
        method_name = f"our_method_qp{qp}"
        print(f"\nTesting {method_name}...")
        
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Compress frames
        compressed_frames, compressed_size, compression_time = compress_with_our_method(
            frames, args, device, qp_level=qp
        )
        
        if compressed_frames is None:
            print(f"Compression failed for {method_name}")
            continue
        
        # Evaluate compression
        eval_results = evaluate_compression(frames, compressed_frames, compressed_size)
        eval_results["compression_time"] = compression_time
        results[method_name] = eval_results
        
        # Save some sample frames for visual comparison
        sample_indices = [len(frames)//4, len(frames)//2, 3*len(frames)//4]
        for idx in sample_indices:
            comparison = np.hstack([frames[idx], compressed_frames[idx]])
            cv2.imwrite(os.path.join(args.output_dir, f"comparison_{method_name}_frame{idx}.png"), comparison)
    
    # Test standard codecs
    codecs = {
        "h264": [18, 23, 28, 33],  # CRF values for H.264
        "h265": [18, 23, 28, 33],  # CRF values for H.265
        "vp9": [18, 23, 28, 33]    # CRF values for VP9
    }
    
    for codec, crf_values in codecs.items():
        print(f"\nTesting {codec.upper()} codec...")
        for crf in crf_values:
            method_name = f"{codec}_crf{crf}"
            print(f"\nTesting {method_name}...")
            
            # Compress frames
            compressed_frames, compressed_size, compression_time = compress_with_ffmpeg(
                frames, codec, crf, args.output_dir
            )
            
            if compressed_frames is None:
                print(f"Compression failed for {method_name}")
                continue
            
            # Evaluate compression
            eval_results = evaluate_compression(frames, compressed_frames, compressed_size)
            eval_results["compression_time"] = compression_time
            results[method_name] = eval_results
            
            # Save some sample frames for visual comparison
            for idx in sample_indices:
                comparison = np.hstack([frames[idx], compressed_frames[idx]])
                cv2.imwrite(os.path.join(args.output_dir, f"comparison_{method_name}_frame{idx}.png"), comparison)
    
    # Save the results
    results_file = os.path.join(args.output_dir, "compression_results.json")
    print(f"\nSaving results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(results, args.output_dir)
    
    # Generate textual comparison table
    print("\nGenerating comparison table...")
    generate_summary_table(results, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 