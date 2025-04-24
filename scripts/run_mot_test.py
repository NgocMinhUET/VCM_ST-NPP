#!/usr/bin/env python3
"""
MOT16 Testing Script for Task-Aware Video Compression with Tracking.

This script runs our video compression and tracking model on MOT16 sequences,
saving tracking output and visualizations for evaluation.
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.combined_model import TaskAwareVideoProcessor
from models.st_npp import STNPP
from models.qal import QAL
from models.proxy_codec import ProxyCodec
from models.task_networks.tracker import VideoObjectTracker
from utils.data_utils import MOTDataset
from utils.model_utils import load_checkpoint
from utils.metric_utils import calculate_bpp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test video compression and tracking on MOT16 dataset")
    
    # Dataset parameters
    parser.add_argument("--data_dir", type=str, default="D:/NCS/propose/dataset/MOT16", 
                        help="Path to MOT16 dataset directory")
    parser.add_argument("--sequence", type=str, default="MOT16-04", 
                        help="Specific MOT16 sequence to process (e.g., MOT16-04)")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save output visualizations and tracking results")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--clip_length", type=int, default=8, 
                        help="Number of frames to process at once")
    parser.add_argument("--resolution", type=int, default=640, 
                        help="Input resolution (width and height)")
    parser.add_argument("--feature_channels", type=int, default=128, 
                        help="Number of feature channels")
    parser.add_argument("--latent_channels", type=int, default=64, 
                        help="Number of latent channels")
    parser.add_argument("--use_quantization", action="store_true", 
                        help="Use quantization in the model")
    parser.add_argument("--qp", type=int, default=4, 
                        help="Quantization parameter (0-7, higher means more compression)")
    
    # Misc parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--visualize", action="store_true", 
                        help="Generate visualizations of tracking results")
    parser.add_argument("--save_frames", action="store_true", 
                        help="Save individual output frames")
    parser.add_argument("--save_video", action="store_true", 
                        help="Save output as video")
    
    return parser.parse_args()


def build_model(args):
    """Build the combined model."""
    # Create ST-NPP model
    st_npp = STNPP(
        channels=3,
        feature_channels=args.feature_channels,
        num_frames=args.clip_length,
        use_attention=True
    )
    
    # Create QAL model
    qal = QAL(
        feature_channels=args.feature_channels,
        latent_channels=args.latent_channels,
        num_frames=args.clip_length,
        num_qp_levels=8
    )
    
    # Create proxy codec
    proxy_codec = ProxyCodec(
        input_channels=args.latent_channels,
        num_qp_levels=8
    )
    
    # Create task network
    task_network = VideoObjectTracker(
        input_channels=args.feature_channels,
        num_frames=args.clip_length
    )
    
    # Create combined model
    model = TaskAwareVideoProcessor(
        st_npp=st_npp,
        qal=qal,
        proxy_codec=proxy_codec,
        task_network=task_network,
        task_type="tracking",
        use_quantization=args.use_quantization
    )
    
    return model


def load_mot_sequence(sequence_path, clip_length=8, resolution=(640, 640)):
    """Load frames from a MOT16 sequence."""
    # Get image directory
    img_dir = os.path.join(sequence_path, "img1")
    if not os.path.exists(img_dir):
        raise ValueError(f"Image directory not found: {img_dir}")
    
    # Get all image files and sort
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    # Read frames
    frames = []
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to desired resolution
        frame = cv2.resize(frame, (resolution[1], resolution[0]))
        
        # Convert to tensor and normalize
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2, 0, 1)  # [C, H, W]
        frames.append(frame)
    
    # Stack frames into a single tensor
    frames_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]
    
    return frames_tensor


def process_video(model, frames, args):
    """Process video frames through the model and generate tracking results."""
    num_frames = frames.shape[0]
    clip_length = args.clip_length
    device = args.device
    
    # Process frames in chunks of clip_length
    all_outputs = []
    all_reconstructed = []
    total_bpp = 0
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_frames, clip_length), desc="Processing frames"):
            end_idx = min(start_idx + clip_length, num_frames)
            
            # Extract clip
            clip = frames[start_idx:end_idx]
            
            # Pad if needed
            if clip.shape[0] < clip_length:
                padding = torch.zeros((clip_length - clip.shape[0], 3, clip.shape[2], clip.shape[3]))
                clip = torch.cat([clip, padding], dim=0)
            
            # Add batch dimension
            clip = clip.unsqueeze(0).to(device)  # [1, T, C, H, W]
            
            # Set QP value
            model.set_qp(args.qp)
            
            # Process through model
            outputs = model(clip)
            
            # Store results (only for valid frames)
            valid_frames = min(clip_length, end_idx - start_idx)
            
            # Store tracking results
            tracking_results = {
                'boxes': outputs['task_outputs']['boxes'][0, :valid_frames],  # [T, 4]
                'scores': outputs['task_outputs']['scores'][0, :valid_frames]  # [T]
            }
            all_outputs.append(tracking_results)
            
            # Store reconstructed frames
            reconstructed = outputs['reconstructed'][0, :, :valid_frames]  # [C, T, H, W]
            reconstructed = reconstructed.permute(1, 0, 2, 3)  # [T, C, H, W]
            all_reconstructed.append(reconstructed.cpu())
            
            # Calculate and accumulate bitrate
            bpp = calculate_bpp(
                outputs['feature_latents'].size(), 
                qp=args.qp,
                num_pixels=clip.size(2) * clip.size(3) * clip.size(4) * valid_frames
            )
            total_bpp += bpp
    
    # Concatenate results
    tracking_results = {
        'boxes': torch.cat([o['boxes'] for o in all_outputs], dim=0),
        'scores': torch.cat([o['scores'] for o in all_outputs], dim=0)
    }
    reconstructed_frames = torch.cat(all_reconstructed, dim=0)
    
    # Calculate average bitrate
    avg_bpp = total_bpp / len(all_outputs)
    
    return tracking_results, reconstructed_frames, avg_bpp


def save_tracking_results(tracking_results, sequence_name, output_dir):
    """Save tracking results in MOT Challenge format."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{sequence_name}.txt")
    
    boxes = tracking_results['boxes'].cpu().numpy()
    scores = tracking_results['scores'].cpu().numpy()
    
    with open(output_file, 'w') as f:
        for frame_idx in range(boxes.shape[0]):
            # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
            x, y, w, h = boxes[frame_idx]
            score = scores[frame_idx]
            
            # MOT format is 1-indexed for frames
            line = f"{frame_idx + 1},1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.6f},-1,-1,-1\n"
            f.write(line)
    
    print(f"Saved tracking results to {output_file}")


def visualize_results(frames, reconstructed_frames, tracking_results, sequence_name, output_dir):
    """Generate visualizations of tracking results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    orig_frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
    recon_frames = reconstructed_frames.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
    boxes = tracking_results['boxes'].cpu().numpy()  # [T, 4]
    
    # Set up video writer if requested
    if args.save_video:
        video_path = os.path.join(output_dir, f"{sequence_name}_tracking.mp4")
        height, width = orig_frames.shape[1], orig_frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width*2, height))
    
    for frame_idx in tqdm(range(len(frames)), desc="Generating visualizations"):
        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original frame
        axes[0].imshow(orig_frames[frame_idx])
        axes[0].set_title("Original Frame")
        axes[0].axis('off')
        
        # Reconstructed frame with tracking
        axes[1].imshow(recon_frames[frame_idx])
        x, y, w, h = boxes[frame_idx]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].set_title("Reconstructed with Tracking")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save as image if requested
        if args.save_frames:
            plt.savefig(os.path.join(output_dir, f"{sequence_name}_frame_{frame_idx:04d}.png"))
        
        # Add to video if requested
        if args.save_video:
            # Convert figure to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img)
        
        plt.close()
    
    if args.save_video:
        video_writer.release()
        print(f"Saved visualization video to {video_path}")


def main(args):
    """Main function."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build model
    model = build_model(args)
    model = model.to(args.device)
    model.eval()
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"No checkpoint found at '{args.checkpoint}'")
        return
    
    # Load MOT sequence
    sequence_path = os.path.join(args.data_dir, "train", args.sequence)
    if not os.path.exists(sequence_path):
        print(f"Sequence not found: {sequence_path}")
        # Try test directory
        sequence_path = os.path.join(args.data_dir, "test", args.sequence)
        if not os.path.exists(sequence_path):
            print(f"Sequence not found: {sequence_path}")
            return
    
    print(f"Loading sequence: {sequence_path}")
    frames = load_mot_sequence(
        sequence_path, 
        clip_length=args.clip_length,
        resolution=(args.resolution, args.resolution)
    )
    
    # Process video
    print(f"Processing {frames.shape[0]} frames...")
    tracking_results, reconstructed_frames, avg_bpp = process_video(model, frames, args)
    
    print(f"Average bitrate: {avg_bpp:.4f} bpp")
    
    # Save tracking results
    save_tracking_results(tracking_results, args.sequence, args.output_dir)
    
    # Visualize results if requested
    if args.visualize:
        print("Generating visualizations...")
        visualize_results(
            frames, 
            reconstructed_frames, 
            tracking_results, 
            args.sequence, 
            args.output_dir
        )
    
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    main(args)