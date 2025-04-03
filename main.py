import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path

# Import our modules
from src.pipeline import VCMPreprocessingPipeline
from utils.video_utils import load_video, preprocess_frames, create_sliding_windows
from utils.codec_utils import run_hevc_pipeline, calculate_bpp

def parse_args():
    parser = argparse.ArgumentParser(description="Video Compression for Machine Vision (VCM) - Preprocessing Pipeline")
    
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input video file or directory of videos")
    parser.add_argument("--output_path", type=str, default="output_features",
                        help="Path to save the output features")
    parser.add_argument("--spatial_backbone", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet-b4"],
                        help="Backbone for the spatial branch")
    parser.add_argument("--temporal_model", type=str, default="3dcnn",
                        choices=["3dcnn", "convlstm"],
                        help="Model type for the temporal branch")
    parser.add_argument("--fusion_type", type=str, default="concatenation",
                        choices=["concatenation", "attention"],
                        help="Fusion method for combining spatial and temporal features")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of time steps for temporal processing")
    parser.add_argument("--qp", type=int, default=23,
                        help="Quantization Parameter value")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize input and output video frames")
    parser.add_argument("--save_compressed", action="store_true",
                        help="Save compressed videos alongside features")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU ID to use (e.g., '0' or '0,1')")
    
    return parser.parse_args()

def process_single_video(video_path, pipeline, args):
    """Process a single video through the preprocessing pipeline."""
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    feature_output_path = os.path.join(output_path, f"{video_name}_features.npy")
    
    print(f"Processing video: {video_path}")
    
    try:
        # Load and preprocess video
        print("Loading video...")
        frames = load_video(video_path)
        if len(frames) < args.time_steps:
            print(f"Warning: Video has only {len(frames)} frames, but {args.time_steps} are required.")
            return None
            
        print(f"Processing {len(frames)} frames...")
        processed_frames = preprocess_frames(frames)
        
        # Create sliding windows
        video_segments = create_sliding_windows(processed_frames, window_size=args.time_steps)
        print(f"Created {len(video_segments)} video segments")
        
        # Preprocess video segments
        all_features = []
        
        # Use tqdm for progress tracking
        for i, segment in enumerate(tqdm(video_segments, desc="Processing segments")):
            # Process through the pipeline
            features = pipeline.preprocess_video(
                np.expand_dims(segment, 0),  # Add batch dimension
                qp=args.qp
            )
            all_features.append(features[0])  # Remove batch dimension
            
            # Visualize if requested
            if args.visualize and i == 0:
                visualize_results(segment, features[0], video_name)
        
        # Save the output features
        all_features = np.array(all_features)
        np.save(feature_output_path, all_features)
        
        # Create a metadata file with processing information
        metadata = {
            "video_name": video_name,
            "original_frames": len(frames),
            "segments": len(video_segments),
            "spatial_backbone": args.spatial_backbone,
            "temporal_model": args.temporal_model,
            "fusion_type": args.fusion_type,
            "time_steps": args.time_steps,
            "qp": args.qp,
            "feature_shape": all_features.shape
        }
        
        with open(os.path.join(output_path, f"{video_name}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save compressed video if requested
        if args.save_compressed:
            print("Compressing video with HEVC...")
            compressed_dir = os.path.join(output_path, "compressed")
            os.makedirs(compressed_dir, exist_ok=True)
            
            # Run HEVC encoding/decoding on original video
            decoded_frames = run_hevc_pipeline(processed_frames, qp=args.qp)
            
            # Save sample frames
            save_sample_frames(decoded_frames, os.path.join(compressed_dir, f"{video_name}_compressed"), max_frames=5)
        
        print(f"Processing complete for {video_path}")
        print(f"Features saved to {feature_output_path}")
        
        return metadata
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None

def visualize_results(input_segment, output_features, video_name):
    """Visualize input and output of the pipeline."""
    
    # Create output directory for visualizations
    viz_dir = os.path.join("visualizations", video_name)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot input frames (sample)
    plt.figure(figsize=(15, 8))
    for i in range(min(4, len(input_segment))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(input_segment[i])
        plt.title(f"Input Frame {i}")
        plt.axis('off')
    
    # Plot feature maps (sample channels)
    feature_channels = min(4, output_features.shape[-1])
    for i in range(feature_channels):
        plt.subplot(2, 4, i + 5)
        # Take middle frame if 3D features
        feature_map = output_features[output_features.shape[0]//2] if len(output_features.shape) > 3 else output_features
        plt.imshow(feature_map[:, :, i], cmap='viridis')
        plt.title(f"Feature Channel {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "pipeline_visualization.png"))
    plt.close()
    
    print(f"Visualization saved to {viz_dir}")

def save_sample_frames(frames, output_prefix, max_frames=5):
    """Save sample frames as images."""
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
    # Determine frames to save (evenly spaced)
    total_frames = len(frames)
    if total_frames <= max_frames:
        indices = range(total_frames)
    else:
        indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    
    # Save frames
    for i, idx in enumerate(indices):
        plt.figure(figsize=(8, 6))
        plt.imshow(frames[idx])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_frame_{i}.png")
        plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    # Print available GPU(s)
    if physical_devices:
        print(f"Using GPU(s): {args.gpu}")
        for i, device in enumerate(physical_devices):
            print(f"  {i}: {device.name}")
    else:
        print("No GPU found. Using CPU.")
    
    # Initialize pipeline
    print("Initializing preprocessing pipeline...")
    pipeline = VCMPreprocessingPipeline(
        input_shape=(None, 224, 224, 3),
        time_steps=args.time_steps,
        spatial_backbone=args.spatial_backbone,
        temporal_model=args.temporal_model,
        fusion_type=args.fusion_type
    )
    
    # Build the pipeline models
    print("Building pipeline models...")
    st_npp_model, qal_model, combined_model = pipeline.build()
    
    # Process videos
    results = []
    
    # Determine if input is a single video or a directory
    if os.path.isdir(args.video_path):
        # Process all videos in the directory
        video_files = []
        for root, _, files in os.walk(args.video_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(root, file))
        
        print(f"Found {len(video_files)} video files to process.")
        
        # Process each video
        for video_file in video_files:
            result = process_single_video(video_file, pipeline, args)
            if result:
                results.append(result)
    else:
        # Process single video
        result = process_single_video(args.video_path, pipeline, args)
        if result:
            results.append(result)
    
    # Save summary of all processed videos
    if results:
        with open(os.path.join(args.output_path, "processing_summary.json"), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nProcessing summary:")
        print(f"- Total videos processed: {len(results)}")
        print(f"- Output directory: {os.path.abspath(args.output_path)}")
    else:
        print("\nNo videos were successfully processed.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds") 