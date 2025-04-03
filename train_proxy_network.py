import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from src.models.proxy_network import ProxyNetwork
from utils.codec_utils import run_hevc_pipeline
from utils.video_utils import load_video, preprocess_frames, create_sliding_windows

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Differentiable Proxy Network")
    
    parser.add_argument("--dataset_path", type=str, default="datasets/MOTChallenge/MOT16",
                        help="Path to the directory containing video files")
    parser.add_argument("--output_dir", type=str, default="models/proxy_network",
                        help="Directory to save trained models")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--qp", type=int, default=27,
                        help="Quantization Parameter for HEVC codec")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of time steps for temporal processing")
    parser.add_argument("--use_ssim", action="store_true",
                        help="Use SSIM instead of MSE for distortion measurement")
    parser.add_argument("--lambda_value", type=float, default=0.1,
                        help="Weight for the distortion term in the proxy loss")
    
    return parser.parse_args()

def load_dataset(dataset_path, time_steps, max_videos=None):
    """
    Load videos from a directory and prepare them for training.
    
    Args:
        dataset_path: Path to the directory containing video files
        time_steps: Number of time steps for temporal processing
        max_videos: Maximum number of videos to load (None for all)
        
    Returns:
        List of video segments for training
    """
    print(f"Loading videos from {dataset_path}...")
    
    # Get list of video files
    video_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    
    if max_videos is not None:
        video_files = video_files[:max_videos]
    
    print(f"Found {len(video_files)} video files.")
    
    # Load and preprocess videos
    all_segments = []
    
    for i, video_path in enumerate(video_files):
        print(f"Processing video {i+1}/{len(video_files)}: {video_path}")
        
        # Load video frames
        frames = load_video(video_path)
        
        # Skip if video doesn't have enough frames
        if len(frames) < time_steps:
            print(f"Skipping video with only {len(frames)} frames (need {time_steps})")
            continue
        
        # Preprocess frames
        processed_frames = preprocess_frames(frames)
        
        # Create sliding windows
        video_segments = create_sliding_windows(processed_frames, window_size=time_steps)
        
        all_segments.extend(video_segments)
    
    print(f"Created {len(all_segments)} video segments.")
    
    return all_segments

def generate_hevc_targets(video_segments, qp=27):
    """
    Generate HEVC-encoded versions of the video segments for training.
    
    Args:
        video_segments: List of video segments
        qp: Quantization Parameter for HEVC codec
        
    Returns:
        List of HEVC-encoded video segments
    """
    print(f"Generating HEVC targets with QP={qp}...")
    
    hevc_segments = []
    
    for i, segment in enumerate(video_segments):
        if i % 10 == 0:  # Print progress every 10 segments
            print(f"Processing segment {i+1}/{len(video_segments)}")
        
        # Run HEVC encoding and decoding
        hevc_frames = run_hevc_pipeline(segment, qp=qp)
        
        hevc_segments.append(hevc_frames)
    
    return hevc_segments

def prepare_training_data(video_segments, hevc_segments):
    """
    Prepare training data for the proxy network.
    
    Args:
        video_segments: List of original video segments
        hevc_segments: List of HEVC-encoded video segments
        
    Returns:
        Tuple of (input_data, target_data)
    """
    # Convert lists to numpy arrays
    input_data = np.array(video_segments)
    target_data = np.array(hevc_segments)
    
    return input_data, target_data

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    video_segments = load_dataset(args.dataset_path, args.time_steps)
    
    # Generate HEVC targets
    hevc_segments = generate_hevc_targets(video_segments, qp=args.qp)
    
    # Prepare training data
    input_data, target_data = prepare_training_data(video_segments, hevc_segments)
    
    # Initialize proxy network
    input_shape = input_data.shape[1:]  # (Time, H, W, C)
    proxy_network = ProxyNetwork(input_shape=input_shape)
    encoder, decoder, autoencoder = proxy_network.build()
    
    # Create custom loss function
    custom_loss = ProxyNetwork.create_custom_loss(
        encoder=encoder,
        lambda_value=args.lambda_value,
        use_ssim=args.use_ssim
    )
    
    # Compile the model
    autoencoder.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss=custom_loss
    )
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(args.output_dir, "proxy_network_best.h5"),
            save_best_only=True,
            monitor="val_loss"
        ),
        TensorBoard(
            log_dir=os.path.join(args.output_dir, "logs"),
            histogram_freq=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    print("Training proxy network...")
    autoencoder.fit(
        input_data,
        target_data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        callbacks=callbacks
    )
    
    # Save the final model
    autoencoder.save(os.path.join(args.output_dir, "proxy_network_final.h5"))
    encoder.save(os.path.join(args.output_dir, "proxy_encoder_final.h5"))
    decoder.save(os.path.join(args.output_dir, "proxy_decoder_final.h5"))
    
    print(f"Training complete. Models saved to {args.output_dir}")

if __name__ == "__main__":
    main() 