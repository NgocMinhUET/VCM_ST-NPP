#!/usr/bin/env python
"""
Training script for the Differentiable Proxy Network.

This script trains the proxy network to approximate the rate-distortion behavior
of the HEVC codec in a differentiable manner.
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import cv2
import subprocess
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.proxy_network import ProxyNetwork
from models.stnpp import STNPP
from utils.codec_utils import HevcCodec
from utils.model_utils import save_model_with_version, get_latest_model

# Import utils.video_utils for its other functions
import utils.video_utils

# Import our MOT dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from mot_dataset import MOTImageSequenceDataset
    HAS_MOT_DATASET = True
except ImportError:
    print("Warning: mot_dataset.py not found, MOT dataset functionality will not be available.")
    print("If you need to use MOT dataset, make sure mot_dataset.py exists in the project root.")
    HAS_MOT_DATASET = False

    # Define a stub class to avoid errors
    class MOTImageSequenceDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("MOTImageSequenceDataset is not available. Please make sure mot_dataset.py exists.")


class VideoDataset(Dataset):
    """Dataset for loading video sequences."""
    
    def __init__(self, dataset_path, time_steps=16, transform=None, max_videos=None, frame_stride=4):
        """
        Initialize the VideoDataset.
        
        Args:
            dataset_path: Path to the directory containing video files
            time_steps: Number of frames in each sequence
            transform: Optional transform to apply to the frames
            max_videos: Maximum number of videos to load (for debugging)
            frame_stride: Stride for frame sampling
        """
        self.dataset_path = Path(dataset_path)
        self.time_steps = time_steps
        self.transform = transform
        self.frame_stride = frame_stride
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        self.video_files = []
        for ext in video_extensions:
            self.video_files.extend(list(self.dataset_path.glob(f'**/*{ext}')))
        
        # Check if any video files were found
        if len(self.video_files) == 0:
            raise ValueError(f"No video files found in {dataset_path}. Make sure the path exists and contains video files with extensions: {video_extensions}")
        
        # Limit the number of videos if specified
        if max_videos is not None:
            self.video_files = self.video_files[:max_videos]
        
        # Extract frames from videos and create sequences
        self.sequences = []
        for video_file in tqdm(self.video_files, desc="Loading videos"):
            self._extract_sequences(video_file)
        
        # Check if any sequences were created
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences could be created from the videos in {dataset_path}. Check that the videos have at least {time_steps} frames and are readable.")
    
    def _extract_sequences(self, video_file):
        """Extract frame sequences from a video file."""
        cap = cv2.VideoCapture(str(video_file))
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to a fixed size for consistency
            frame = cv2.resize(frame, (224, 224))
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
        
        cap.release()
        
        # Create sequences with stride
        if len(frames) >= self.time_steps:
            for i in range(0, len(frames) - self.time_steps + 1, self.frame_stride):
                sequence = frames[i:i + self.time_steps]
                self.sequences.append(sequence)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Convert to tensor
        sequence = np.array(sequence)
        sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Apply transform if specified
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence


def hevc_encode_decode(frames, qp, temp_dir=None):
    """
    Encode and decode frames using HEVC codec.
    
    Args:
        frames: Tensor of frames (T, C, H, W) in range [0, 1]
        qp: Quantization Parameter for HEVC
        temp_dir: Directory for temporary files
    
    Returns:
        Decoded frames tensor (T, C, H, W)
    """
    if temp_dir is None:
        temp_dir = Path('trained_models/proxy/temp_codec')
    
    # Ensure temp directory exists with proper permissions
    os.makedirs(temp_dir, exist_ok=True)
    os.chmod(temp_dir, 0o777)  # Give full permissions
    
    # Convert frames to numpy and back to [0, 255] range
    frames_np = frames.permute(0, 2, 3, 1).cpu().numpy() * 255.0
    frames_np = frames_np.astype(np.uint8)
    
    # Save frames as PNG files (lossless)
    frame_paths = []
    for i, frame in enumerate(frames_np):
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
    
    # Create YUV file from frames
    yuv_path = os.path.join(temp_dir, "temp.yuv")
    with open(yuv_path, 'wb') as yuv_file:
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path))
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
            yuv_file.write(yuv.tobytes())
    
    # Encode with HEVC
    height, width = frames_np.shape[1:3]
    encoded_path = os.path.join(temp_dir, "encoded.hevc")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}", "-i", str(yuv_path),
        "-c:v", "libx265", "-preset", "medium",
        "-x265-params", f"qp={qp}", str(encoded_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"FFmpeg encode error: {result.stderr.decode()}")
        raise RuntimeError("FFmpeg encoding failed")
    
    # Decode back to YUV
    decoded_yuv_path = os.path.join(temp_dir, "decoded.yuv")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(encoded_path),
        "-c:v", "rawvideo", "-pix_fmt", "yuv420p",
        str(decoded_yuv_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"FFmpeg decode error: {result.stderr.decode()}")
        raise RuntimeError("FFmpeg decoding failed")
    
    # Read decoded YUV back to tensors
    frame_size = width * height * 3 // 2  # YUV420 uses 1.5 bytes per pixel
    decoded_frames = []
    with open(decoded_yuv_path, 'rb') as f:
        for _ in range(len(frames_np)):
            yuv_bytes = f.read(frame_size)
            if not yuv_bytes:
                break
            
            # Create a YUV frame and convert to RGB
            yuv_np = np.frombuffer(yuv_bytes, dtype=np.uint8)
            yuv_np = yuv_np.reshape((height * 3 // 2, width))
            rgb = cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_I420)
            
            # Convert to tensor and normalize to [0, 1]
            rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            decoded_frames.append(rgb_tensor)
    
    # Stack frames
    decoded_frames = torch.stack(decoded_frames, dim=0)
    
    # Clean up
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.remove(yuv_path)
    os.remove(encoded_path)
    os.remove(decoded_yuv_path)
    
    return decoded_frames


def parse_args():
    parser = argparse.ArgumentParser(description="Train Proxy Network")
    
    # Dataset parameters
    parser.add_argument("--dataset", "--dataset_path", type=str, required=True,
                        help="Path to the video dataset or MOT16 dataset directory")
    parser.add_argument("--time_steps", type=int, default=16,
                        help="Number of frames in each sequence")
    parser.add_argument("--frame_stride", type=int, default=4,
                        help="Stride for frame sampling")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Maximum number of videos to load (for debugging)")
    parser.add_argument("--use_mot_dataset", action="store_true",
                        help="Use MOT16 dataset format instead of video files")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use when using MOT16 (train or test)")
    
    # Model parameters
    parser.add_argument("--lambda_value", type=float, default=0.1,
                        help="Weight for the distortion term in the proxy loss")
    parser.add_argument("--use_ssim", action="store_true",
                        help="Use SSIM instead of MSE for distortion measurement")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training")
    parser.add_argument("--lr", "--learning_rate", type=float, default=0.0001,
                        help="Learning rate for optimizer")
    parser.add_argument("--codec_interval", type=int, default=10,
                        help="Interval for running the actual codec (slower)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="trained_models",
                        help="Directory to save the trained model")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Epoch interval for saving checkpoints")
    
    # HEVC parameters
    parser.add_argument("--qp", type=int, default=27,
                        help="Quantization Parameter for HEVC codec")
    parser.add_argument("--qp_values", type=str, default=None,
                        help="Comma-separated list of QP values to train on (e.g., '22,27,32,37')")
    
    # Additional model parameters
    parser.add_argument("--backbone", type=str, default="resnet34",
                        help="Backbone network for ST-NPP (resnet50, resnet34, efficientnet_b4)")
    parser.add_argument("--temporal_model", type=str, default="3dcnn",
                        help="Temporal model for ST-NPP (3dcnn, convlstm)")
    parser.add_argument("--fusion_type", type=str, default="concatenation",
                        help="Fusion type for ST-NPP (concatenation, attention)")
    parser.add_argument("--feature_channels", type=int, default=128,
                        help="Number of feature channels for ST-NPP and proxy network")
    parser.add_argument("--base_channels", type=int, default=64,
                        help="Base channels for the proxy network")
    parser.add_argument("--latent_channels", type=int, default=32,
                        help="Latent channels for the proxy network")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision training")
    
    # TensorBoard parameters
    parser.add_argument("--log_dir", type=str, default="logs/proxy",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval (batches)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume training")
    
    # Debug parameters
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Process QP values if provided
    if args.qp_values:
        try:
            args.qp_values = [int(qp) for qp in args.qp_values.split(',')]
            print(f"Training with multiple QP values: {args.qp_values}")
        except ValueError:
            print(f"Error parsing QP values: {args.qp_values}")
            print("Using default QP value instead.")
            args.qp_values = None
    
    return args


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(args):
    """Main training function."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print key parameters for debugging
    print(f"Training parameters:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    if args.qp_values:
        print(f"  QP values: {args.qp_values}")
    else:
        print(f"  QP: {args.qp}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create temporary directory for codec operations
    temp_dir = os.path.join(args.output_dir, "temp_codec")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set up TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'proxy_training_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize ST-NPP model (for feature extraction)
    print("Initializing ST-NPP model...")
    try:
        stnpp = STNPP(
            input_channels=3,
            output_channels=args.feature_channels,
            spatial_backbone=args.backbone,
            temporal_model=args.temporal_model,
            fusion_type=args.fusion_type,
            pretrained=True
        ).to(device)
        stnpp.eval()  # Set to evaluation mode for feature extraction
    except Exception as e:
        print(f"Error initializing ST-NPP model: {e}")
        raise
        
    # Initialize proxy network
    print("Initializing Proxy Network...")
    try:
        proxy_net = ProxyNetwork(
            input_channels=args.feature_channels,
            base_channels=args.base_channels,
            latent_channels=args.latent_channels
        ).to(device)
    except Exception as e:
        print(f"Error initializing Proxy Network: {e}")
        raise
        
    # Initialize optimizer
    optimizer = optim.Adam(proxy_net.parameters(), lr=args.lr)
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if args.amp else None
    
    # Load dataset
    print("Loading dataset...")
    try:
        # Check if dataset path exists
        if not os.path.exists(args.dataset):
            print(f"ERROR: Dataset path does not exist: {args.dataset}")
            print("Please provide a valid path to the dataset.")
            if args.use_mot_dataset:
                print("\nFor MOT16 dataset, the path should contain the following structure:")
                print("  MOT16/")
                print("  ├── train/")
                print("  │   ├── MOT16-02/")
                print("  │   ├── MOT16-04/")
                print("  │   └── ...")
                print("  └── test/")
                print("      ├── MOT16-01/")
                print("      ├── MOT16-03/")
                print("      └── ...")
            sys.exit(1)
            
        if args.use_mot_dataset:
            # Use MOT16 dataset format
            if not HAS_MOT_DATASET:
                print("ERROR: MOT dataset functionality is not available.")
                print("Please make sure mot_dataset.py exists in the project root.")
                sys.exit(1)
                
            dataset = MOTImageSequenceDataset(
                dataset_path=args.dataset,
                time_steps=args.time_steps,
                split=args.split,
                frame_stride=args.frame_stride
            )
        else:
            # Use video files
            dataset = VideoDataset(
                dataset_path=args.dataset,
                time_steps=args.time_steps,
                transform=None,
                max_videos=args.max_videos,
                frame_stride=args.frame_stride
            )
        
        print(f"Dataset loaded with {len(dataset)} sequences")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        proxy_net, metadata = get_latest_model(proxy_net, args.resume, device, optimizer)
        if 'epoch' in metadata:
            start_epoch = metadata['epoch']
        if 'metrics' in metadata and 'val_loss' in metadata['metrics']:
            best_val_loss = metadata['metrics']['val_loss']
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        total_loss = 0.0
        total_rate = 0.0
        total_distortion = 0.0
        
        # Training phase
        proxy_net.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = batch.to(device)  # (B, T, C, H, W)
            batch = batch.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            
            # Extract features using ST-NPP
            with torch.no_grad():
                preprocessed_features = stnpp(batch)
            
            # Get ground truth by running HEVC codec on a sample from batch
            if batch_idx % args.codec_interval == 0:
                with torch.no_grad():
                    # Use only the first item in batch to reduce processing time
                    sample_frames = batch[0].permute(1, 0, 2, 3)  # (T, C, H, W)
                    
                    # Select QP value
                    if args.qp_values:
                        # Randomly select a QP value from the list
                        qp = random.choice(args.qp_values)
                        if args.verbose:
                            print(f"Using QP value: {qp} for batch {batch_idx}")
                    else:
                        qp = args.qp
                    
                    decoded_frames = hevc_encode_decode(
                        sample_frames, qp, temp_dir
                    ).to(device)
                    
                    # Extract features from decoded frames
                    decoded_frames = decoded_frames.unsqueeze(0)  # (1, T, C, H, W)
                    decoded_frames = decoded_frames.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
                    hevc_features = stnpp(decoded_frames)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass through proxy network with mixed precision
            if scaler is not None:
                with autocast():
                    reconstructed, latent = proxy_net(preprocessed_features)
                    rd_loss, rate, distortion = proxy_net.calculate_rd_loss(
                        preprocessed_features, reconstructed, latent,
                        lambda_value=args.lambda_value,
                        use_ssim=args.use_ssim
                    )
                
                # Backward pass and optimizer step with gradient scaling
                scaler.scale(rd_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstructed, latent = proxy_net(preprocessed_features)
                rd_loss, rate, distortion = proxy_net.calculate_rd_loss(
                    preprocessed_features, reconstructed, latent,
                    lambda_value=args.lambda_value,
                    use_ssim=args.use_ssim
                )
                
                # Backward pass and optimizer step
                rd_loss.backward()
                optimizer.step()
            
            # Update statistics
            total_loss += rd_loss.item()
            total_rate += rate.item()
            total_distortion += distortion.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{rd_loss.item():.4f}",
                'rate': f"{rate.item():.4f}",
                'distortion': f"{distortion.item():.4f}"
            })
            
            # Log to TensorBoard more frequently if verbose
            if args.verbose and batch_idx % (args.log_interval // 4) == 0:
                step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('batch/loss', rd_loss.item(), step)
                writer.add_scalar('batch/rate', rate.item(), step)
                writer.add_scalar('batch/distortion', distortion.item(), step)
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        avg_rate = total_rate / len(dataloader)
        avg_distortion = total_distortion / len(dataloader)
        
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Loss: {avg_loss:.4f}, Rate: {avg_rate:.4f}, "
              f"Distortion: {avg_distortion:.4f}, "
              f"Time: {elapsed_time:.2f}s")
        
        # Log to TensorBoard
        global_step = epoch * len(dataloader)
        writer.add_scalar('train/loss', avg_loss, global_step)
        writer.add_scalar('train/rate', avg_rate, global_step)
        writer.add_scalar('train/distortion', avg_distortion, global_step)
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        
        # Save checkpoint if this is the best model so far
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            checkpoint_path = save_model_with_version(
                proxy_net,
                args.output_dir,
                "proxy_network_best",
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": best_val_loss},
                version=timestamp
            )
            print(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = save_model_with_version(
                proxy_net,
                args.output_dir,
                f"proxy_network_epoch_{epoch+1}",
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": avg_loss},
                version=timestamp
            )
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = save_model_with_version(
        proxy_net,
        args.output_dir,
        "proxy_network_final",
        optimizer=optimizer,
        epoch=args.epochs,
        metrics={"val_loss": avg_loss},
        version=timestamp
    )
    print(f"Saved final model to {final_path}")
    
    # Clean up
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    
    print("Training completed!")
    writer.close()
    
    return {
        "best_model": checkpoint_path if 'checkpoint_path' in locals() else None,
        "final_model": final_path,
        "best_val_loss": best_val_loss,
        "final_val_loss": avg_loss,
        "log_dir": log_dir
    }


if __name__ == "__main__":
    args = parse_args()
    train(args) 