#!/usr/bin/env python
"""
Joint fine-tuning script for ST-NPP and QAL models.

This script implements joint training of previously trained ST-NPP and QAL models
for end-to-end optimization. It sets up TensorBoard logging and implements
a model versioning system.
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
import cv2
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stnpp import STNPP
from models.qal import QAL
from models.proxy_network import ProxyNetwork
from utils.model_utils import save_model_with_version, load_model_with_version
from utils.codec_utils import HevcCodec

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
    """Dataset for loading video sequences or image sequences."""
    
    def __init__(self, dataset_path, time_steps=16, transform=None, max_videos=None, frame_stride=4):
        """
        Initialize the VideoDataset.
        
        Args:
            dataset_path: Path to the directory containing video files or image sequences
            time_steps: Number of frames in each sequence
            transform: Optional transform to apply to the frames
            max_videos: Maximum number of videos to load (for debugging)
            frame_stride: Stride for frame sampling
        """
        self.dataset_path = Path(dataset_path)
        self.time_steps = time_steps
        self.transform = transform
        self.frame_stride = frame_stride
        
        # Check if this is an image sequence dataset (like MOT16)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        is_image_sequence = False
        
        # Check for image sequences first
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.dataset_path.glob(f'**/*{ext}')))
        
        if len(image_files) > 0:
            print(f"Found {len(image_files)} image files, treating as image sequence dataset")
            is_image_sequence = True
            # Extract sequences from image directories
            self.sequences = self._extract_image_sequences(image_files)
        else:
            # If no image files found, look for video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            self.video_files = []
            for ext in video_extensions:
                self.video_files.extend(list(self.dataset_path.glob(f'**/*{ext}')))
            
            # Limit the number of videos if specified
            if max_videos is not None:
                self.video_files = self.video_files[:max_videos]
            
            # Check if any video files were found
            if len(self.video_files) == 0:
                raise ValueError(f"No video files or image sequences found in {dataset_path}. "
                                f"Make sure the path exists and contains video files with extensions: {video_extensions} "
                                f"or image files with extensions: {image_extensions}")
            
            # Extract frames from videos and create sequences
            self.sequences = []
            for video_file in tqdm(self.video_files, desc="Loading videos"):
                self._extract_video_sequences(video_file)
        
        # Check if any sequences were created
        if len(self.sequences) == 0:
            raise ValueError(f"No valid sequences could be created from the data in {dataset_path}. "
                            f"Check that the videos/images have at least {time_steps} frames and are readable.")
        
        print(f"Created {len(self.sequences)} sequences")
    
    def _extract_image_sequences(self, image_files):
        """Extract frame sequences from image files."""
        sequences = []
        current_sequence = []
        prev_dir = None
        
        # Sort image files to ensure proper sequence
        image_files = sorted(image_files)
        
        for img_file in tqdm(image_files, desc="Processing image files"):
            current_dir = img_file.parent
            
            # Start new sequence if directory changes
            if prev_dir is not None and current_dir != prev_dir:
                if len(current_sequence) >= self.time_steps:
                    sequences.extend(self._create_subsequences(current_sequence))
                current_sequence = []
            
            # Load and preprocess image
            img = cv2.imread(str(img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float32) / 255.0
                current_sequence.append(img)
            
            prev_dir = current_dir
            
            # Process sequence if it's long enough
            if len(current_sequence) >= self.time_steps * 2:
                sequences.extend(self._create_subsequences(current_sequence))
                current_sequence = current_sequence[-self.time_steps:]  # Keep last sequence for overlap
        
        # Process remaining sequence
        if len(current_sequence) >= self.time_steps:
            sequences.extend(self._create_subsequences(current_sequence))
        
        return sequences
    
    def _create_subsequences(self, sequence):
        """Create subsequences of length time_steps with stride."""
        subsequences = []
        for i in range(0, len(sequence) - self.time_steps + 1, self.frame_stride):
            subsequence = sequence[i:i + self.time_steps]
            subsequences.append(subsequence)
        return subsequences
    
    def _extract_video_sequences(self, video_file):
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
        
        return {'frames': sequence}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Joint fine-tuning of ST-NPP and QAL models')
    
    # Model paths
    parser.add_argument('--stnpp_model', type=str, required=True,
                        help='Path to pretrained ST-NPP model')
    parser.add_argument('--qal_model', type=str, required=True,
                        help='Path to pretrained QAL model')
    parser.add_argument('--proxy_model', type=str, required=True,
                        help='Path to trained Proxy Network model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to video dataset or MOT16 dataset directory')
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='Path to validation dataset (if None, uses a portion of training data)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--use_mot_dataset', action='store_true',
                        help='Use MOT16 dataset format instead of video files')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to use when using MOT16 (train or test)')
    parser.add_argument('--time_steps', type=int, default=16,
                        help='Number of frames in each sequence')
    parser.add_argument('--frame_stride', type=int, default=4,
                        help='Stride for frame sampling')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to load (for debugging)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--qp_values', type=str, default='22,27,32,37',
                        help='Comma-separated list of QP values to train with')
    parser.add_argument('--lambda_distortion', type=float, default=1.0,
                        help='Weight for distortion loss component')
    parser.add_argument('--lambda_rate', type=float, default=0.1,
                        help='Weight for rate loss component')
    parser.add_argument('--lambda_perception', type=float, default=0.01,
                        help='Weight for perceptual loss component')
    parser.add_argument('--use_real_codec', action='store_true',
                        help='Use real HEVC codec instead of proxy network')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./trained_models/joint',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs/joint',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save model every N epochs')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda/cpu)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class JointRDLoss(nn.Module):
    """
    Joint Rate-Distortion-Perception Loss for training ST-NPP + QAL.
    
    This loss combines:
    1. Distortion: Reconstruction quality (MSE or PSNR)
    2. Rate: Bitrate estimation from proxy network or real codec
    3. Perception: Feature preservation for downstream tasks
    """
    def __init__(self, lambda_distortion=1.0, lambda_rate=0.1, lambda_perception=0.01):
        super(JointRDLoss, self).__init__()
        self.lambda_distortion = lambda_distortion
        self.lambda_rate = lambda_rate
        self.lambda_perception = lambda_perception
        self.mse_loss = nn.MSELoss()
        
    def forward(self, original, preprocessed, estimated_rate, perceptual_loss=None):
        # Distortion loss (MSE between original and preprocessed)
        distortion_loss = self.mse_loss(original, preprocessed)
        
        # Rate loss (directly from rate estimator)
        # Handle both tensor and scalar rate values
        if isinstance(estimated_rate, torch.Tensor):
            rate_loss = estimated_rate.mean()
        else:
            rate_loss = estimated_rate
        
        # Total loss
        total_loss = (self.lambda_distortion * distortion_loss + 
                      self.lambda_rate * rate_loss)
        
        # Add perceptual loss if provided
        if perceptual_loss is not None:
            total_loss += self.lambda_perception * perceptual_loss
            
        return total_loss, {
            'distortion_loss': distortion_loss.item(),
            'rate_loss': rate_loss.item() if isinstance(rate_loss, torch.Tensor) else rate_loss,
            'perceptual_loss': perceptual_loss.item() if perceptual_loss is not None else 0,
            'total_loss': total_loss.item()
        }


def train_joint(args):
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'joint_training_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load ST-NPP model
    print(f"Loading ST-NPP model from {args.stnpp_model}")
    stnpp_model = STNPP(
        input_channels=3,
        output_channels=128,  # Default value, will be overridden by loaded model
        spatial_backbone='resnet50',  # Default value, will be overridden by loaded model
        temporal_model='3dcnn',  # Default value, will be overridden by loaded model
        fusion_type='concatenation',  # Default value, will be overridden by loaded model
        pretrained=False  # We're loading weights, so no need for pretrained
    )
    stnpp_model, _ = load_model_with_version(stnpp_model, args.stnpp_model, device)
    
    # Load QAL model
    print(f"Loading QAL model from {args.qal_model}")
    qal_model = QAL(
        feature_channels=128,  # Default value, will be overridden by loaded model
        hidden_dim=64  # Default value, will be overridden by loaded model
    )
    qal_model, _ = load_model_with_version(qal_model, args.qal_model, device)
    
    # Load Proxy Network or initialize codec
    if args.use_real_codec:
        print("Using real HEVC codec")
        codec = HevcCodec()
    else:
        print(f"Loading Proxy Network model from {args.proxy_model}")
        try:
            proxy_model = ProxyNetwork(
                input_channels=128,
                base_channels=64,
                latent_channels=32
            )
            proxy_model, _ = load_model_with_version(proxy_model, args.proxy_model, device)
            proxy_model.eval()
        except Exception as e:
            print(f"Error loading Proxy Network model: {e}")
            raise
    
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
                
            train_dataset = MOTImageSequenceDataset(
                dataset_path=args.dataset,
                time_steps=args.time_steps,
                split=args.split,
                frame_stride=args.frame_stride
            )
        else:
            # Use video files
            train_dataset = VideoDataset(
                dataset_path=args.dataset,
                time_steps=args.time_steps,
                transform=None,
                max_videos=args.max_videos,
                frame_stride=args.frame_stride
            )
        
        print(f"Dataset loaded with {len(train_dataset)} sequences")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Create train dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Handle validation dataset
    if args.val_dataset:
        try:
            if args.use_mot_dataset:
                val_dataset = MOTImageSequenceDataset(
                    dataset_path=args.val_dataset,
                    time_steps=args.time_steps,
                    split='test',  # Use test split for validation
                    frame_stride=args.frame_stride
                )
            else:
                val_dataset = VideoDataset(
                    dataset_path=args.val_dataset,
                    time_steps=args.time_steps,
                    transform=None,
                    max_videos=args.max_videos,
                    frame_stride=args.frame_stride
                )
        except Exception as e:
            print(f"Error loading validation dataset: {e}")
            print("Using a portion of training data for validation instead.")
            # Split training dataset for validation
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
    else:
        # Use a portion of training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Parse QP values
    qp_values = [int(qp) for qp in args.qp_values.split(',')]
    
    # Set up optimizer
    # We're training both ST-NPP and QAL jointly
    joint_params = list(stnpp_model.parameters()) + list(qal_model.parameters())
    optimizer = optim.Adam(joint_params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Set up loss function
    criterion = JointRDLoss(
        lambda_distortion=args.lambda_distortion,
        lambda_rate=args.lambda_rate,
        lambda_perception=args.lambda_perception
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        stnpp_model.train()
        qal_model.train()
        train_losses = []
        
        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)
            batch_size = frames.size(0)
            
            # Randomly select QP for this batch
            qp = random.choice(qp_values)
            
            # Forward pass through ST-NPP
            preprocessed = stnpp_model(frames)
            
            # Convert QP to tensor with proper shape for QAL model
            qp_tensor = torch.full((frames.size(0),), qp, dtype=torch.float32, device=device)
            
            # Forward pass through QAL
            qal_output = qal_model(preprocessed, qp_tensor)
            
            # Estimate rate (using proxy network or real codec)
            if args.use_real_codec:
                # This would be a placeholder for using a real codec
                # In practice, you would need to implement a differentiable
                # approximation or use straight-through estimator
                raise NotImplementedError("Real codec training not implemented yet")
            else:
                # Use proxy network for rate estimation
                # ProxyNetwork returns (reconstructed, latent)
                reconstructed_proxy, latent = proxy_model(qal_output)
                
                # Calculate the rate from the latent representation
                estimated_rate = proxy_model.calculate_bitrate(latent)
                
                # Use the reconstructed output for perceptual evaluation if needed
                # For now, we only use the bitrate estimate
            
            # Calculate loss
            loss, loss_components = criterion(frames, qal_output, estimated_rate)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track training statistics
            train_losses.append(loss.item())
            
            # Log to TensorBoard every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
                global_step = epoch * len(train_loader) + batch_idx
                
                # Log loss components
                for name, value in loss_components.items():
                    writer.add_scalar(f'train/{name}', value, global_step)
                
                # Log learning rate
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                # Log sample images (original vs preprocessed)
                if batch_idx % 50 == 0:
                    # Only log the first image of the batch
                    writer.add_image('train/original', frames[0].cpu(), global_step)
                    writer.add_image('train/preprocessed', preprocessed[0].cpu(), global_step)
                    writer.add_image('train/qal_output', qal_output[0].cpu(), global_step)
        
        train_time = time.time() - start_time
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"Training Loss: {avg_train_loss:.6f}, Time: {train_time:.2f}s")
        
        # Validation phase
        stnpp_model.eval()
        qal_model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                frames = batch['frames'].to(device)
                
                # We'll validate across all QP values
                qp_results = []
                for qp in qp_values:
                    # Forward pass through ST-NPP
                    preprocessed = stnpp_model(frames)
                    
                    # Convert QP to tensor with proper shape for QAL model
                    qp_tensor = torch.full((frames.size(0),), qp, dtype=torch.float32, device=device)
                    
                    # Forward pass through QAL
                    qal_output = qal_model(preprocessed, qp_tensor)
                    
                    # Estimate rate
                    if args.use_real_codec:
                        raise NotImplementedError("Real codec validation not implemented yet")
                    else:
                        # ProxyNetwork returns (reconstructed, latent)
                        reconstructed_proxy, latent = proxy_model(qal_output)
                        
                        # Calculate the rate from the latent representation
                        estimated_rate = proxy_model.calculate_bitrate(latent)
                    
                    # Calculate loss
                    loss, _ = criterion(frames, qal_output, estimated_rate)
                    qp_results.append(loss.item())
                
                # Average loss across QP values
                val_losses.append(sum(qp_results) / len(qp_results))
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # Log validation loss to TensorBoard
        writer.add_scalar('validation/avg_loss', avg_val_loss, epoch)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save model periodically
        if (epoch + 1) % args.save_interval == 0:
            # Save ST-NPP model
            stnpp_path = save_model_with_version(
                stnpp_model,
                args.output_dir,
                "stnpp_joint",
                optimizer=None,  # We don't save optimizer state for intermediate saves
                epoch=epoch + 1,
                metrics={"val_loss": avg_val_loss},
                version=f"{timestamp}_e{epoch+1}"
            )
            
            # Save QAL model
            qal_path = save_model_with_version(
                qal_model,
                args.output_dir,
                "qal_joint",
                optimizer=None,
                epoch=epoch + 1,
                metrics={"val_loss": avg_val_loss},
                version=f"{timestamp}_e{epoch+1}"
            )
            
            print(f"Saved models to {stnpp_path} and {qal_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save best ST-NPP model
            best_stnpp_path = save_model_with_version(
                stnpp_model,
                args.output_dir,
                "stnpp_joint_best",
                optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": best_val_loss},
                version=timestamp
            )
            
            # Save best QAL model
            best_qal_path = save_model_with_version(
                qal_model,
                args.output_dir,
                "qal_joint_best",
                optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": best_val_loss},
                version=timestamp
            )
            
            print(f"New best model saved to {best_stnpp_path} and {best_qal_path}")
    
    # Final save at the end of training
    final_stnpp_path = save_model_with_version(
        stnpp_model,
        args.output_dir,
        "stnpp_joint_final",
        optimizer,
        epoch=args.epochs,
        metrics={"val_loss": avg_val_loss},
        version=timestamp
    )
    
    final_qal_path = save_model_with_version(
        qal_model,
        args.output_dir,
        "qal_joint_final",
        optimizer,
        epoch=args.epochs,
        metrics={"val_loss": avg_val_loss},
        version=timestamp
    )
    
    print(f"Training completed. Final models saved to {final_stnpp_path} and {final_qal_path}")
    writer.close()
    
    return {
        "best_stnpp_model": best_stnpp_path if 'best_stnpp_path' in locals() else None,
        "best_qal_model": best_qal_path if 'best_qal_path' in locals() else None,
        "final_stnpp_model": final_stnpp_path,
        "final_qal_model": final_qal_path,
        "best_val_loss": best_val_loss,
        "final_val_loss": avg_val_loss,
        "log_dir": log_dir
    }


if __name__ == "__main__":
    args = parse_args()
    train_joint(args) 