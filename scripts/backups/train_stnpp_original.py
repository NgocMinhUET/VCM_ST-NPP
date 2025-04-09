#!/usr/bin/env python3
"""
Script for training the Spatio-Temporal Neural Preprocessing (ST-NPP) model.
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
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from models.stnpp import STNPP
from models.qal import QAL, ConditionalQAL, PixelwiseQAL
from utils.model_utils import save_model_with_version, load_model_with_version


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
        # Sort image files to ensure proper sequence
        image_files = sorted(image_files)
        
        # Group images by directory (each directory is typically a different sequence)
        sequences = {}
        for img_file in image_files:
            # Use parent directory as sequence key
            seq_key = str(img_file.parent)
            if seq_key not in sequences:
                sequences[seq_key] = []
            
            # Load and preprocess image
            img = cv2.imread(str(img_file))
            if img is not None:
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to a fixed size for consistency
                img = cv2.resize(img, (224, 224))
                
                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0
                
                sequences[seq_key].append(img)
        
        # Create time_steps sequences with stride
        final_sequences = []
        for seq_name, frames in sequences.items():
            if len(frames) >= self.time_steps:
                for i in range(0, len(frames) - self.time_steps + 1, self.frame_stride):
                    sequence = frames[i:i + self.time_steps]
                    final_sequences.append(sequence)
        
        return final_sequences
    
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
    parser = argparse.ArgumentParser(description='Train ST-NPP and QAL models')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to video dataset')
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='Path to validation dataset (if None, uses a portion of training data)')
    
    # Model parameters
    parser.add_argument('--stnpp_backbone', type=str, default='resnet50',
                        help='Backbone CNN for spatial branch (resnet34, resnet50, efficientnet_b4)')
    parser.add_argument('--temporal_model', type=str, default='3dcnn',
                        help='Temporal model type (3dcnn, convlstm)')
    parser.add_argument('--qal_type', type=str, default='standard',
                        help='QAL type (standard, conditional, pixelwise)')
    parser.add_argument('--fusion_type', type=str, default='concatenation',
                        help='Fusion type for ST-NPP (concatenation, attention)')
    parser.add_argument('--output_channels', type=int, default=128,
                        help='Number of output channels for ST-NPP')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--qp_values', type=str, default='22,27,32,37',
                        help='Comma-separated list of QP values to train with')
    parser.add_argument('--lambda_value', type=float, default=0.1,
                        help='Lambda for rate-distortion tradeoff')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--stnpp_dir', type=str, default='stnpp',
                        help='Subdirectory for ST-NPP models')
    parser.add_argument('--qal_dir', type=str, default='qal',
                        help='Subdirectory for QAL models')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save models every N epochs')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--resume_stnpp', type=str, default=None,
                        help='Path to ST-NPP model to resume training')
    parser.add_argument('--resume_qal', type=str, default=None,
                        help='Path to QAL model to resume training')
    
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


class RDLoss(nn.Module):
    """
    Rate-Distortion Loss module.
    
    Combines distortion loss (MSE) with rate penalty.
    """
    def __init__(self, lambda_value=0.1):
        super(RDLoss, self).__init__()
        self.lambda_value = lambda_value
        self.mse_loss = nn.MSELoss()
        
    def forward(self, original, processed, estimated_rate=None):
        # Distortion loss
        distortion_loss = self.mse_loss(original, processed)
        
        # Rate loss (if provided)
        if estimated_rate is not None:
            return distortion_loss + self.lambda_value * estimated_rate
        
        return distortion_loss


def train(args):
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    stnpp_output_dir = os.path.join(args.output_dir, args.stnpp_dir)
    qal_output_dir = os.path.join(args.output_dir, args.qal_dir)
    os.makedirs(stnpp_output_dir, exist_ok=True)
    os.makedirs(qal_output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f'stnpp_training_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing ST-NPP model...")
    try:
        stnpp_model = STNPP(
            input_channels=3,
            output_channels=args.output_channels,
            spatial_backbone=args.stnpp_backbone,
            temporal_model=args.temporal_model,
            fusion_type=args.fusion_type,
            pretrained=True
        ).to(device)
    except Exception as e:
        print(f"Error initializing ST-NPP model: {e}")
        raise
    
    # Initialize QAL model based on type
    print(f"Initializing QAL model with type: {args.qal_type}")
    try:
        if args.qal_type == 'standard':
            qal_model = QAL(
                feature_channels=args.output_channels,
                hidden_dim=64
            ).to(device)
        elif args.qal_type == 'conditional':
            qal_model = ConditionalQAL(
                feature_channels=args.output_channels,
                hidden_dim=64,
                kernel_size=3,
                temporal_kernel_size=3
            ).to(device)
        elif args.qal_type == 'pixelwise':
            qal_model = PixelwiseQAL(
                feature_channels=args.output_channels,
                hidden_dim=64
            ).to(device)
        else:
            raise ValueError(f"Unknown QAL type: {args.qal_type}")
    except Exception as e:
        print(f"Error initializing QAL model: {e}")
        raise
    
    # Set up optimizers
    stnpp_optimizer = optim.Adam(stnpp_model.parameters(), lr=args.lr)
    qal_optimizer = optim.Adam(qal_model.parameters(), lr=args.lr)
    
    # Learning rate schedulers
    stnpp_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        stnpp_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    qal_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        qal_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_stnpp:
        print(f"Resuming ST-NPP from checkpoint: {args.resume_stnpp}")
        stnpp_model, metadata = load_model_with_version(
            stnpp_model, args.resume_stnpp, device, stnpp_optimizer
        )
        if 'epoch' in metadata:
            start_epoch = metadata['epoch']
        if 'metrics' in metadata and 'val_loss' in metadata['metrics']:
            best_val_loss = metadata['metrics']['val_loss']
    
    if args.resume_qal:
        print(f"Resuming QAL from checkpoint: {args.resume_qal}")
        qal_model, _ = load_model_with_version(
            qal_model, args.resume_qal, device, qal_optimizer
        )
    
    # Parse QP values
    qp_values = [int(qp) for qp in args.qp_values.split(',')]
    
    # Set up datasets and dataloaders
    train_dataset = VideoDataset(args.dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    if args.val_dataset:
        val_dataset = VideoDataset(args.val_dataset)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
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
            num_workers=args.num_workers
        )
    
    # Set up loss function
    criterion = RDLoss(lambda_value=args.lambda_value)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        stnpp_model.train()
        qal_model.train()
        train_loss = 0.0
        train_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            frames = batch['frames'].to(device)
            
            # Choose a random QP for this batch
            qp = random.choice(qp_values)
            
            # Convert qp to a tensor with batch_size elements to match the batch size of frames
            qp_tensor = torch.full((frames.size(0),), qp, dtype=torch.float32, device=device)
            
            # Forward pass through ST-NPP
            preprocessed_frames = stnpp_model(frames)
            
            # Forward pass through QAL with the chosen QP
            reconstructed_frames = qal_model(preprocessed_frames, qp_tensor)
            
            # Calculate loss
            loss = criterion(frames, reconstructed_frames)
            
            # Backward pass and optimization
            stnpp_optimizer.zero_grad()
            qal_optimizer.zero_grad()
            loss.backward()
            stnpp_optimizer.step()
            qal_optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * frames.size(0)
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
                
                # Log to TensorBoard
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/stnpp_lr', stnpp_optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('train/qal_lr', qal_optimizer.param_groups[0]['lr'], global_step)
                
                # Add sample images periodically
                if batch_idx % 50 == 0:
                    # Original frame
                    writer.add_image('train/original', frames[0].cpu(), global_step)
                    # Preprocessed frame
                    writer.add_image('train/preprocessed', preprocessed_frames[0].cpu(), global_step)
                    # Reconstructed frame
                    writer.add_image('train/reconstructed', reconstructed_frames[0].cpu(), global_step)
                    # Add histograms for model parameters
                    for name, param in stnpp_model.named_parameters():
                        if param.requires_grad:
                            writer.add_histogram(f'stnpp/{name}', param.data.cpu().numpy(), global_step)
                    for name, param in qal_model.named_parameters():
                        if param.requires_grad:
                            writer.add_histogram(f'qal/{name}', param.data.cpu().numpy(), global_step)
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataset)
        train_time = time.time() - train_start_time
        print(f"Training Loss: {avg_train_loss:.6f}, Time: {train_time:.2f}s")
        
        # Validation phase
        stnpp_model.eval()
        qal_model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                frames = batch['frames'].to(device)
                
                # Evaluate on all QP values
                batch_loss = 0
                for qp in qp_values:
                    # Create QP tensor with batch_size elements
                    qp_tensor = torch.full((frames.size(0),), qp, dtype=torch.float32, device=device)
                    
                    # Forward pass through ST-NPP
                    preprocessed_frames = stnpp_model(frames)
                    
                    # Forward pass through QAL with the current QP
                    reconstructed_frames = qal_model(preprocessed_frames, qp_tensor)
                    
                    # Calculate loss
                    loss = criterion(frames, reconstructed_frames)
                    batch_loss += loss.item()
                
                # Average loss across QP values
                batch_loss /= len(qp_values)
                val_loss += batch_loss * frames.size(0)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataset)
        val_time = time.time() - val_start_time
        print(f"Validation Loss: {avg_val_loss:.6f}, Time: {val_time:.2f}s")
        
        # Update learning rate schedulers
        stnpp_scheduler.step(avg_val_loss)
        qal_scheduler.step(avg_val_loss)
        
        # Log to TensorBoard
        writer.add_scalar('val/loss', avg_val_loss, (epoch + 1) * len(train_loader))
        
        # Save models
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            stnpp_path = save_model_with_version(
                stnpp_model,
                stnpp_output_dir,
                f"stnpp_epoch_{epoch+1}",
                optimizer=stnpp_optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": avg_val_loss},
                version=timestamp
            )
            qal_path = save_model_with_version(
                qal_model,
                qal_output_dir,
                f"qal_epoch_{epoch+1}",
                optimizer=qal_optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": avg_val_loss},
                version=timestamp
            )
            print(f"Saved models to {stnpp_path} and {qal_path}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stnpp_path = save_model_with_version(
                stnpp_model,
                stnpp_output_dir,
                "stnpp_best",
                optimizer=stnpp_optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": avg_val_loss},
                version=timestamp
            )
            qal_path = save_model_with_version(
                qal_model,
                qal_output_dir,
                "qal_best",
                optimizer=qal_optimizer,
                epoch=epoch + 1,
                metrics={"val_loss": avg_val_loss},
                version=timestamp
            )
            print(f"Saved best models to {stnpp_path} and {qal_path}")
    
    print("Training completed!")
    return stnpp_model, qal_model


if __name__ == "__main__":
    args = parse_args()
    train(args) 