"""
Combined Task-Aware Video Processing model.

This module integrates the ST-NPP, QAL, and Proxy Codec components into a complete
end-to-end pipeline for task-aware video preprocessing and compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.st_npp import STNPP
from models.qal import QAL
from models.proxy_codec import ProxyCodec


class TaskAwareVideoProcessor(nn.Module):
    """
    Complete task-aware video processing pipeline.
    
    Integrates Spatio-Temporal Neural Preprocessing (ST-NPP), Quantization Adaptation Layer (QAL),
    and a Proxy Codec into an end-to-end trainable model that optimizes video preprocessing
    for downstream computer vision tasks.
    """
    def __init__(
        self,
        input_channels=3,
        feature_dim=64,
        fusion_dim=128,
        latent_channels=8,
        num_embeddings=512
    ):
        super(TaskAwareVideoProcessor, self).__init__()
        
        # Spatio-Temporal Neural Preprocessing module
        self.stnpp = STNPP(
            input_channels=input_channels,
            feature_dim=feature_dim,
            fusion_dim=fusion_dim
        )
        
        # Quantization Adaptation Layer
        self.qal = QAL(feature_dim=fusion_dim)
        
        # Proxy Codec
        self.proxy_codec = ProxyCodec(
            input_channels=fusion_dim,
            latent_channels=latent_channels,
            num_embeddings=num_embeddings
        )
        
        # Store parameters
        self.feature_dim = feature_dim
        self.fusion_dim = fusion_dim
        self.latent_channels = latent_channels
        self.num_embeddings = num_embeddings
        
    def forward(self, x, qp=None):
        """
        Forward pass through the complete pipeline.
        
        Args:
            x (torch.Tensor): Input video tensor of shape [B, C, T, H, W]
            qp (torch.Tensor, optional): Quantization parameter of shape [B, 1]
                                         If None, defaults to 22 (medium quality)
            
        Returns:
            tuple: (reconstructed, latent, indices, vq_loss)
                reconstructed: Reconstructed video from the proxy codec
                latent: Latent representation
                indices: Quantization indices (for bitrate calculation)
                vq_loss: Vector quantization loss
        """
        # Default QP value if not provided
        if qp is None:
            qp = torch.ones(x.shape[0], 1, device=x.device) * 22.0
        
        # Pass through ST-NPP
        features = self.stnpp(x)
        
        # Apply QAL
        scaling_factors = self.qal(qp)
        
        # Reshape scaling factors for multiplication with features
        scaling_factors = scaling_factors.view(
            scaling_factors.shape[0], scaling_factors.shape[1], 1, 1, 1
        )
        
        # Apply scaling factors to features
        scaled_features = features * scaling_factors
        
        # Pass through proxy codec
        reconstructed, latent, indices, vq_loss = self.proxy_codec(scaled_features)
        
        return reconstructed, latent, indices, vq_loss
    
    def calc_rate_distortion_loss(
        self,
        original,
        reconstructed,
        indices,
        vq_loss,
        lambda_distortion=1.0,
        lambda_rate=0.1
    ):
        """
        Calculate rate-distortion loss.
        
        Args:
            original (torch.Tensor): Original video tensor
            reconstructed (torch.Tensor): Reconstructed video tensor
            indices (torch.Tensor): Quantization indices
            vq_loss (torch.Tensor): Vector quantization loss
            lambda_distortion (float): Weight for distortion term
            lambda_rate (float): Weight for rate term
            
        Returns:
            tuple: (total_loss, distortion_loss, rate_loss)
        """
        # Distortion loss (MSE)
        distortion_loss = F.mse_loss(reconstructed, original)
        
        # Rate loss (estimated from indices)
        batch_size, time_steps, height, width = indices.shape
        bits_per_index = torch.log2(torch.tensor(self.num_embeddings, device=indices.device))
        total_bits = batch_size * time_steps * height * width * bits_per_index
        
        # Original dimensions
        original_time_steps = time_steps * 4  # Compensate for temporal downsampling
        original_height = height * 16        # Compensate for spatial downsampling
        original_width = width * 16          # Compensate for spatial downsampling
        total_pixels = batch_size * original_time_steps * original_height * original_width
        
        rate_loss = total_bits / total_pixels  # Bits per pixel
        
        # Combined loss
        total_loss = lambda_distortion * distortion_loss + lambda_rate * rate_loss + 0.1 * vq_loss
        
        return total_loss, distortion_loss, rate_loss
    
    def calc_task_aware_loss(
        self,
        task_loss,
        distortion_loss,
        rate_loss,
        vq_loss,
        lambda_task=1.0,
        lambda_distortion=1.0,
        lambda_rate=0.1,
        lambda_vq=0.1
    ):
        """
        Calculate combined task-aware loss.
        
        Args:
            task_loss (torch.Tensor): Loss from downstream task (e.g., detection)
            distortion_loss (torch.Tensor): Distortion loss (MSE)
            rate_loss (torch.Tensor): Rate loss (bits per pixel)
            vq_loss (torch.Tensor): Vector quantization loss
            lambda_task (float): Weight for task loss
            lambda_distortion (float): Weight for distortion loss
            lambda_rate (float): Weight for rate loss
            lambda_vq (float): Weight for VQ loss
            
        Returns:
            torch.Tensor: Combined task-aware loss
        """
        return (
            lambda_task * task_loss +
            lambda_distortion * distortion_loss +
            lambda_rate * rate_loss +
            lambda_vq * vq_loss
        )
    
    def encode_with_actual_codec(self, x, qp=None, output_path=None, codec="libx265"):
        """
        Process video with actual codec (non-differentiable).
        For inference and evaluation only.
        
        Args:
            x (torch.Tensor): Input video tensor of shape [B, C, T, H, W]
            qp (int, optional): Quantization parameter for the codec. Default is 22.
            output_path (str, optional): Path to save the encoded video.
            codec (str, optional): Codec to use ('libx264', 'libx265', etc.).
            
        Returns:
            tuple: (encoded_size, frames_numpy)
                encoded_size: Size of the encoded video in bytes
                frames_numpy: Decoded frames as a NumPy array
        """
        import numpy as np
        import tempfile
        import os
        import subprocess
        import cv2
        
        # Default QP
        if qp is None:
            qp = 22
            
        # Apply preprocessing
        batch_size, channels, time_steps, height, width = x.shape
        
        # Process through ST-NPP + QAL
        with torch.no_grad():
            qp_tensor = torch.ones(batch_size, 1, device=x.device) * qp
            features = self.stnpp(x)
            scaling_factors = self.qal(qp_tensor)
            scaling_factors = scaling_factors.view(batch_size, self.fusion_dim, 1, 1, 1)
            scaled_features = features * scaling_factors
        
        # Convert scaled features back to RGB frames
        # This would typically be done by a trainable decoder in a real implementation
        # Here we use a simple projection for demonstration
        with torch.no_grad():
            projection = nn.Conv3d(self.fusion_dim, 3, kernel_size=1).to(x.device)
            projected = projection(scaled_features)
            projected = torch.sigmoid(projected)  # Normalize to [0, 1]
        
        # Convert to NumPy for FFmpeg processing
        frames_numpy = projected.cpu().numpy()
        frames_numpy = np.transpose(frames_numpy[0], (1, 2, 3, 0))  # [T, H, W, C]
        frames_numpy = (frames_numpy * 255).astype(np.uint8)
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save frames as images
            frame_paths = []
            for i in range(frames_numpy.shape[0]):
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frames_numpy[i], cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)
            
            # Define output video path
            if output_path is None:
                output_path = os.path.join(temp_dir, "output.mp4")
            
            # Encode with FFmpeg
            if codec == "libx264":
                cmd = [
                    "ffmpeg", "-y", "-r", "24", "-i", os.path.join(temp_dir, "frame_%04d.png"),
                    "-c:v", "libx264", "-crf", str(qp), "-preset", "medium",
                    "-pix_fmt", "yuv420p", output_path
                ]
            elif codec == "libx265":
                cmd = [
                    "ffmpeg", "-y", "-r", "24", "-i", os.path.join(temp_dir, "frame_%04d.png"),
                    "-c:v", "libx265", "-crf", str(qp), "-preset", "medium",
                    "-pix_fmt", "yuv420p", output_path
                ]
            else:
                raise ValueError(f"Unsupported codec: {codec}")
            
            # Run FFmpeg
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Get encoded size
            encoded_size = os.path.getsize(output_path)
            
            # Decode video
            cap = cv2.VideoCapture(output_path)
            decoded_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                decoded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            decoded_frames_numpy = np.array(decoded_frames)
        
        return encoded_size, decoded_frames_numpy


if __name__ == "__main__":
    # Simple test code
    batch_size = 2
    channels = 3
    time_steps = 16
    height = 256
    width = 256
    
    # Create a random input tensor
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    # Create model
    model = TaskAwareVideoProcessor(
        input_channels=channels,
        feature_dim=64,
        fusion_dim=128,
        latent_channels=8,
        num_embeddings=512
    )
    
    # Forward pass with default QP
    reconstructed, latent, indices, vq_loss = model(x)
    
    # Calculate rate-distortion loss
    total_loss, distortion_loss, rate_loss = model.calc_rate_distortion_loss(
        x, reconstructed, indices, vq_loss
    )
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"VQ Loss: {vq_loss.item():.6f}")
    print(f"Distortion Loss: {distortion_loss.item():.6f}")
    print(f"Rate Loss: {rate_loss.item():.6f}")
    print(f"Total Loss: {total_loss.item():.6f}")
 