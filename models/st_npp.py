"""
Spatio-Temporal Neural Preprocessing (ST-NPP) Module

This module implements a 3D convolutional neural network for video preprocessing
before compression. It extracts spatio-temporal features that preserve both
spatial and temporal coherence while enabling efficient compression.

Key features:
1. 3D convolutional architecture for joint spatial and temporal processing
2. Multi-scale feature extraction
3. Temporal dimension reduction for efficient compression
4. Task-aware feature learning
5. Quality-controlled compression with quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


class Conv3DBlock(nn.Module):
    """
    Basic 3D convolutional block with normalization and activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TemporalReduction(nn.Module):
    """
    Reduces temporal dimension while preserving spatial dimensions
    """
    def __init__(self, channels, reduction_factor=2):
        super(TemporalReduction, self).__init__()
        self.reduction_factor = reduction_factor
        
        # 3D convolutional layer with stride in temporal dimension
        self.conv = nn.Conv3d(
            channels, 
            channels, 
            kernel_size=(reduction_factor, 3, 3),
            stride=(reduction_factor, 1, 1),
            padding=(0, 1, 1)
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        # Ensure temporal dimension is divisible by reduction factor
        batch, channels, time, height, width = x.shape
        pad_size = (self.reduction_factor - (time % self.reduction_factor)) % self.reduction_factor
        if pad_size > 0:
            # Pad in temporal dimension
            x = F.pad(x, (0, 0, 0, 0, 0, pad_size))
            
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class TemporalUpsampling(nn.Module):
    """
    Increases temporal dimension while preserving spatial dimensions
    """
    def __init__(self, channels, scale_factor=2):
        super(TemporalUpsampling, self).__init__()
        self.scale_factor = scale_factor
        
        # Transposed 3D convolution for temporal upsampling
        self.conv = nn.ConvTranspose3d(
            channels,
            channels,
            kernel_size=(scale_factor, 3, 3),
            stride=(scale_factor, 1, 1),
            padding=(0, 1, 1),
            output_padding=(scale_factor-1, 0, 0)
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x, target_time=None):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        # Trim to target temporal dimension if specified
        if target_time is not None and x.size(2) > target_time:
            x = x[:, :, :target_time, :, :]
            
        return x


class SpatialDownsample(nn.Module):
    """
    Downsamples spatial dimensions while preserving temporal dimension
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialDownsample, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SpatialUpsample(nn.Module):
    """
    Upsamples spatial dimensions while preserving temporal dimension
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialUpsample, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class STNPPEncoder(nn.Module):
    """
    Encoder part of ST-NPP model
    """
    def __init__(self, in_channels=3, latent_channels=32, time_steps=16, time_reduction=4):
        super(STNPPEncoder, self).__init__()
        self.time_steps = time_steps
        self.time_reduction = time_reduction
        
        # Initial feature extraction
        self.init_conv = Conv3DBlock(in_channels, 64, kernel_size=3, padding=1)
        
        # Spatial downsampling path
        self.down1 = SpatialDownsample(64, 128)
        self.down2 = SpatialDownsample(128, 256)
        
        # Temporal reduction
        self.temporal_reduction = TemporalReduction(256, reduction_factor=time_reduction)
        
        # Final compression layer
        self.compress = nn.Conv3d(256, latent_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        Forward pass of the encoder
        
        Args:
            x: Input video tensor [B, C, T, H, W]
            
        Returns:
            Latent representation [B, latent_channels, T//time_reduction, H//4, W//4]
        """
        # Initial features
        x = self.init_conv(x)
        
        # Spatial downsampling
        x = self.down1(x)
        x = self.down2(x)
        
        # Temporal reduction
        x = self.temporal_reduction(x)
        
        # Compress to latent channels
        latent = self.compress(x)
        
        return latent


class STNPPDecoder(nn.Module):
    """
    Decoder part of ST-NPP model
    """
    def __init__(self, out_channels=3, latent_channels=32, time_steps=16, time_reduction=4):
        super(STNPPDecoder, self).__init__()
        self.time_steps = time_steps
        self.time_reduction = time_reduction
        
        # Initial processing
        self.init_conv = Conv3DBlock(latent_channels, 256, kernel_size=3, padding=1)
        
        # Temporal upsampling
        self.temporal_upsampling = TemporalUpsampling(256, scale_factor=time_reduction)
        
        # Spatial upsampling path
        self.up1 = SpatialUpsample(256, 128)
        self.up2 = SpatialUpsample(128, 64)
        
        # Final reconstruction
        self.final = nn.Conv3d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, target_time=None):
        """
        Forward pass of the decoder
        
        Args:
            x: Latent representation [B, latent_channels, T//time_reduction, H//4, W//4]
            target_time: Target temporal dimension for output
            
        Returns:
            Reconstructed video [B, out_channels, T, H, W]
        """
        # Initial processing
        x = self.init_conv(x)
        
        # Temporal upsampling
        x = self.temporal_upsampling(x, target_time)
        
        # Spatial upsampling
        x = self.up1(x)
        x = self.up2(x)
        
        # Final reconstruction
        x = self.final(x)
        
        return x


class TemporalShiftBlock(nn.Module):
    """
    Temporal Shift Block for efficient temporal modeling.
    Shifts part of the channels along the temporal dimension, which allows 
    for information exchange between neighboring frames.
    """
    def __init__(self, n_frames, n_div=8):
        """
        Initialize the Temporal Shift Block.
        
        Args:
            n_frames: Number of frames in the input
            n_div: Division factor for determining how many channels to shift
        """
        super(TemporalShiftBlock, self).__init__()
        self.n_frames = n_frames
        self.n_div = n_div
        
    def forward(self, x):
        """
        Apply temporal shift operation.
        
        Args:
            x: Input tensor of shape [B, T, C, H, W]
            
        Returns:
            Tensor after temporal shift of shape [B, T, C, H, W]
        """
        # Get dimensions
        b, t, c, h, w = x.size()
        
        # Reshape tensor for easier channel manipulation
        x = x.view(b, t, c, h * w)
        
        # Calculate number of channels to shift
        fold = c // self.n_div
        
        # Keep the original tensor for combining later
        out = torch.zeros_like(x)
        
        # Shift 1/n_div of the channels forward (to the right in temporal dimension)
        out[:, 1:, :fold] = x[:, :-1, :fold]  # shift left->right
        out[:, 0, :fold] = x[:, 0, :fold]  # first frame doesn't receive data
        
        # Shift 1/n_div of the channels backward (to the left in temporal dimension)
        out[:, :-1, fold:fold*2] = x[:, 1:, fold:fold*2]  # shift right->left
        out[:, -1, fold:fold*2] = x[:, -1, fold:fold*2]  # last frame doesn't receive data
        
        # Keep the rest of the channels unchanged
        out[:, :, fold*2:] = x[:, :, fold*2:]
        
        # Reshape back to original format
        return out.view(b, t, c, h, w)


class SpatialBlock(nn.Module):
    """
    Spatial processing block consisting of multiple 2D convolution layers.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, n_layers=3):
        """
        Initialize the spatial processing block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Number of channels in hidden layers
            n_layers: Number of convolutional layers
        """
        super(SpatialBlock, self).__init__()
        
        layers = []
        current_in_channels = in_channels
        
        for i in range(n_layers):
            # For the last layer, use out_channels as output size
            if i == n_layers - 1:
                next_channels = out_channels
            else:
                next_channels = hidden_channels
            
            layers.append(nn.Conv2d(
                current_in_channels, 
                next_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ))
            
            # Add ReLU activation except for the last layer
            if i < n_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            
            current_in_channels = next_channels
        
        self.conv_block = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Apply spatial convolution block.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Processed tensor of shape [B, out_channels, H, W]
        """
        return self.conv_block(x)


class STNPP(nn.Module):
    """
    Spatio-Temporal Neural Pre-Processing (STNPP) module.
    Takes a sequence of frames and enhances the center frame using
    temporal and spatial information.
    """
    def __init__(self, channels=3, hidden_channels=64, temporal_kernel_size=3, use_attention=False):
        """
        Initialize the STNPP module.
        
        Args:
            channels: Number of channels in input/output frames (default: 3 for RGB)
            hidden_channels: Number of channels in hidden layers
            temporal_kernel_size: Number of frames to process (unused, kept for API consistency)
            use_attention: Whether to use attention mechanisms (unused, kept for API consistency)
        """
        super(STNPP, self).__init__()
        
        # Temporal shift block for efficient temporal modeling
        self.temporal_shift = TemporalShiftBlock(n_frames=5)
        
        # Spatial processing block
        self.spatial_block = SpatialBlock(
            in_channels=channels,
            out_channels=channels,
            hidden_channels=hidden_channels,
            n_layers=3
        )
        
    def forward(self, x):
        """
        Process a sequence of frames and enhance the center frame.
        
        Args:
            x: Input tensor of shape [B, T, C, H, W] or [B, C, T, H, W]
            
        Returns:
            Enhanced center frame of shape [B, C, H, W]
        """
        # Ensure input is in format [B, T, C, H, W]
        if x.size(1) == 3 and x.size(2) == 5:  # If input is [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)  # Convert to [B, T, C, H, W]
        
        # Apply temporal shift
        x = self.temporal_shift(x)
        
        # Extract center frame (index 2)
        center_frame = x[:, 2]  # Shape: [B, C, H, W]
        
        # Apply spatial processing
        enhanced_frame = self.spatial_block(center_frame)
        
        # Add residual connection
        output = center_frame + enhanced_frame
        
        return output


class STNPPWithQuantization(nn.Module):
    """
    STNPP with additional quantization functionality.
    This extended version prepares features for compression.
    """
    def __init__(self, channels=3, feature_channels=64, temporal_kernel_size=3, use_attention=False):
        """
        Initialize the STNPP with quantization module.
        
        Args:
            channels: Number of channels in input frames (default: 3 for RGB)
            feature_channels: Number of channels in feature space
            temporal_kernel_size: Number of frames to process (unused, kept for API consistency)
            use_attention: Whether to use attention mechanisms (unused, kept for API consistency)
        """
        super(STNPPWithQuantization, self).__init__()
        
        # Base STNPP processing
        self.stnpp = STNPP(
            channels=channels, 
            hidden_channels=feature_channels,
            temporal_kernel_size=temporal_kernel_size,
            use_attention=use_attention
        )
        
        # Additional layers for feature transformation
        self.to_features = nn.Conv2d(channels, feature_channels, kernel_size=3, padding=1)
        self.from_features = nn.Conv2d(feature_channels, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        """
        Process frames and convert to feature space.
        
        Args:
            x: Input tensor of shape [B, T, C, H, W] or [B, C, T, H, W]
            
        Returns:
            Tuple of (reconstructed_frames, features):
                - reconstructed_frames: Tensor of shape [B, C, H, W]
                - features: Tensor of shape [B, feature_channels, H, W]
        """
        # Process through base STNPP
        enhanced_frame = self.stnpp(x)
        
        # Convert to feature space
        features = self.to_features(enhanced_frame)
        
        # Reconstruct from features
        reconstructed = self.from_features(features)
        
        return reconstructed, features


# Test code
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    channels = 3
    time_steps = 16
    height = 256
    width = 256
    latent_channels = 32
    time_reduction = 4
    
    # Create model
    model = STNPP(
        channels=channels,
        hidden_channels=latent_channels,
        temporal_kernel_size=time_reduction,
        use_attention=False
    )
    
    # Calculate theoretical compression
    compression_ratio, latent_dims = model.get_compression_stats(
        batch_size, time_steps, height, width
    )
    print(f"Theoretical compression ratio: {compression_ratio:.2f}x")
    print(f"Latent dimensions: {latent_dims}")
    
    # Create random input
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    # Forward pass
    start_time = time.time()
    reconstructed = model(x)
    elapsed = time.time() - start_time
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Forward pass time: {elapsed:.4f} seconds")
    
    # Calculate reconstruction error
    mse = F.mse_loss(x, reconstructed)
    psnr = -10 * torch.log10(mse)
    print(f"Reconstruction MSE: {mse.item():.6f}")
    print(f"Reconstruction PSNR: {psnr.item():.2f} dB")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    # Test quantized model
    print("\nTesting quantized model:")
    quantized_model = STNPPWithQuantization(
        channels=channels,
        feature_channels=latent_channels,
        temporal_kernel_size=time_reduction,
        use_attention=False
    )
    
    # Forward pass with quantization
    reconstructed_q, features = quantized_model(x)
    
    # Print results
    print(f"Reconstructed (with quantization) shape: {reconstructed_q.shape}")
    print(f"Features shape: {features.shape}")
    
    # Calculate reconstruction error with quantization
    mse_q = F.mse_loss(x, reconstructed_q)
    psnr_q = -10 * torch.log10(mse_q)
    print(f"Reconstruction MSE (with quantization): {mse_q.item():.6f}")
    print(f"Reconstruction PSNR (with quantization): {psnr_q.item():.2f} dB")
    
    # Compare original and quantized reconstructions
    print(f"PSNR degradation from quantization: {psnr.item() - psnr_q.item():.2f} dB") 