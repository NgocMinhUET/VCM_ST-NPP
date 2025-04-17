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


class STNPP(nn.Module):
    """
    Complete ST-NPP model combining encoder and decoder
    """
    def __init__(self, channels=3, latent_channels=32, time_steps=16, time_reduction=4):
        super(STNPP, self).__init__()
        
        self.channels = channels
        self.latent_channels = latent_channels
        self.time_steps = time_steps
        self.time_reduction = time_reduction
        
        # Create encoder and decoder
        self.encoder = STNPPEncoder(
            in_channels=channels,
            latent_channels=latent_channels,
            time_steps=time_steps,
            time_reduction=time_reduction
        )
        
        self.decoder = STNPPDecoder(
            out_channels=channels,
            latent_channels=latent_channels,
            time_steps=time_steps,
            time_reduction=time_reduction
        )
        
    def forward(self, x):
        """
        Forward pass of ST-NPP
        
        Args:
            x: Input video tensor [B, C, T, H, W]
            
        Returns:
            Reconstructed video and latent representation
        """
        # Get input dimensions
        original_time = x.size(2)
        
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent, target_time=original_time)
        
        return reconstructed, latent
    
    def get_compression_stats(self, batch_size, time_steps, height, width):
        """
        Calculate compression statistics
        
        Args:
            batch_size: Batch size
            time_steps: Number of time steps
            height: Height of video
            width: Width of video
            
        Returns:
            Compression ratio and latent dimensions
        """
        # Calculate original size
        original_pixels = batch_size * self.channels * time_steps * height * width
        
        # Calculate latent size
        latent_time = time_steps // self.time_reduction
        latent_height = height // 4
        latent_width = width // 4
        latent_pixels = batch_size * self.latent_channels * latent_time * latent_height * latent_width
        
        # Calculate compression ratio
        compression_ratio = original_pixels / latent_pixels
        
        # Latent dimensions
        latent_dims = (batch_size, self.latent_channels, latent_time, latent_height, latent_width)
        
        return compression_ratio, latent_dims


class STNPPWithQuantization(nn.Module):
    """
    ST-NPP model with quantization for actual compression
    """
    def __init__(self, channels=3, latent_channels=32, time_steps=16, time_reduction=4, num_quantize_levels=256):
        super(STNPPWithQuantization, self).__init__()
        
        # Base ST-NPP model
        self.stnpp = STNPP(channels, latent_channels, time_steps, time_reduction)
        
        # Number of quantization levels (e.g. 256 for 8-bit quantization)
        self.num_quantize_levels = num_quantize_levels
        
        # Register scale and zero point for quantization
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        
    def update_quantization_params(self, latent):
        """
        Update quantization parameters based on latent activation statistics
        
        Args:
            latent: Latent tensor
            
        Returns:
            Updated scale and zero point
        """
        # Find min and max values
        min_val = torch.min(latent)
        max_val = torch.max(latent)
        
        # Compute scale and zero point for quantization
        range_val = max_val - min_val
        self.scale = range_val / (self.num_quantize_levels - 1)
        self.zero_point = min_val
        
        return self.scale, self.zero_point
        
    def quantize(self, latent, update_params=True):
        """
        Quantize latent representation
        
        Args:
            latent: Latent tensor
            update_params: Whether to update quantization parameters
            
        Returns:
            Quantized latent and dequantized latent for reconstruction
        """
        # Update quantization parameters if needed
        if update_params:
            self.update_quantization_params(latent)
            
        # Quantize: x_q = round((x - zero_point) / scale)
        latent_q = torch.round((latent - self.zero_point) / self.scale)
        
        # Clamp to quantization range
        latent_q = torch.clamp(latent_q, 0, self.num_quantize_levels - 1)
        
        # Dequantize: x_dq = x_q * scale + zero_point
        latent_dq = latent_q * self.scale + self.zero_point
        
        return latent_q, latent_dq
    
    def estimate_bitrate(self, latent_q, height, width):
        """
        Estimate bitrate for the quantized latent
        
        Args:
            latent_q: Quantized latent tensor
            height: Original height of video
            width: Original width of video
            
        Returns:
            Bitrate in bits per pixel
        """
        # Calculate bits per element in latent
        bits_per_element = torch.log2(torch.tensor(self.num_quantize_levels, dtype=torch.float))
        
        # Total bits
        total_elements = torch.numel(latent_q)
        total_bits = total_elements * bits_per_element
        
        # Original number of pixels
        batch_size = latent_q.size(0)
        time_steps = self.stnpp.time_steps
        original_pixels = batch_size * time_steps * height * width
        
        # Bits per pixel
        bpp = total_bits / original_pixels
        
        return bpp
        
    def forward(self, x, training=True):
        """
        Forward pass with quantization
        
        Args:
            x: Input video tensor [B, C, T, H, W]
            training: Whether in training mode
            
        Returns:
            Reconstructed video, quantized latent, and bitrate estimate
        """
        # Get dimensions
        batch_size, _, _, height, width = x.shape
        
        # Encode
        reconstructed, latent = self.stnpp(x)
        
        # Apply quantization
        if training:
            # In training, use straight-through estimator
            latent_q, latent_dq = self.quantize(latent)
            latent_dq = latent + (latent_dq - latent).detach()  # STE
        else:
            # In inference, use actual quantization
            latent_q, latent_dq = self.quantize(latent)
            
        # Decode quantized latent
        reconstructed_q = self.stnpp.decoder(latent_dq)
        
        # Estimate bitrate
        bpp = self.estimate_bitrate(latent_q, height, width)
        
        return reconstructed_q, latent_q, bpp


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
        latent_channels=latent_channels,
        time_steps=time_steps,
        time_reduction=time_reduction
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
    reconstructed, latent = model(x)
    elapsed = time.time() - start_time
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
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
        latent_channels=latent_channels,
        time_steps=time_steps,
        time_reduction=time_reduction
    )
    
    # Forward pass with quantization
    reconstructed_q, latent_q, bpp = quantized_model(x)
    
    # Print results
    print(f"Quantized latent shape: {latent_q.shape}")
    print(f"Reconstructed (with quantization) shape: {reconstructed_q.shape}")
    print(f"Estimated bitrate: {bpp.item():.4f} bits per pixel")
    
    # Calculate reconstruction error with quantization
    mse_q = F.mse_loss(x, reconstructed_q)
    psnr_q = -10 * torch.log10(mse_q)
    print(f"Reconstruction MSE (with quantization): {mse_q.item():.6f}")
    print(f"Reconstruction PSNR (with quantization): {psnr_q.item():.2f} dB")
    
    # Compare original and quantized reconstructions
    print(f"PSNR degradation from quantization: {psnr.item() - psnr_q.item():.2f} dB") 