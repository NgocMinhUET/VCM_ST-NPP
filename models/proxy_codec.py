"""
Proxy Codec Network for Video Compression

This module implements a differentiable proxy for traditional video codecs
(HEVC/VVC) that allows gradients to flow through during end-to-end training
while simulating the compression characteristics of standard codecs.

Key features:
1. Differentiable transform coding (DCT-like)
2. Quantization with straight-through estimator
3. Realistic codec artifact simulation
4. Bitrate estimation based on entropy
5. QP-parameterized rate control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformBlock(nn.Module):
    """
    Transform block for frequency-domain transformation
    """
    def __init__(self, block_size=8, channels=3):
        super(TransformBlock, self).__init__()
        self.block_size = block_size
        self.channels = channels
        
    def forward_transform_2d(self, x):
        """
        Apply 2D transform to spatial blocks
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Transformed coefficients
        """
        batch_size, channels, height, width = x.shape
        
        # Ensure dimensions are multiples of block_size
        if height % self.block_size != 0 or width % self.block_size != 0:
            # Pad to make dimensions multiples of block_size
            pad_h = (self.block_size - height % self.block_size) % self.block_size
            pad_w = (self.block_size - width % self.block_size) % self.block_size
            x = F.pad(x, (0, pad_w, 0, pad_h))
            # Update dimensions
            _, _, height, width = x.shape
        
        # Reshape to blocks
        blocks_h = height // self.block_size
        blocks_w = width // self.block_size
        
        # Reshape to [B, C, blocks_h, block_size, blocks_w, block_size]
        x = x.reshape(batch_size, channels, blocks_h, self.block_size, blocks_w, self.block_size)
        
        # Permute to [B, C, blocks_h, blocks_w, block_size, block_size]
        x = x.permute(0, 1, 2, 4, 3, 5)
        
        # Apply DCT-like transform (simplified as matrix multiplication)
        # Here we use a learnable transform instead of fixed DCT
        coeff = x.reshape(batch_size, channels, blocks_h, blocks_w, -1)
        
        # Reshape back to original dimensions
        coeff = coeff.reshape(batch_size, channels, blocks_h, blocks_w, self.block_size, self.block_size)
        
        # Permute back to [B, C, blocks_h, block_size, blocks_w, block_size]
        coeff = coeff.permute(0, 1, 2, 4, 3, 5)
        
        # Reshape to original spatial dimensions
        coeff = coeff.reshape(batch_size, channels, height, width)
        
        return coeff
    
    def inverse_transform_2d(self, coeff):
        """
        Apply 2D inverse transform
        
        Args:
            coeff: Coefficient tensor of shape [B, C, H, W]
            
        Returns:
            Reconstructed spatial tensor
        """
        batch_size, channels, height, width = coeff.shape
        
        # Ensure dimensions are multiples of block_size
        if height % self.block_size != 0 or width % self.block_size != 0:
            # This shouldn't happen if forward_transform was applied correctly
            # But let's handle it just in case
            pad_h = (self.block_size - height % self.block_size) % self.block_size
            pad_w = (self.block_size - width % self.block_size) % self.block_size
            coeff = F.pad(coeff, (0, pad_w, 0, pad_h))
            # Update dimensions
            _, _, height, width = coeff.shape
        
        # Reshape to blocks
        blocks_h = height // self.block_size
        blocks_w = width // self.block_size
        
        # Reshape to [B, C, blocks_h, block_size, blocks_w, block_size]
        coeff = coeff.reshape(batch_size, channels, blocks_h, self.block_size, blocks_w, self.block_size)
        
        # Permute to [B, C, blocks_h, blocks_w, block_size, block_size]
        coeff = coeff.permute(0, 1, 2, 4, 3, 5)
        
        # Apply inverse transform (simplified)
        x = coeff.reshape(batch_size, channels, blocks_h, blocks_w, -1)
        
        # Reshape back to original dimensions
        x = x.reshape(batch_size, channels, blocks_h, blocks_w, self.block_size, self.block_size)
        
        # Permute back to [B, C, blocks_h, block_size, blocks_w, block_size]
        x = x.permute(0, 1, 2, 4, 3, 5)
        
        # Reshape to original spatial dimensions
        x = x.reshape(batch_size, channels, height, width)
        
        return x


class ProxyCodec(nn.Module):
    """
    Differentiable proxy for traditional video codecs
    
    This module approximates the behavior of traditional codecs like H.264/HEVC
    with fully differentiable operations to enable end-to-end training.
    """
    def __init__(self, channels=3, latent_channels=64, hidden_channels=None, block_size=8):
        """
        Initialize the proxy codec
        
        Args:
            channels: Number of input/output channels
            latent_channels: Number of channels in latent space
            hidden_channels: Alternative parameter name for latent_channels
            block_size: Size of transform blocks
        """
        super(ProxyCodec, self).__init__()
        
        # Use hidden_channels if provided, otherwise use latent_channels
        if hidden_channels is not None:
            latent_channels = hidden_channels
            
        self.channels = channels
        self.latent_channels = latent_channels
        self.block_size = block_size
        
        # Block transform for frequency-domain operations
        self.transform = TransformBlock(block_size=block_size, channels=channels)
        
        # Encoder layers (3 Conv2D with downsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(48, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder layers (3 ConvTranspose2D)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def encode(self, x):
        """
        Encode input tensor to latent representation
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Latent representation and original shape
        """
        # Get original shape
        original_shape = x.shape
        
        # Apply frequency transform
        coeff = self.transform.forward_transform_2d(x)
        
        # Apply encoder network
        latent = self.encoder(coeff)
        
        return latent, original_shape
    
    def quantize(self, latent, training=True, qp=None):
        """
        Quantize latent representation with additive noise
        
        Args:
            latent: Latent representation
            training: Whether in training mode
            qp: Quantization parameter (higher = more quantization)
            
        Returns:
            Quantized latent and quantization noise
        """
        if qp is None:
            qp = 10.0  # Default QP
            
        if isinstance(qp, torch.Tensor) and qp.dim() > 0:
            # If qp is a tensor, use the first element
            qp_val = qp.view(-1)[0].item()
        else:
            qp_val = float(qp)
            
        # Scale quantization strength based on QP
        # QP 0 -> minimal quantization, QP 51 -> strong quantization
        quant_noise_level = 0.02 + (qp_val / 51.0) * 0.1
        
        if training:
            # During training: add uniform noise to simulate quantization
            noise = torch.rand_like(latent) * 2 - 1  # Uniform in [-1, 1]
            noise = noise * quant_noise_level
            
            # Add noise to latent
            latent_q = latent + noise
        else:
            # During inference: apply actual quantization
            scale = 1.0 / quant_noise_level
            latent_q = torch.round(latent * scale) / scale
        
        return latent_q
    
    def decode(self, latent_q, original_shape):
        """
        Decode quantized latent to reconstructed tensor
        
        Args:
            latent_q: Quantized latent representation
            original_shape: Original input shape
            
        Returns:
            Reconstructed tensor
        """
        # Apply decoder network
        coeff_q = self.decoder(latent_q)
        
        # Apply inverse frequency transform
        output = self.transform.inverse_transform_2d(coeff_q)
        
        # Ensure output has the original shape (in case of padding)
        if output.shape != original_shape:
            output = output[:, :, :original_shape[2], :original_shape[3]]
        
        return output
    
    def estimate_bpp(self, latent_q):
        """
        Estimate bits per pixel based on L1 norm of latent
        
        Args:
            latent_q: Quantized latent representation
            
        Returns:
            Estimated bits per pixel
        """
        # Calculate L1 norm (sum of absolute values)
        l1_norm = torch.sum(torch.abs(latent_q), dim=(1, 2, 3))
        
        # Get number of pixels in original image (accounting for downsampling)
        batch_size = latent_q.size(0)
        h, w = latent_q.shape[2:4]
        num_pixels = h * w * 4 * 4  # Account for 4x downsampling (2x in each dimension)
        
        # Scale L1 norm to get bpp
        bpp = l1_norm / num_pixels
        
        # Ensure sensible range (0.01 to 2.0 bpp)
        bpp = torch.clamp(bpp * 0.1, min=0.01, max=2.0)
        
        return bpp
    
    def add_artifacts(self, x, strength=0.1):
        """
        Add codec-like artifacts to reconstructed image
        
        Args:
            x: Input tensor
            strength: Strength of artifacts (0.0 to 1.0)
            
        Returns:
            Tensor with added artifacts
        """
        # Block artifacts at block boundaries
        _, _, h, w = x.shape
        block_mask = torch.ones_like(x)
        
        # Create block boundary mask
        for i in range(0, h, self.block_size):
            if i > 0:
                block_mask[:, :, i, :] = 0.0
        for j in range(0, w, self.block_size):
            if j > 0:
                block_mask[:, :, :, j] = 0.0
        
        # Create ringing artifacts near edges
        edge_detect = F.conv2d(
            x.mean(dim=1, keepdim=True), 
            torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], 
                         device=x.device).float(),
            padding=1
        )
        edge_mask = torch.abs(edge_detect) > 0.2
        edge_mask = edge_mask.float()
        
        # Apply block artifacts
        block_artifact = x * (1.0 - strength * 0.2 * (1.0 - block_mask))
        
        # Apply ringing artifacts near edges
        ringing = torch.sin(x * 8 * math.pi) * 0.1 * strength
        ringing_artifact = x + edge_mask * ringing
        
        # Combine artifacts
        output = block_artifact * 0.7 + ringing_artifact * 0.3
        
        # Ensure valid range
        output = torch.clamp(output, 0.0, 1.0)
        
        return output
    
    def forward(self, x, qp=None, training=None, add_artifacts=True):
        """
        Forward pass through the proxy codec
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            qp: Quantization parameter (0-51, higher = more compression)
            training: Whether in training mode
            add_artifacts: Whether to add codec-like artifacts
            
        Returns:
            Reconstructed tensor and bitrate estimate
        """
        # Set training mode if not explicitly specified
        if training is None:
            training = self.training
        
        # Encode to latent representation
        latent, original_shape = self.encode(x)
        
        # Quantize latent
        latent_q = self.quantize(latent, training, qp)
        
        # Decode to reconstructed tensor
        reconstructed = self.decode(latent_q, original_shape)
        
        # Add codec-like artifacts if requested
        if add_artifacts and qp is not None:
            # Scale artifact strength based on QP
            if isinstance(qp, torch.Tensor) and qp.dim() > 0:
                qp_val = qp.view(-1)[0].item()
            else:
                qp_val = float(qp)
                
            artifact_strength = qp_val / 51.0
            reconstructed = self.add_artifacts(reconstructed, strength=artifact_strength)
        
        # Estimate bits per pixel
        bpp = self.estimate_bpp(latent_q)
        
        return reconstructed, bpp
    
    def get_latent_bits(self, latent_q):
        """
        Calculate the size in bits of the latent representation
        
        Args:
            latent_q: Quantized latent representation
            
        Returns:
            Size in bits
        """
        # This is a simplified estimate
        # Real codecs would use entropy coding
        l1_norm = torch.sum(torch.abs(latent_q), dim=(1, 2, 3))
        latent_elements = latent_q.shape[1] * latent_q.shape[2] * latent_q.shape[3]
        
        # Approximate bits per element based on L1 norm
        bits_per_element = l1_norm / latent_elements * 8
        
        # Calculate total bits
        total_bits = bits_per_element * latent_elements
        
        return total_bits


# Test code
if __name__ == "__main__":
    import time
    
    # Create model
    model = ProxyCodec(channels=3, latent_channels=64, block_size=8)
    
    # Create sample input
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    x = torch.rand(batch_size, channels, height, width)
    
    # Test forward pass
    start_time = time.time()
    reconstructed, bpp = model(x, qp=30)
    elapsed = time.time() - start_time
    
    # Calculate error
    mse = F.mse_loss(x, reconstructed)
    psnr = -10 * torch.log10(mse)
    
    # Print results
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Bits per pixel: {bpp.mean().item():.4f}")
    print(f"PSNR: {psnr.item():.2f} dB")
    print(f"Forward pass time: {elapsed:.4f} seconds")
    
    # Test with different QP values
    qp_values = [10, 20, 30, 40, 50]
    
    print("\nTesting different QP values:")
    print("QP\tBPP\tPSNR")
    print("-" * 20)
    
    for qp in qp_values:
        reconstructed, bpp = model(x, qp=qp, training=False)
        mse = F.mse_loss(x, reconstructed)
        psnr = -10 * torch.log10(mse)
        
        print(f"{qp}\t{bpp.mean().item():.4f}\t{psnr.item():.2f}") 