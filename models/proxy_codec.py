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
    Learnable transform coding block that approximates DCT/DST transforms
    in traditional codecs.
    """
    def __init__(self, block_size=8, channels=1):
        """
        Initialize transform block.
        
        Args:
            block_size: Size of transform blocks (typically 4, 8, 16, or 32)
            channels: Number of channels to process
        """
        super(TransformBlock, self).__init__()
        
        self.block_size = block_size
        self.channels = channels
        
        # Learnable transform kernels (approximating DCT)
        self.forward_transform = nn.Parameter(
            self._init_dct_weights(block_size), 
            requires_grad=True
        )
        
        self.inverse_transform = nn.Parameter(
            self._init_dct_weights(block_size).transpose(0, 1), 
            requires_grad=True
        )
    
    def _init_dct_weights(self, size):
        """Initialize with DCT-like basis functions"""
        weights = torch.zeros(size, size)
        
        # First row is DC component
        weights[0, :] = 1.0 / math.sqrt(size)
        
        # Remaining rows are cosine waves of increasing frequency
        for i in range(1, size):
            for j in range(size):
                weights[i, j] = math.cos(math.pi * i * (2 * j + 1) / (2 * size))
                weights[i, j] *= math.sqrt(2.0 / size)
        
        return weights
    
    def forward_transform_2d(self, x):
        """Apply 2D separable transform to image blocks"""
        # Reshape input to blocks: [B, C, H, W] -> [B*C*num_blocks_h*num_blocks_w, block_size, block_size]
        batch_size, channels, height, width = x.shape
        
        # Ensure dimensions are multiples of block_size
        pad_h = (self.block_size - height % self.block_size) % self.block_size
        pad_w = (self.block_size - width % self.block_size) % self.block_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height, width = x.shape[2:]
        
        # Reshape to blocks
        x = x.reshape(batch_size, channels, height // self.block_size, self.block_size, 
                     width // self.block_size, self.block_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.reshape(-1, self.block_size, self.block_size)
        
        # Apply transforms (rows then columns)
        y = torch.matmul(self.forward_transform, x)
        y = torch.matmul(y, self.forward_transform.transpose(0, 1))
        
        return y, (batch_size, channels, height, width)
    
    def inverse_transform_2d(self, y, original_shape):
        """Apply 2D inverse transform to frequency coefficients"""
        batch_size, channels, height, width = original_shape
        
        # Apply inverse transforms
        x = torch.matmul(self.inverse_transform, y)
        x = torch.matmul(x, self.inverse_transform.transpose(0, 1))
        
        # Reshape back to image
        num_blocks_h = height // self.block_size
        num_blocks_w = width // self.block_size
        
        x = x.reshape(batch_size, channels, num_blocks_h, num_blocks_w, 
                     self.block_size, self.block_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.reshape(batch_size, channels, height, width)
        
        return x
    
    def forward(self, x):
        """
        Forward pass: transform -> coefficients -> inverse transform.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Reconstructed tensor and transform coefficients
        """
        # Forward transform
        coeffs, original_shape = self.forward_transform_2d(x)
        
        # Inverse transform
        x_reconstructed = self.inverse_transform_2d(coeffs, original_shape)
        
        return x_reconstructed, coeffs, original_shape


class QuantizationSimulator(nn.Module):
    """
    Differentiable quantization module that simulates codec quantization.
    """
    def __init__(self, num_qp_levels=52, qp_scale_base=2.0):
        """
        Initialize quantization simulator.
        
        Args:
            num_qp_levels: Number of QP levels to support
            qp_scale_base: Base for exponential QP scaling
        """
        super(QuantizationSimulator, self).__init__()
        
        self.num_qp_levels = num_qp_levels
        self.qp_scale_base = qp_scale_base
        
        # QP to quantization step size mapping (exponential)
        self.register_buffer(
            'qp_to_stepsize', 
            torch.tensor([qp_scale_base**(qp/6.0) for qp in range(num_qp_levels)])
        )
    
    def quantize(self, x, qp, training=True):
        """
        Apply quantization with straight-through estimator.
        
        Args:
            x: Input tensor
            qp: Quantization parameter [B]
            training: Whether in training mode
            
        Returns:
            Quantized tensor
        """
        # Get quantization step size for each sample in batch
        stepsize = self.qp_to_stepsize[qp].view(-1, 1, 1, 1)
        
        # Quantize
        x_scaled = x / stepsize
        
        if training:
            # Differentiable approximation (additive uniform noise)
            noise = torch.zeros_like(x_scaled).uniform_(-0.5, 0.5)
            x_quant = x_scaled + noise
            x_quant = torch.round(x_quant) * stepsize
            
            # Straight-through estimator (pass gradients through)
            x_quant = x + (x_quant - x).detach()
        else:
            # Hard rounding for inference
            x_quant = torch.round(x_scaled) * stepsize
        
        return x_quant
    
    def estimate_bitrate(self, x_quant, qp):
        """
        Estimate bitrate based on entropy of quantized coefficients.
        
        Args:
            x_quant: Quantized coefficients
            qp: Quantization parameter
            
        Returns:
            Estimated bits per pixel
        """
        # Simple entropy estimate based on coefficient distribution
        eps = 1e-10
        stepsize = self.qp_to_stepsize[qp].view(-1, 1, 1, 1)
        
        # Normalize by step size to get discrete indices
        x_norm = x_quant / stepsize
        
        # Separate DC and AC components for more accurate estimation
        dc = x_norm[:, :1]  # First coefficient is DC
        ac = x_norm[:, 1:]
        
        # Entropy estimation based on standard deviation
        # Higher variance = more bits needed
        ac_std = torch.std(ac, dim=(1, 2, 3))
        dc_std = torch.std(dc, dim=(1, 2, 3))
        
        # Bits per coefficient (BPC) estimate
        # Using simple log2 relationship with variance
        bpc_ac = torch.log2(ac_std + eps) + 1.0  # +1 for sign bit
        bpc_dc = torch.log2(dc_std + eps) + 2.0  # +2 for higher precision
        
        # Combine DC and AC bits, weighted by coefficient count
        total_coeffs = x_norm.shape[1] * x_norm.shape[2] * x_norm.shape[3]
        dc_coeffs = dc.shape[1] * dc.shape[2] * dc.shape[3]
        ac_coeffs = total_coeffs - dc_coeffs
        
        total_bits = (bpc_dc * dc_coeffs) + (bpc_ac * ac_coeffs)
        bits_per_pixel = total_bits / total_coeffs
        
        return bits_per_pixel


class BlockingArtifactSimulator(nn.Module):
    """
    Simulates blocking artifacts similar to those in traditional codecs.
    """
    def __init__(self, block_size=8, severity_factor=1.0):
        """
        Initialize blocking artifact simulator.
        
        Args:
            block_size: Size of transform blocks
            severity_factor: Controls severity of blocking artifacts
        """
        super(BlockingArtifactSimulator, self).__init__()
        
        self.block_size = block_size
        self.severity_factor = severity_factor
        
        # Learnable parameters to control blocking artifact intensity
        self.block_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, qp):
        """
        Apply blocking artifacts to reconstructed image.
        
        Args:
            x: Input tensor
            qp: Quantization parameter
            
        Returns:
            Tensor with simulated blocking artifacts
        """
        batch_size, channels, height, width = x.shape
        
        # Create block boundary mask
        block_mask = torch.ones_like(x)
        
        # Horizontal boundaries
        for i in range(self.block_size, height, self.block_size):
            block_mask[:, :, i-1:i+1, :] = 0.8
            
        # Vertical boundaries
        for i in range(self.block_size, width, self.block_size):
            block_mask[:, :, :, i-1:i+1] = 0.8
        
        # Severity increases with QP
        qp_factor = (qp.float() / 51.0).view(-1, 1, 1, 1)
        severity = self.severity_factor * qp_factor * self.block_weight
        
        # Apply blocking artifact effect
        x_blocked = x * (block_mask ** severity)
        
        return x_blocked


class RingingArtifactSimulator(nn.Module):
    """
    Simulates ringing artifacts around edges due to high-frequency loss.
    """
    def __init__(self):
        """Initialize ringing artifact simulator."""
        super(RingingArtifactSimulator, self).__init__()
        
        # Edge detection kernel (Sobel)
        self.register_buffer('edge_kernel', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Ringing kernel (approximates sinc function)
        self.register_buffer('ring_kernel', torch.tensor([
            [-0.1, -0.2, -0.1],
            [-0.2,  1.2, -0.2],
            [-0.1, -0.2, -0.1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Learnable parameter for ringing intensity
        self.ring_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, x, qp):
        """
        Apply ringing artifacts to reconstructed image.
        
        Args:
            x: Input tensor
            qp: Quantization parameter
            
        Returns:
            Tensor with simulated ringing artifacts
        """
        batch_size, channels, height, width = x.shape
        
        # Calculate edge map
        edge_maps = []
        for c in range(channels):
            # Apply edge detection to each channel
            edge = F.conv2d(
                x[:, c:c+1], 
                self.edge_kernel,
                padding=1
            )
            edge = torch.abs(edge)
            edge_maps.append(edge)
        
        edge_map = torch.cat(edge_maps, dim=1)
        
        # Apply ringing kernel to each channel
        ring_maps = []
        for c in range(channels):
            ring = F.conv2d(
                x[:, c:c+1],
                self.ring_kernel,
                padding=1
            )
            ring_maps.append(ring)
        
        ring_map = torch.cat(ring_maps, dim=1)
        
        # Modulate ringing by edge map and QP
        qp_factor = (qp.float() / 51.0).view(-1, 1, 1, 1)
        ring_intensity = self.ring_weight * qp_factor
        
        # Apply ringing near edges
        x_ringed = x + ring_intensity * ring_map * edge_map
        
        return x_ringed


class ProxyCodec(nn.Module):
    """
    Differentiable proxy for traditional video codecs.
    
    Simulates key aspects of codecs like HEVC/VVC while allowing
    gradient flow during training.
    """
    def __init__(self, 
                block_size=8, 
                channels=3, 
                num_qp_levels=52,
                artifact_simulation=True):
        """
        Initialize proxy codec.
        
        Args:
            block_size: Size of transform blocks
            channels: Number of input channels
            num_qp_levels: Number of QP levels to support
            artifact_simulation: Whether to simulate codec artifacts
        """
        super(ProxyCodec, self).__init__()
        
        self.block_size = block_size
        self.channels = channels
        self.num_qp_levels = num_qp_levels
        self.artifact_simulation = artifact_simulation
        
        # Transform coding
        self.transform = TransformBlock(block_size, channels)
        
        # Quantization
        self.quantization = QuantizationSimulator(num_qp_levels)
        
        # Artifact simulators
        if artifact_simulation:
            self.blocking_simulator = BlockingArtifactSimulator(block_size)
            self.ringing_simulator = RingingArtifactSimulator()
        
    def encode(self, x, qp, training=True):
        """
        Encode input tensor.
        
        Args:
            x: Input tensor [B, C, H, W]
            qp: Quantization parameter [B]
            training: Whether in training mode
            
        Returns:
            Tuple of (quantized_coeffs, original_shape)
        """
        # Apply transform
        _, coeffs, original_shape = self.transform.forward(x)
        
        # Quantize coefficients
        quant_coeffs = self.quantization.quantize(coeffs, qp, training)
        
        return quant_coeffs, original_shape
    
    def decode(self, quant_coeffs, original_shape, qp):
        """
        Decode quantized coefficients.
        
        Args:
            quant_coeffs: Quantized coefficients
            original_shape: Original input shape
            qp: Quantization parameter
            
        Returns:
            Reconstructed tensor
        """
        # Apply inverse transform
        x_rec = self.transform.inverse_transform_2d(quant_coeffs, original_shape)
        
        # Apply artifact simulation
        if self.artifact_simulation:
            x_rec = self.blocking_simulator(x_rec, qp)
            x_rec = self.ringing_simulator(x_rec, qp)
        
        # Clip to valid range
        x_rec = torch.clamp(x_rec, 0, 1)
        
        return x_rec
    
    def forward(self, x, qp, training=True):
        """
        Forward pass through proxy codec.
        
        Args:
            x: Input tensor [B, C, H, W]
            qp: Quantization parameter [B]
            training: Whether in training mode
            
        Returns:
            Tuple of (reconstructed, quantized_coeffs, bitrate)
        """
        # Encode
        quant_coeffs, original_shape = self.encode(x, qp, training)
        
        # Estimate bitrate
        bitrate = self.quantization.estimate_bitrate(quant_coeffs, qp)
        
        # Decode
        x_rec = self.decode(quant_coeffs, original_shape, qp)
        
        return x_rec, quant_coeffs, bitrate
    
    def compress_at_target_quality(self, x, target_psnr=35.0, max_iterations=10):
        """
        Compress input at target quality level by searching for appropriate QP.
        
        Args:
            x: Input tensor [B, C, H, W]
            target_psnr: Target PSNR value
            max_iterations: Maximum number of iterations for QP search
            
        Returns:
            Tuple of (reconstructed, quantized_coeffs, bitrate, qp)
        """
        batch_size = x.shape[0]
        
        # Start with middle QP
        qp_low = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        qp_high = torch.ones(batch_size, dtype=torch.long, device=x.device) * (self.num_qp_levels - 1)
        qp = torch.ones(batch_size, dtype=torch.long, device=x.device) * (self.num_qp_levels // 2)
        
        # Binary search for target quality
        for _ in range(max_iterations):
            # Compress with current QP
            x_rec, quant_coeffs, bitrate = self.forward(x, qp, training=False)
            
            # Compute PSNR
            mse = F.mse_loss(x_rec, x, reduction='none').mean(dim=[1, 2, 3])
            psnr = -10 * torch.log10(mse + 1e-10)
            
            # Update QP based on PSNR
            too_low_quality = psnr < target_psnr
            too_high_quality = psnr > target_psnr + 1.0
            
            # Update search bounds
            qp_high[too_low_quality] = qp[too_low_quality]
            qp_low[too_high_quality] = qp[too_high_quality]
            
            # Break if converged
            if not (too_low_quality.any() or too_high_quality.any()):
                break
                
            # Update QP for next iteration
            qp = (qp_low + qp_high) // 2
        
        # Final compression with selected QP
        x_rec, quant_coeffs, bitrate = self.forward(x, qp, training=False)
        
        return x_rec, quant_coeffs, bitrate, qp


# Test code
if __name__ == "__main__":
    # Parameters for testing
    batch_size = 2
    channels = 3
    height = 64
    width = 64
    
    # Create random input (simulating normalized image)
    x = torch.rand(batch_size, channels, height, width)
    
    # Create random QP values
    qp = torch.randint(0, 51, (batch_size,))
    
    # Create proxy codec
    model = ProxyCodec(
        block_size=8,
        channels=channels,
        artifact_simulation=True
    )
    
    # Forward pass (compress and decompress)
    x_rec, quant_coeffs, bitrate = model(x, qp)
    
    # Print shapes and statistics
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {x_rec.shape}")
    print(f"Quantized coefficients shape: {quant_coeffs.shape}")
    print(f"Estimated bitrate: {bitrate.mean().item():.4f} bits per pixel")
    
    # Compute PSNR
    mse = F.mse_loss(x_rec, x)
    psnr = -10 * torch.log10(mse)
    print(f"PSNR: {psnr.item():.2f} dB")
    
    # Test compression at different QP levels
    for test_qp in [5, 15, 25, 35, 45]:
        qp_tensor = torch.tensor([test_qp] * batch_size)
        x_rec, _, bitrate = model(x, qp_tensor)
        
        # Compute PSNR
        mse = F.mse_loss(x_rec, x)
        psnr = -10 * torch.log10(mse)
        
        print(f"QP {test_qp}: PSNR = {psnr.item():.2f} dB, Bitrate = {bitrate.mean().item():.4f} bpp")
    
    # Test quality-targeted compression
    print("\nQuality-targeted compression:")
    for target_psnr in [30.0, 35.0, 40.0]:
        x_rec, _, bitrate, qp = model.compress_at_target_quality(x, target_psnr)
        
        # Compute actual PSNR
        mse = F.mse_loss(x_rec, x)
        psnr = -10 * torch.log10(mse)
        
        print(f"Target PSNR {target_psnr:.1f} dB: Actual PSNR = {psnr.item():.2f} dB, " +
              f"QP = {qp.float().mean().item():.1f}, Bitrate = {bitrate.mean().item():.4f} bpp") 