"""
Quantization Adaptation Layer (QAL) Module

This module implements the Quantization Adaptation Layer (QAL) that bridges
neural representations with traditional video codecs. It adapts the neural
features produced by the ST-NPP module to the quantization characteristics
of standard codecs like HEVC/VVC.

Key features:
1. QP-conditional processing
2. Importance map generation for bit allocation
3. Differentiable quantization with learnable centers
4. Temporal context modeling
5. Rate-distortion optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoftQuantizer(nn.Module):
    """
    Differentiable soft quantization module with learnable centers
    """
    def __init__(self, num_centers=16, init_range=(-1, 1), temperature=1.0):
        """
        Initialize soft quantizer with learnable centers
        
        Args:
            num_centers: Number of quantization centers
            init_range: Range for initializing centers
            temperature: Temperature for softmax (lower = harder quantization)
        """
        super(SoftQuantizer, self).__init__()
        
        # Initialize quantization centers
        centers = torch.linspace(init_range[0], init_range[1], num_centers)
        self.register_parameter("centers", nn.Parameter(centers))
        
        # Temperature parameter for softmax-based soft assignment
        self.temperature = temperature
        self.num_centers = num_centers
        
    def forward(self, x, hard=False):
        """
        Forward pass with soft or hard quantization
        
        Args:
            x: Input tensor to quantize
            hard: Whether to use hard quantization (for inference)
            
        Returns:
            Quantized tensor and soft assignment probabilities
        """
        # Get input shape
        input_shape = x.shape
        
        # Reshape input to [N, 1] where N is the total number of elements
        x_flat = x.reshape(-1, 1)
        
        # Calculate distances to centers
        dist = torch.abs(x_flat - self.centers.view(1, -1))
        
        if hard:
            # Hard assignment (nearest center)
            _, indices = torch.min(dist, dim=1)
            quant = self.centers[indices].view(-1, 1)
            
            # One-hot encoding of assignments
            assign = torch.zeros(x_flat.size(0), self.num_centers, device=x.device)
            assign.scatter_(1, indices.unsqueeze(1), 1)
        else:
            # Soft assignment using softmax
            assign = F.softmax(-dist / self.temperature, dim=1)
            
            # Weighted sum of centers
            quant = torch.matmul(assign, self.centers.view(-1, 1))
            
        # Reshape quantized values back to input shape
        quant = quant.reshape(input_shape)
        
        # For straight-through gradient estimation during training
        if not hard:
            # Pass through gradients from quant to x
            quant_st = x + (quant - x).detach()
        else:
            quant_st = quant
            
        # Reshape assignments for return
        assign = assign.reshape(*input_shape, self.num_centers)
        
        return quant_st, assign
    
    def get_rate(self, assign):
        """
        Estimate rate (bits) from soft assignments
        
        Args:
            assign: Soft assignment probabilities
            
        Returns:
            Estimated bits required for encoding
        """
        # Calculate entropy of assignments
        # assign has shape [..., num_centers]
        eps = 1e-10
        entropy = -torch.sum(assign * torch.log2(assign + eps), dim=-1)
        
        # Return average entropy
        return entropy.mean()


class QALModule(nn.Module):
    """
    Core Quantization Adaptation Layer module for individual frames
    """
    def __init__(self, channels, qp_levels=51, hidden_dim=128):
        """
        Initialize QAL module
        
        Args:
            channels: Number of input/output channels
            qp_levels: Number of QP levels to support (typically 0-51 for H.264/HEVC)
            hidden_dim: Dimension of hidden layers
        """
        super(QALModule, self).__init__()
        
        self.channels = channels
        self.qp_levels = qp_levels
        self.hidden_dim = hidden_dim
        
        # QP embedding
        self.qp_embedding = nn.Embedding(qp_levels, hidden_dim)
        
        # QP conditioning network
        self.qp_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, channels * 2)  # Scale and bias
        )
        
        # Feature transform network
        self.feature_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Importance map generator for adaptive bit allocation
        self.importance_map = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Differentiable quantizer
        self.quantizer = SoftQuantizer(num_centers=16, temperature=0.5)
        
    def forward(self, x, qp, training=True):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            qp: Quantization parameter (0-51)
            training: Whether in training mode
            
        Returns:
            Quantized tensor and rate estimate
        """
        batch_size = x.size(0)
        
        # QP conditioning
        qp_embed = self.qp_embedding(qp)  # [B, hidden_dim]
        qp_params = self.qp_adapter(qp_embed)  # [B, channels*2]
        
        # Split into scale and bias
        scale, bias = torch.split(qp_params, self.channels, dim=1)
        scale = scale.view(batch_size, self.channels, 1, 1)
        bias = bias.view(batch_size, self.channels, 1, 1)
        
        # Apply feature transformation
        x = self.feature_transform(x)
        
        # Apply QP-conditional scaling
        x = x * scale + bias
        
        # Generate importance map
        importance = self.importance_map(x)
        
        # Apply scaled quantization according to importance
        # Higher importance = finer quantization
        x_scaled = x * importance
        quant, assign = self.quantizer(x_scaled, hard=not training)
        
        # Rescale back
        quant = quant / (importance + 1e-8)
        
        # Estimate rate
        rate = self.quantizer.get_rate(assign)
        
        return quant, rate, importance


class QAL(nn.Module):
    """
    Complete Quantization Adaptation Layer with temporal context
    """
    def __init__(self, channels, qp_levels=51, temporal_kernel_size=3):
        """
        Initialize QAL
        
        Args:
            channels: Number of input/output channels
            qp_levels: Number of QP levels to support
            temporal_kernel_size: Size of temporal kernel for context
        """
        super(QAL, self).__init__()
        
        self.channels = channels
        self.qp_levels = qp_levels
        self.temporal_kernel_size = temporal_kernel_size
        
        # Temporal context module
        padding = (temporal_kernel_size - 1) // 2
        self.temporal_context = nn.Conv3d(
            channels, channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            padding=(padding, 0, 0)
        )
        
        # Frame-level QAL module
        self.qal_module = QALModule(channels, qp_levels)
        
        # Output transform
        self.output_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x, qp, training=True):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, T, H, W]
            qp: Quantization parameter (0-51) or tensor of parameters
            training: Whether in training mode
            
        Returns:
            Quantized tensor, rate estimate, and importance maps
        """
        batch_size, channels, time, height, width = x.shape
        
        # Ensure QP is a tensor
        if isinstance(qp, int):
            qp = torch.tensor([qp] * batch_size, device=x.device)
        
        # Apply temporal context
        x_temporal = self.temporal_context(x)
        
        # Process each frame with QAL
        outputs = []
        rates = []
        importance_maps = []
        
        for t in range(time):
            # Extract frame with temporal context
            x_t = x_temporal[:, :, t]
            
            # Apply QAL module
            quant_t, rate_t, importance_t = self.qal_module(x_t, qp, training)
            
            # Apply output transform
            out_t = self.output_transform(quant_t)
            
            # Store results
            outputs.append(out_t)
            rates.append(rate_t)
            importance_maps.append(importance_t)
            
        # Stack outputs along temporal dimension
        output = torch.stack(outputs, dim=2)
        importance = torch.stack(importance_maps, dim=2)
        
        # Average rate across frames
        rate = torch.stack(rates).mean()
        
        return output, rate, importance
    
    def compress(self, x, target_bpp, max_qp=51, min_qp=0, tolerance=0.1):
        """
        Compress with target bitrate using binary search on QP
        
        Args:
            x: Input tensor [B, C, T, H, W]
            target_bpp: Target bits per pixel
            max_qp: Maximum QP value to try
            min_qp: Minimum QP value to try
            tolerance: Acceptable deviation from target bitrate
            
        Returns:
            Compressed tensor, actual bitrate, and QP value used
        """
        device = x.device
        batch_size = x.size(0)
        
        # Start with middle QP
        low_qp = min_qp
        high_qp = max_qp
        
        # Binary search for appropriate QP
        for _ in range(10):  # Max 10 iterations
            mid_qp = (low_qp + high_qp) // 2
            qp = torch.tensor([mid_qp] * batch_size, device=device)
            
            # Compress with current QP
            with torch.no_grad():
                output, rate, _ = self(x, qp, training=False)
            
            # Check if bitrate is close enough to target
            if abs(rate.item() - target_bpp) < tolerance:
                break
                
            # Adjust QP range
            if rate.item() > target_bpp:
                # Too many bits, increase QP (reduce quality)
                low_qp = mid_qp
            else:
                # Too few bits, decrease QP (increase quality)
                high_qp = mid_qp
                
            # Check if search range is exhausted
            if high_qp - low_qp <= 1:
                break
                
        # Final compression with selected QP
        with torch.no_grad():
            output, rate, importance = self(x, qp, training=False)
            
        return output, rate, mid_qp, importance


# Test code
if __name__ == "__main__":
    # Parameters for testing
    batch_size = 2
    channels = 32
    time_steps = 8
    height = 64
    width = 64
    
    # Test QAL module (frame level)
    qal_module = QALModule(channels)
    
    # Create random input for a single frame
    x_frame = torch.randn(batch_size, channels, height, width)
    qp = torch.tensor([30, 40])  # Different QP for each example
    
    # Forward pass
    quant_frame, rate_frame, importance_frame = qal_module(x_frame, qp)
    
    # Print shapes and metrics
    print(f"Frame input shape: {x_frame.shape}")
    print(f"Frame output shape: {quant_frame.shape}")
    print(f"Frame importance map shape: {importance_frame.shape}")
    print(f"Frame estimated rate: {rate_frame.item():.4f} bits per element")
    
    # Test complete QAL (with temporal context)
    qal = QAL(channels)
    
    # Create random input for video
    x_video = torch.randn(batch_size, channels, time_steps, height, width)
    
    # Forward pass
    quant_video, rate_video, importance_video = qal(x_video, qp)
    
    # Print shapes and metrics
    print(f"\nVideo input shape: {x_video.shape}")
    print(f"Video output shape: {quant_video.shape}")
    print(f"Video importance map shape: {importance_video.shape}")
    print(f"Video estimated rate: {rate_video.item():.4f} bits per element")
    
    # Test compression with target bitrate
    target_bpp = 1.0
    compressed, actual_bpp, used_qp, importance = qal.compress(x_video, target_bpp)
    
    # Print results
    print(f"\nTarget bitrate: {target_bpp:.2f} bpp")
    print(f"Actual bitrate: {actual_bpp.item():.2f} bpp")
    print(f"QP value used: {used_qp}")
    print(f"Compressed output shape: {compressed.shape}")
    
    # Calculate reconstruction error
    mse = F.mse_loss(x_video, compressed)
    psnr = -10 * torch.log10(mse)
    print(f"Reconstruction MSE: {mse.item():.6f}")
    print(f"Reconstruction PSNR: {psnr.item():.2f} dB") 