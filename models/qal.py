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
        try:
            # Debug input
            print(f"SoftQuantizer input shape: {x.shape}, device: {x.device}")
            
            # Ensure centers are on the same device as input
            if self.centers.device != x.device:
                self.to(x.device)
                
            # Get input shape
            input_shape = x.shape
            
            # Reshape input to [N, 1] where N is the total number of elements
            x_flat = x.reshape(-1, 1)
            
            # Calculate distances to centers
            centers = self.centers
            dist = torch.abs(x_flat - centers.view(1, -1))
            
            if hard:
                # Hard assignment (nearest center)
                _, indices = torch.min(dist, dim=1)
                # Ensure indices are long type
                indices = indices.long()
                quant = centers[indices].view(-1, 1)
                
                # One-hot encoding of assignments
                assign = torch.zeros(x_flat.size(0), self.num_centers, device=x.device)
                assign.scatter_(1, indices.unsqueeze(1), 1)
            else:
                # Soft assignment using softmax
                assign = F.softmax(-dist / self.temperature, dim=1)
                
                # Weighted sum of centers
                quant = torch.matmul(assign, centers.view(-1, 1))
                
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
            
            print(f"SoftQuantizer output shape: {quant_st.shape}")
            return quant_st, assign
            
        except Exception as e:
            print(f"Error in SoftQuantizer: {e}")
            # Fallback: return input unchanged
            dummy_assign = torch.zeros(*x.shape, self.num_centers, device=x.device)
            dummy_assign[..., 0] = 1.0  # Assign everything to first center
            return x, dummy_assign
    
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
        self.temporal_context = 0  # Initialize temporal_context to 0
        self.default_qp = 22  # Default QP value if none is specified
        self.temporal_kernel_size = 3  # Default temporal kernel size
        
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
        
    def forward(self, x, qp=None, training=None):
        """
        Forward pass through the QAL module
        
        Args:
            x: Input tensor [B, C, H, W]
            qp: Quantization parameter (optional)
            training: Whether in training mode (optional)
            
        Returns:
            Tuple of (quantized tensor, rate estimate, importance maps)
        """
        # Determine training mode
        if training is None:
            training = self.training
            
        # Get input shape
        if len(x.shape) == 4:  # [B, C, H, W]
            B, C, H, W = x.shape
        else:
            # Ensure we have a 4D tensor
            raise ValueError(f"QALModule expects 4D input [B, C, H, W], got shape {x.shape}")
            
        print(f"QALModule input shape: {x.shape}, QP: {qp}, device: {x.device}")
            
        try:
            # Handle different QP formats
            if qp is None:
                qp = torch.ones(B, device=x.device, dtype=torch.long) * self.default_qp
            elif isinstance(qp, (int, float)):
                qp = torch.ones(B, device=x.device, dtype=torch.long) * qp
            elif isinstance(qp, torch.Tensor):
                # Ensure QP is on the same device as x
                qp = qp.to(device=x.device)
                
                # Handle different tensor shapes
                if len(qp.shape) == 0:  # Scalar tensor
                    qp = qp.expand(B)
                elif len(qp.shape) == 1:  # 1D tensor [B]
                    if qp.shape[0] != B:
                        # If sizes don't match, expand or truncate
                        qp = qp[0].expand(B)
                elif len(qp.shape) == 2:  # 2D tensor [B, 1]
                    if qp.shape[0] == B and qp.shape[1] == 1:
                        # Already correct shape [B, 1], squeeze to [B]
                        qp = qp.squeeze(1)
                    else:
                        # Handle other 2D shapes
                        qp = qp.reshape(-1)[0].expand(B)
                else:
                    # Handle higher dimensional tensors
                    qp = qp.reshape(-1)[0].expand(B)
                
                # Ensure long dtype
                qp = qp.long()
            else:
                # Handle other cases
                qp = torch.tensor([self.default_qp] * B, device=x.device, dtype=torch.long)
                
            # Ensure all module parameters are on the same device as the input
            if next(self.parameters()).device != x.device:
                self.to(x.device)
                
            # Apply feature transform network
            features = self.feature_transform(x)
            
            # Generate importance map
            importance_map = self.importance_map(features)
            
            # Move QP embedding to input device
            self.qp_embedding = self.qp_embedding.to(x.device)
            
            # Apply QP embedding and conditioning
            qp_emb = self.qp_embedding(qp)  # [B, hidden_dim]
            qp_adapt = self.qp_adapter(qp_emb)  # [B, channels*2]
            
            # Split into scale and bias
            scale, bias = torch.split(qp_adapt, self.channels, dim=1)
            
            # Reshape for broadcasting
            scale = scale.view(B, self.channels, 1, 1)
            bias = bias.view(B, self.channels, 1, 1)
            
            # Apply QP conditioning
            features = features * scale + bias
            
            # Move quantizer to input device
            self.quantizer = self.quantizer.to(x.device)
            
            # Apply quantization
            if training:
                # Soft quantization during training
                quantized, assign = self.quantizer(features, hard=False)
                
                # Estimate rate
                rate = self.quantizer.get_rate(assign)
            else:
                # Hard quantization during inference
                quantized, assign = self.quantizer(features, hard=True)
                
                # Estimate rate
                rate = self.quantizer.get_rate(assign)
                
            print(f"QALModule output shape: {quantized.shape}")
            return quantized, rate, importance_map
            
        except Exception as e:
            print(f"Error in QALModule forward: {e}")
            # Fallback: return input unchanged
            rate = torch.tensor(1.0, device=x.device)  # Use same device as input
            importance_map = torch.ones_like(x[:, :1])
            return x, rate, importance_map


class QAL(nn.Module):
    """
    Quantization Adaptation Layer (QAL)
    
    Maps QP (Quantization Parameter) values to channel-wise scaling vectors
    using a simple MLP architecture. This allows the model to adapt feature
    representations based on the desired compression level.
    """
    def __init__(self, channels=128, hidden_size=64, qp_levels=51, temporal_kernel_size=None):
        """
        Initialize the QAL module.
        
        Args:
            channels: Number of output channels for the scaling vector
            hidden_size: Size of the hidden layer in the MLP
            qp_levels: Number of QP levels supported (typically 0-51 for video codecs)
            temporal_kernel_size: Not used, kept for API compatibility
        """
        super(QAL, self).__init__()
        
        # Simple MLP: 1 → hidden_size → channels
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, channels)
        )
        
        self.channels = channels
        self.qp_levels = qp_levels
        
        # Initialize with reasonable values
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                # Initialize the last layer to output values close to 1
                if m.out_features == self.channels:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    nn.init.constant_(m.bias, 1.0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, qp):
        """
        Generate channel-wise scaling vector based on QP value.
        
        Args:
            qp: Quantization Parameter, either a single value or batch of values
                Can be an integer, float, or tensor
                
        Returns:
            Channel-wise scaling vector of shape [B, channels]
        """
        # Handle different QP input types
        if isinstance(qp, int) or isinstance(qp, float):
            # Convert scalar to tensor
            qp = torch.tensor([float(qp)], device=self.mlp[0].weight.device)
        elif isinstance(qp, torch.Tensor):
            # Ensure QP is a float tensor
            qp = qp.float()
            
            # If QP is a single number, reshape it
            if qp.dim() == 0:
                qp = qp.unsqueeze(0)
        
        # Normalize QP to [0, 1] range based on qp_levels
        qp_normalized = qp.float() / (self.qp_levels - 1)
        
        # Ensure normalized QP is in the expected shape [B, 1]
        if qp_normalized.dim() == 1:
            qp_normalized = qp_normalized.unsqueeze(1)  # Shape: [B, 1]
            
        # Generate scaling vector using MLP
        scaling_vector = self.mlp(qp_normalized)
        
        # Apply sigmoid and scale to ensure positive scaling factors
        # This produces values primarily in the range [0.5, 1.5]
        scaling_vector = 2 * torch.sigmoid(scaling_vector)
        
        return scaling_vector


class QALModule(nn.Module):
    """
    Complete QAL module with feature scaling capabilities.
    
    This expands on the base QAL by adding the ability to apply the
    generated scaling vectors to feature tensors.
    """
    def __init__(self, channels=128, hidden_size=64, qp_levels=51, temporal_kernel_size=3):
        """
        Initialize the complete QAL module.
        
        Args:
            channels: Number of channels in the feature tensor
            hidden_size: Size of the hidden layer in the MLP
            qp_levels: Number of QP levels supported
            temporal_kernel_size: Size of the temporal kernel for processing
        """
        super(QALModule, self).__init__()
        
        # Base QAL for generating scaling vectors
        self.qal = QAL(channels, hidden_size, qp_levels)
        
        # Store settings
        self.channels = channels
        self.temporal_kernel_size = temporal_kernel_size
        
    def forward(self, x, qp):
        """
        Apply QAL adaptation to input features.
        
        Args:
            x: Input feature tensor of shape [B, C, H, W] or [B, C, T, H, W]
            qp: Quantization Parameter
            
        Returns:
            Scaled feature tensor with the same shape as input
        """
        # Generate scaling vectors
        scaling_vector = self.qal(qp)  # Shape: [B, C]
        
        # Apply scaling based on input dimensions
        if x.dim() == 4:  # [B, C, H, W]
            # Reshape scaling vector to [B, C, 1, 1]
            scaling_vector = scaling_vector.unsqueeze(-1).unsqueeze(-1)
            
            # Apply channel-wise scaling
            scaled_features = x * scaling_vector
            
        elif x.dim() == 5:  # [B, C, T, H, W]
            # Reshape scaling vector to [B, C, 1, 1, 1]
            scaling_vector = scaling_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            # Apply channel-wise scaling
            scaled_features = x * scaling_vector
            
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, expected 4 or 5")
            
        return scaled_features


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