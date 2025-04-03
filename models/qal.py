"""
Quantization Adaptation Layer (QAL) Model

This module implements the Quantization Adaptation Layer, which adapts the
feature maps produced by the ST-NPP module based on the quantization parameter (QP)
of the downstream video codec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple


class QAL(nn.Module):
    """
    Quantization Adaptation Layer (QAL)
    
    This layer adapts the feature maps based on the quantization parameter (QP)
    to optimize the trade-off between rate and distortion under different
    compression levels.
    """
    
    def __init__(self, feature_channels: int = 128, hidden_dim: int = 64):
        """
        Initialize the Quantization Adaptation Layer.
        
        Args:
            feature_channels: Number of channels in the feature maps to adapt
            hidden_dim: Dimension of the hidden layers in the MLP
        """
        super(QAL, self).__init__()
        
        # MLP to predict scaling factors based on QP
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_channels),
            nn.Sigmoid()  # Scale factors between 0 and 1
        )
        
    def forward(self, qp: Union[torch.Tensor, float, int], 
                features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass to generate scale factors based on QP.
        
        Args:
            qp: Quantization Parameter (scalar or batch of scalars)
            features: Optional feature tensor to apply scaling (B, C, T, H, W)
            
        Returns:
            If features is None: Scale factors (B, feature_channels)
            If features is provided: Scaled features (B, C, T, H, W)
        """
        # Ensure QP is a tensor with the right shape
        if not isinstance(qp, torch.Tensor):
            qp = torch.tensor([qp], dtype=torch.float32)
        
        if qp.dim() == 0:  # Handle scalar tensor
            qp = qp.unsqueeze(0)
        
        if qp.dim() == 1:  # Handle 1D tensor (batch of scalars)
            qp = qp.unsqueeze(1)  # Shape: (B, 1)
        
        # Generate scale factors
        scale_factors = self.mlp(qp)  # Shape: (B, feature_channels)
        
        # If features are provided, apply the scaling
        if features is not None:
            # Check batch size consistency
            assert scale_factors.size(0) == features.size(0), \
                f"Batch size mismatch: scale_factors ({scale_factors.size(0)}) " \
                f"vs features ({features.size(0)})"
            
            # Reshape scale factors to match feature dimensions for broadcasting
            B, C = scale_factors.shape
            
            if features.dim() == 5:  # (B, C, T, H, W)
                scale_factors = scale_factors.view(B, C, 1, 1, 1)
            else:  # (B, C, H, W)
                scale_factors = scale_factors.view(B, C, 1, 1)
            
            # Apply scaling
            scaled_features = features * scale_factors
            return scaled_features
        
        return scale_factors


class ConditionalQAL(nn.Module):
    """
    Conditional Quantization Adaptation Layer
    
    An enhanced version of QAL that adapts features based on both QP and
    local spatial-temporal characteristics of the features.
    """
    
    def __init__(self, feature_channels: int = 128, hidden_dim: int = 64, 
                 kernel_size: int = 3, temporal_kernel_size: int = 3):
        """
        Initialize the Conditional Quantization Adaptation Layer.
        
        Args:
            feature_channels: Number of channels in the feature maps to adapt
            hidden_dim: Dimension of the hidden layers
            kernel_size: Spatial kernel size for convolution
            temporal_kernel_size: Temporal kernel size for 3D convolution
        """
        super(ConditionalQAL, self).__init__()
        
        # QP encoding
        self.qp_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Feature context encoding (spatial-temporal)
        self.context_encoder = nn.Sequential(
            nn.Conv3d(
                feature_channels, 
                hidden_dim, 
                kernel_size=(temporal_kernel_size, kernel_size, kernel_size),
                padding=(temporal_kernel_size//2, kernel_size//2, kernel_size//2)
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                hidden_dim, 
                hidden_dim, 
                kernel_size=1
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Feature pooling for global context
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fusion of QP and feature context
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_channels),
            nn.Sigmoid()
        )
        
    def forward(self, qp: Union[torch.Tensor, float, int], 
                features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate conditional scale factors and apply them.
        
        Args:
            qp: Quantization Parameter (scalar or batch of scalars)
            features: Feature tensor to analyze and scale (B, C, T, H, W)
            
        Returns:
            Scaled features (B, C, T, H, W)
        """
        B, C, T, H, W = features.shape
        
        # Ensure QP is a tensor with the right shape
        if not isinstance(qp, torch.Tensor):
            qp = torch.tensor([qp], dtype=torch.float32, device=features.device)
        
        if qp.dim() == 0:  # Handle scalar tensor
            qp = qp.unsqueeze(0)
        
        if qp.dim() == 1:  # Handle 1D tensor (batch of scalars)
            qp = qp.unsqueeze(1)  # Shape: (B, 1)
        
        # Ensure QP is on the same device as features
        qp = qp.to(features.device)
        
        # Encode QP
        qp_feat = self.qp_encoder(qp)  # (B, hidden_dim)
        
        # Encode feature context
        context_feat = self.context_encoder(features)  # (B, hidden_dim, T, H, W)
        
        # Pool to get global context
        global_context = self.global_pool(context_feat).view(B, -1)  # (B, hidden_dim)
        
        # Fuse QP and context features
        fused = torch.cat([qp_feat, global_context], dim=1)  # (B, hidden_dim*2)
        scale_factors = self.fusion(fused)  # (B, C)
        
        # Reshape scale factors for broadcasting
        scale_factors = scale_factors.view(B, C, 1, 1, 1)
        
        # Apply scaling
        scaled_features = features * scale_factors
        
        return scaled_features


class PixelwiseQAL(nn.Module):
    """
    Pixelwise Quantization Adaptation Layer
    
    This layer applies different scaling factors to different spatial
    positions in the feature map, allowing for more fine-grained adaptation.
    """
    
    def __init__(self, feature_channels: int = 128, hidden_dim: int = 64):
        """
        Initialize the Pixelwise Quantization Adaptation Layer.
        
        Args:
            feature_channels: Number of channels in the feature maps to adapt
            hidden_dim: Dimension of the hidden layers
        """
        super(PixelwiseQAL, self).__init__()
        
        # QP encoding
        self.qp_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Scale factor generator network
        self.scale_generator = nn.Sequential(
            nn.Conv3d(feature_channels + 1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, feature_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, qp: Union[torch.Tensor, float, int], 
                features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate pixelwise scale factors and apply them.
        
        Args:
            qp: Quantization Parameter (scalar or batch of scalars)
            features: Feature tensor to analyze and scale (B, C, T, H, W)
            
        Returns:
            Scaled features (B, C, T, H, W)
        """
        B, C, T, H, W = features.shape
        
        # Ensure QP is a tensor with the right shape
        if not isinstance(qp, torch.Tensor):
            qp = torch.tensor([qp], dtype=torch.float32, device=features.device)
        
        if qp.dim() == 0:  # Handle scalar tensor
            qp = qp.unsqueeze(0)
        
        if qp.dim() == 1:  # Handle 1D tensor (batch of scalars)
            qp = qp.unsqueeze(1)  # Shape: (B, 1)
        
        # Ensure QP is on the same device as features
        qp = qp.to(features.device)
        
        # Create QP feature map (broadcast QP to feature map dimensions)
        qp_map = qp.view(B, 1, 1, 1, 1).expand(B, 1, T, H, W)
        
        # Concatenate features and QP map along channel dimension
        concat_features = torch.cat([features, qp_map], dim=1)  # (B, C+1, T, H, W)
        
        # Generate pixelwise scale factors
        scale_map = self.scale_generator(concat_features)  # (B, C, T, H, W)
        
        # Apply scaling
        scaled_features = features * scale_map
        
        return scaled_features


def test_qal():
    """Test function to verify QAL implementations."""
    # Create a random feature tensor
    batch_size = 2
    channels = 128
    time_steps = 16
    height = 64
    width = 64
    
    features = torch.randn(batch_size, channels, time_steps, height, width)
    qp = torch.tensor([23, 37], dtype=torch.float32)  # Different QP for each batch item
    
    # Test standard QAL
    print("Testing standard QAL...")
    qal = QAL(feature_channels=channels)
    scale_factors = qal(qp)
    scaled_features = qal(qp, features)
    
    print(f"QP shape: {qp.shape}")
    print(f"Scale factors shape: {scale_factors.shape}")
    print(f"Scaled features shape: {scaled_features.shape}")
    print(f"Scale factor range: {scale_factors.min().item():.4f} - {scale_factors.max().item():.4f}")
    
    # Test conditional QAL
    print("\nTesting conditional QAL...")
    cqal = ConditionalQAL(feature_channels=channels)
    cond_scaled_features = cqal(qp, features)
    
    print(f"Conditionally scaled features shape: {cond_scaled_features.shape}")
    
    # Test pixelwise QAL
    print("\nTesting pixelwise QAL...")
    pqal = PixelwiseQAL(feature_channels=channels)
    pw_scaled_features = pqal(qp, features)
    
    print(f"Pixelwise scaled features shape: {pw_scaled_features.shape}")
    
    return True


if __name__ == "__main__":
    test_qal() 