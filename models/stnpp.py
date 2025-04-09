"""
Spatio-Temporal Neural Preprocessing (ST-NPP) Model

This module implements the Spatio-Temporal Neural Preprocessing module,
which consists of a spatial branch, a temporal branch, and a fusion module.
The ST-NPP module is designed to reduce spatial-temporal redundancy
before video compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional, List, Dict, Any


class SpatialBranch(nn.Module):
    """
    Spatial branch of the ST-NPP module using a pre-trained CNN backbone.
    
    This branch extracts spatial features from individual frames.
    """
    
    def __init__(self, backbone_name: str = 'resnet50', pretrained: bool = True, 
                 output_channels: int = 128):
        super(SpatialBranch, self).__init__()
        
        # Load the backbone model
        if backbone_name == 'resnet50':
            weights = 'IMAGENET1K_V1' if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_channels = 2048
        elif backbone_name == 'resnet34':
            weights = 'IMAGENET1K_V1' if pretrained else None
            backbone = models.resnet34(weights=weights)
            feature_channels = 512
        elif backbone_name == 'efficientnet_b4':
            weights = 'IMAGENET1K_V1' if pretrained else None
            backbone = models.efficientnet_b4(weights=weights)
            feature_channels = 1792
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Extract convolutional layers up to the final pooling layer
        if 'resnet' in backbone_name:
            self.backbone_features = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            )
        else:  # EfficientNet
            self.backbone_features = backbone.features
        
        # Projection layer to reduce channel dimensions
        self.projection = nn.Conv2d(feature_channels, output_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spatial branch.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, C, H, W)
        
        Returns:
            Spatial features of shape (B, T, output_channels, H/4, W/4) or (B, output_channels, H/4, W/4)
        """
        orig_shape = x.shape
        
        # Handle both 4D and 5D inputs
        if len(orig_shape) == 5:
            # Input is (B, T, C, H, W), process each frame independently
            B, T, C, H, W = orig_shape
            x = x.reshape(B * T, C, H, W)
            process_sequence = True
        else:
            process_sequence = False
        
        # Extract features
        features = self.backbone_features(x)
        
        # Project to desired channel dimensions
        features = self.projection(features)
        
        # Reshape back to sequence if needed
        if process_sequence:
            features = features.reshape(B, T, *features.shape[1:])
        
        return features


class Conv3DBlock(nn.Module):
    """A 3D convolutional block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super(Conv3DBlock, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TemporalBranch3DCNN(nn.Module):
    """
    Temporal branch using 3D CNN for the ST-NPP module.
    
    This branch captures temporal dependencies across frames.
    """
    
    def __init__(self, input_channels: int = 3, output_channels: int = 128, 
                 base_channels: int = 64):
        super(TemporalBranch3DCNN, self).__init__()
        
        self.conv1 = Conv3DBlock(input_channels, base_channels)
        self.conv2 = Conv3DBlock(base_channels, base_channels * 2)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3 = Conv3DBlock(base_channels * 2, output_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the temporal branch.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Temporal features of shape (B, output_channels, T, H/4, W/4)
        """
        x = self.conv1(x)          # (B, 64, T, H, W)
        x = self.conv2(x)          # (B, 128, T, H, W)
        x = self.pool(x)           # (B, 128, T, H/2, W/2)
        x = self.conv3(x)          # (B, 128, T, H/2, W/2)
        
        return x


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for temporal modeling.
    """
    
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Input gate, forget gate, cell gate, output gate
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 
            hidden_channels * 4, 
            kernel_size=kernel_size,
            padding=self.padding
        )
        
    def forward(self, input_tensor: torch.Tensor, 
                hidden_states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single time step.
        
        Args:
            input_tensor: Input tensor of shape (B, C, H, W)
            hidden_states: Tuple of (hidden_state, cell_state), each of shape (B, hidden_channels, H, W)
            
        Returns:
            Tuple of (new_hidden_state, new_cell_state)
        """
        h_prev, c_prev = hidden_states
        
        # Concatenate input and previous hidden state
        combined = torch.cat([input_tensor, h_prev], dim=1)
        
        # Compute gates
        gates = self.conv(combined)
        
        # Split gates
        i, f, c, o = torch.split(gates, self.hidden_channels, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        c = torch.tanh(c)     # Cell gate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state
        c_next = f * c_prev + i * c
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states with zeros."""
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        )


class TemporalBranchConvLSTM(nn.Module):
    """
    Temporal branch using ConvLSTM for the ST-NPP module.
    
    This branch captures temporal dependencies across frames.
    """
    
    def __init__(self, input_channels: int = 3, output_channels: int = 128, 
                 hidden_channels: int = 64):
        super(TemporalBranchConvLSTM, self).__init__()
        
        # Initial 2D convolution to process input frames
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduce spatial dimensions
        )
        
        # ConvLSTM for temporal modeling
        self.convlstm = ConvLSTMCell(hidden_channels, hidden_channels * 2)
        
        # Final convolution to produce output features
        self.conv_out = nn.Conv2d(hidden_channels * 2, output_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the temporal branch.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Temporal features of shape (B, output_channels, T, H/4, W/4)
        """
        # Get dimensions
        B, C, T, H, W = x.shape
        
        # Process each frame with initial convolution
        x_processed = []
        for t in range(T):
            x_t = x[:, :, t]  # (B, C, H, W)
            x_t = self.conv_in(x_t)  # (B, hidden_channels, H/2, W/2)
            x_processed.append(x_t)
        
        # Initialize hidden states
        h, c = self.convlstm.init_hidden(B, H // 2, W // 2)
        
        # Process sequence with ConvLSTM
        outputs = []
        for t in range(T):
            h, c = self.convlstm(x_processed[t], (h, c))
            outputs.append(h)
        
        # Stack outputs along temporal dimension
        outputs = torch.stack(outputs, dim=2)  # (B, hidden_channels*2, T, H/2, W/2)
        
        # Apply final convolution to each timestep
        out_features = []
        for t in range(T):
            out_t = self.conv_out(outputs[:, :, t])  # (B, output_channels, H/2, W/2)
            out_features.append(out_t)
        
        # Stack outputs along temporal dimension
        out_features = torch.stack(out_features, dim=2)  # (B, output_channels, T, H/2, W/2)
        
        return out_features


class ConcatenationFusion(nn.Module):
    """
    Fusion module using concatenation.
    
    This module fuses spatial and temporal features through concatenation.
    """
    
    def __init__(self, spatial_channels: int = 128, temporal_channels: int = 128, 
                 output_channels: int = 128):
        super(ConcatenationFusion, self).__init__()
        
        # Compute input channels after concatenation
        total_channels = spatial_channels + temporal_channels
        
        # 3D convolution to reduce channel dimensions
        self.conv = nn.Conv3d(total_channels, output_channels, kernel_size=1)
        
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fusion module.
        
        Args:
            spatial_features: Spatial features of shape (B, T, C_s, H, W) or (B, C_s, H, W)
            temporal_features: Temporal features of shape (B, C_t, T, H, W)
            
        Returns:
            Fused features of shape (B, output_channels, T, H, W)
        """
        # Debugging: Print the shapes of input tensors
        print(f"DEBUG - spatial_features shape: {spatial_features.shape}")
        print(f"DEBUG - temporal_features shape: {temporal_features.shape}")
        
        # Handle case where spatial features are for a single frame
        if len(spatial_features.shape) == 4:
            B, C, H, W = spatial_features.shape
            _, C_t, T, Ht, Wt = temporal_features.shape
            spatial_features = spatial_features.unsqueeze(1).expand(-1, T, -1, -1, -1)
            print(f"DEBUG - expanded spatial_features shape: {spatial_features.shape}")
        
        # Get dimensions from temporal features
        B, C_t, T, H, W = temporal_features.shape
        
        # Reshape spatial features to match temporal dimensions
        if len(spatial_features.shape) == 5:
            B_s, T_s, C_s, H_s, W_s = spatial_features.shape
            
            # Ensure time dimension matches
            if T_s != T:
                # If time dimensions don't match, resize to match temporal features
                spatial_features = F.interpolate(
                    spatial_features.permute(0, 2, 1, 3, 4),  # (B, C, T, H, W)
                    size=(T, H_s, W_s),
                    mode='trilinear',
                    align_corners=False
                )
                print(f"DEBUG - interpolated spatial_features shape: {spatial_features.shape}")
            else:
                # Just permute to match the format
                spatial_features = spatial_features.permute(0, 2, 1, 3, 4)  # (B, C_s, T, H, W)
        
        # Final debug print before concatenation
        print(f"DEBUG - final spatial_features shape: {spatial_features.shape}")
        print(f"DEBUG - final temporal_features shape: {temporal_features.shape}")
        
        # Ensure spatial dimensions match
        if spatial_features.shape[3:] != temporal_features.shape[3:]:
            spatial_features = F.interpolate(
                spatial_features,
                size=temporal_features.shape[2:],  # (T, H, W)
                mode='trilinear',
                align_corners=False
            )
            print(f"DEBUG - resized spatial_features shape: {spatial_features.shape}")
        
        # Concatenate along channel dimension
        fused = torch.cat([spatial_features, temporal_features], dim=1)
        print(f"DEBUG - fused shape: {fused.shape}")
        
        # Apply 3D convolution
        fused = self.conv(fused)
        
        return fused


class AttentionFusion(nn.Module):
    """
    Fusion module using attention mechanism.
    
    This module fuses spatial and temporal features through attention.
    """
    
    def __init__(self, spatial_channels: int = 128, temporal_channels: int = 128, 
                 output_channels: int = 128):
        super(AttentionFusion, self).__init__()
        
        # Query, key, value projections
        self.query_proj = nn.Conv3d(spatial_channels, output_channels, kernel_size=1)
        self.key_proj = nn.Conv3d(temporal_channels, output_channels, kernel_size=1)
        self.value_proj = nn.Conv3d(temporal_channels, output_channels, kernel_size=1)
        
        # Output projection
        self.output_proj = nn.Conv3d(output_channels, output_channels, kernel_size=1)
        
        # Scaling factor for attention
        self.scale = output_channels ** -0.5
        
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fusion module.
        
        Args:
            spatial_features: Spatial features of shape (B, T, C_s, H, W) or (B, C_s, H, W)
            temporal_features: Temporal features of shape (B, C_t, T, H, W)
            
        Returns:
            Fused features of shape (B, output_channels, T, H, W)
        """
        # Handle case where spatial features are for a single frame
        if len(spatial_features.shape) == 4:
            B, C, H, W = spatial_features.shape
            _, _, T, _, _ = temporal_features.shape
            spatial_features = spatial_features.unsqueeze(1).expand(-1, T, -1, -1, -1)
            
        # Reshape spatial features to match temporal dimensions
        B, T, C_s, H, W = spatial_features.shape
        spatial_features = spatial_features.permute(0, 2, 1, 3, 4)  # (B, C_s, T, H, W)
        
        # Project to query, key, value
        query = self.query_proj(spatial_features)
        key = self.key_proj(temporal_features)
        value = self.value_proj(temporal_features)
        
        # Reshape for attention
        Q = query.reshape(B, -1, T * H * W).permute(0, 2, 1)  # (B, T*H*W, C)
        K = key.reshape(B, -1, T * H * W)  # (B, C, T*H*W)
        V = value.reshape(B, -1, T * H * W).permute(0, 2, 1)  # (B, T*H*W, C)
        
        # Compute attention weights
        attention = torch.bmm(Q, K) * self.scale  # (B, T*H*W, T*H*W)
        attention = F.softmax(attention, dim=2)
        
        # Apply attention
        out = torch.bmm(attention, V)  # (B, T*H*W, C)
        out = out.permute(0, 2, 1).reshape(B, -1, T, H, W)  # (B, C, T, H, W)
        
        # Final projection
        out = self.output_proj(out)
        
        return out


class STNPP(nn.Module):
    """
    Spatio-Temporal Neural Preprocessing (ST-NPP) Module.
    
    This module combines spatial and temporal processing for video frames.
    """
    
    def __init__(self, 
                 input_channels: int = 3, 
                 output_channels: int = 128,
                 spatial_backbone: str = 'resnet50', 
                 temporal_model: str = '3dcnn',
                 fusion_type: str = 'concatenation',
                 pretrained: bool = True):
        """
        Initialize the ST-NPP module.
        
        Args:
            input_channels: Number of input channels (e.g., 3 for RGB)
            output_channels: Number of output channels for features
            spatial_backbone: Backbone for spatial branch ('resnet50', 'resnet34', 'efficientnet_b4')
            temporal_model: Model for temporal branch ('3dcnn' or 'convlstm')
            fusion_type: Type of fusion ('concatenation' or 'attention')
            pretrained: Whether to use pretrained weights for the spatial backbone
        """
        super(STNPP, self).__init__()
        
        # Create spatial branch
        self.spatial_branch = SpatialBranch(
            backbone_name=spatial_backbone, 
            pretrained=pretrained,
            output_channels=output_channels
        )
        
        # Create temporal branch
        if temporal_model == '3dcnn':
            self.temporal_branch = TemporalBranch3DCNN(
                input_channels=input_channels,
                output_channels=output_channels
            )
        elif temporal_model == 'convlstm':
            self.temporal_branch = TemporalBranchConvLSTM(
                input_channels=input_channels,
                output_channels=output_channels
            )
        else:
            raise ValueError(f"Unsupported temporal model: {temporal_model}")
        
        # Create fusion module
        if fusion_type == 'concatenation':
            self.fusion = ConcatenationFusion(
                spatial_channels=output_channels,
                temporal_channels=output_channels,
                output_channels=output_channels
            )
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(
                spatial_channels=output_channels,
                temporal_channels=output_channels,
                output_channels=output_channels
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ST-NPP module.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, C, T, H, W)
            
        Returns:
            Processed features of shape (B, output_channels, T, H/4, W/4)
        """
        # Handle different input formats
        if x.shape[1] == 3:  # (B, C, T, H, W) format
            B, C, T, H, W = x.shape
            # Reshape to (B, T, C, H, W) for spatial branch
            x_spatial = x.permute(0, 2, 1, 3, 4)
        else:  # (B, T, C, H, W) format
            B, T, C, H, W = x.shape
            x_spatial = x
            # Reshape to (B, C, T, H, W) for temporal branch
            x = x.permute(0, 2, 1, 3, 4)
            
        # Process through spatial branch
        spatial_features = self.spatial_branch(x_spatial)  # (B, T, C, H/4, W/4)
        
        # Process through temporal branch
        temporal_features = self.temporal_branch(x)  # (B, C, T, H/4, W/4)
        
        # Fuse features
        fused_features = self.fusion(spatial_features, temporal_features)  # (B, C, T, H/4, W/4)
        
        return fused_features


def test_stnpp():
    """Test function to verify the ST-NPP implementation."""
    # Create a random input tensor
    batch_size = 2
    time_steps = 16
    channels = 3
    height = 224
    width = 224
    
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    # Create the ST-NPP module
    stnpp = STNPP(
        input_channels=channels,
        output_channels=128,
        spatial_backbone='resnet34',  # Smaller model for testing
        temporal_model='3dcnn',
        fusion_type='concatenation'
    )
    
    # Forward pass
    features = stnpp(x)
    
    # Print the shapes
    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    
    # Expected output shape: (B, C, T, H/4, W/4)
    expected_shape = (batch_size, 128, time_steps, height // 4, width // 4)
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"
    
    return True


if __name__ == "__main__":
    test_stnpp() 