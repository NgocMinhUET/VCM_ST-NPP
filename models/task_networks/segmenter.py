"""
Semantic Segmentation model for task-aware video processing.

This module implements a U-Net inspired segmentation model designed to work
with the output of the ST-NPP video preprocessing module. It performs
pixel-wise classification for semantic segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """Encoder block with downsampling and double convolution"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and double convolution"""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class SegmentationNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    This model is designed to work with outputs from the ST-NPP module
    and serves as a downstream task for task-aware video compression.
    """
    def __init__(self, in_channels=3, num_classes=21):
        super(SegmentationNet, self).__init__()
        
        # Initial convolution
        self.inc = ConvBlock(in_channels, 64)
        
        # Encoder pathway
        self.enc1 = EncoderBlock(64, 128)
        self.enc2 = EncoderBlock(128, 256)
        self.enc3 = EncoderBlock(256, 512)
        self.enc4 = EncoderBlock(512, 1024)
        
        # Decoder pathway
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        
        # Final convolution
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the segmentation model
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Encoder pathway with skip connections
        x1 = self.inc(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        
        # Decoder pathway using skip connections
        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        x = self.outc(x)
        
        return x
    
    def compute_loss(self, predictions, targets, class_weights=None):
        """
        Compute segmentation loss
        
        Args:
            predictions: Predicted segmentation logits [B, C, H, W]
            targets: Ground truth segmentation masks [B, H, W]
            class_weights: Optional weights for each class to handle class imbalance
            
        Returns:
            Segmentation loss (cross-entropy)
        """
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=predictions.device)
            
        loss = F.cross_entropy(predictions, targets, weight=class_weights, reduction='mean')
        return loss


class AttentionBlock(nn.Module):
    """Attention gate for focusing on relevant features"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net for improved segmentation with attention gates.
    
    This model adds attention mechanisms to focus on relevant features
    from the encoder pathway during decoding.
    """
    def __init__(self, in_channels=3, num_classes=21):
        super(AttentionUNet, self).__init__()
        
        # Initial convolution
        self.inc = ConvBlock(in_channels, 64)
        
        # Encoder pathway
        self.enc1 = EncoderBlock(64, 128)
        self.enc2 = EncoderBlock(128, 256)
        self.enc3 = EncoderBlock(256, 512)
        self.enc4 = EncoderBlock(512, 1024)
        
        # Attention gates
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        
        # Decoder pathway
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        
        # Final convolution
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through the attention U-Net model
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Encoder pathway with skip connections
        x1 = self.inc(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        
        # Decoder pathway with attention gates
        x4_att = self.att1(g=x5, x=x4)
        x = self.dec1(x5, x4_att)
        
        x3_att = self.att2(g=x, x=x3)
        x = self.dec2(x, x3_att)
        
        x2_att = self.att3(g=x, x=x2)
        x = self.dec3(x, x2_att)
        
        x1_att = self.att4(g=x, x=x1)
        x = self.dec4(x, x1_att)
        
        x = self.outc(x)
        
        return x


class VideoSegmentationNet(nn.Module):
    """
    Video semantic segmentation model.
    
    This model extends the frame-level segmenter to process video sequences
    by incorporating temporal information.
    """
    def __init__(self, in_channels=3, num_classes=21, use_attention=True):
        super(VideoSegmentationNet, self).__init__()
        
        # Choose base segmentation model
        if use_attention:
            self.segmenter = AttentionUNet(in_channels, num_classes)
        else:
            self.segmenter = SegmentationNet(in_channels, num_classes)
        
        # Temporal convolution module
        self.temporal_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        
        # Temporal consistency module
        self.consistency_conv = nn.Conv3d(num_classes, num_classes, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
    def forward(self, x):
        """
        Forward pass for video input
        
        Args:
            x: Input video tensor [B, C, T, H, W]
            
        Returns:
            Segmentation logits for each frame [B, T, num_classes, H, W]
        """
        batch_size, channels, time_steps, height, width = x.shape
        
        # Apply temporal convolution
        x = self.temporal_conv(x)
        
        # Process each frame
        segmentations = []
        for t in range(time_steps):
            frame = x[:, :, t]
            seg = self.segmenter(frame)
            segmentations.append(seg)
        
        # Stack frame segmentations
        segmentations = torch.stack(segmentations, dim=2)  # [B, num_classes, T, H, W]
        
        # Apply temporal consistency
        segmentations = self.consistency_conv(segmentations)
        
        # Permute to [B, T, num_classes, H, W] for easier handling
        segmentations = segmentations.permute(0, 2, 1, 3, 4)
        
        return segmentations
    
    def process_video_batch(self, x):
        """
        Process an entire video batch at once for more efficient computation
        
        Args:
            x: Input video tensor [B, C, T, H, W]
            
        Returns:
            Segmentation logits for the video [B, T, num_classes, H, W]
        """
        batch_size, channels, time_steps, height, width = x.shape
        
        # Apply temporal convolution
        x = self.temporal_conv(x)
        
        # Reshape for batch processing: [B, C, T, H, W] -> [B*T, C, H, W]
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous()
        x_reshaped = x_reshaped.view(-1, channels, height, width)
        
        # Process all frames at once
        segmentations = self.segmenter(x_reshaped)
        
        # Get the number of classes
        num_classes = segmentations.size(1)
        
        # Reshape back to video format: [B*T, num_classes, H, W] -> [B, T, num_classes, H, W]
        segmentations = segmentations.view(batch_size, time_steps, num_classes, height, width)
        
        # Apply temporal consistency (need to permute dimensions)
        segmentations_temporal = segmentations.permute(0, 2, 1, 3, 4)  # [B, num_classes, T, H, W]
        segmentations_temporal = self.consistency_conv(segmentations_temporal)
        segmentations = segmentations_temporal.permute(0, 2, 1, 3, 4)  # [B, T, num_classes, H, W]
        
        return segmentations
    
    def compute_loss(self, predictions, targets, class_weights=None):
        """
        Compute segmentation loss for video
        
        Args:
            predictions: Predicted segmentation logits [B, T, num_classes, H, W]
            targets: Ground truth segmentation masks [B, T, H, W]
            class_weights: Optional weights for each class
            
        Returns:
            Segmentation loss (cross-entropy)
        """
        batch_size, time_steps = predictions.shape[:2]
        
        # Reshape for loss computation
        pred_reshaped = predictions.view(batch_size * time_steps, predictions.shape[2], 
                                         predictions.shape[3], predictions.shape[4])
        
        target_reshaped = targets.view(batch_size * time_steps, targets.shape[2], 
                                       targets.shape[3])
        
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, device=predictions.device)
            
        loss = F.cross_entropy(pred_reshaped, target_reshaped, weight=class_weights, reduction='mean')
        return loss


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture for semantic segmentation with atrous convolutions.
    
    This model uses atrous spatial pyramid pooling (ASPP) to capture multi-scale
    context and decoder module to refine segmentation results along object boundaries.
    """
    def __init__(self, in_channels=3, num_classes=21):
        super(DeepLabV3Plus, self).__init__()
        
        # Use ResNet-like backbone (simplified for this implementation)
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 512),
        )
        
        # Atrous Spatial Pyramid Pooling
        self.aspp = ASPP(512, 256)
        
        # Low-level features
        self.low_level_features = nn.Sequential(
            nn.Conv2d(128, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            ConvBlock(304, 256),  # 256 from ASPP + 48 from low-level features
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        size = x.size()[2:]
        
        # Get low-level features from early in the backbone
        x_low = self.backbone[0:4](x)
        
        # Get high-level features from later in the backbone
        x = self.backbone[4:](x_low)
        
        # Apply ASPP
        x = self.aspp(x)
        
        # Upsample high-level features
        x = F.interpolate(x, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        
        # Process low-level features
        x_low = self.low_level_features(x_low)
        
        # Concatenate low-level and high-level features
        x = torch.cat((x, x_low), dim=1)
        
        # Apply decoder
        x = self.decoder(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module for DeepLabV3+
    
    Applies multiple atrous convolutions with different dilation rates
    to capture multi-scale context.
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with dilation=6
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with dilation=12
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with dilation=18
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Combine all branches
        self.combine = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.size()[2:]
        
        # Apply different atrous convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        
        # Apply global pooling and upsample
        pool = self.pool(x)
        pool = F.interpolate(pool, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate all features
        x = torch.cat((conv1, conv2, conv3, conv4, pool), dim=1)
        
        # Combine with 1x1 convolution
        x = self.combine(x)
        
        return x


class DummySegmenter(nn.Module):
    def __init__(self, in_channels=128, num_classes=21):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)


# Test code
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    in_channels = 3
    time_steps = 8
    height = 256
    width = 256
    num_classes = 21  # PASCAL VOC has 21 classes
    
    # Create model
    model = VideoSegmentationNet(in_channels, num_classes, use_attention=True)
    
    # Create random input
    x = torch.randn(batch_size, in_channels, time_steps, height, width)
    
    # Forward pass
    segmentations = model.process_video_batch(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Segmentation output shape: {segmentations.shape}")
    
    # Test DeepLabV3+
    deeplab = DeepLabV3Plus(in_channels, num_classes)
    y = torch.randn(batch_size, in_channels, height, width)
    output = deeplab(y)
    print(f"DeepLabV3+ output shape: {output.shape}") 