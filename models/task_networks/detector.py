"""
Object Detection Networks for Task-Aware Video Compression

This module contains detector networks that can be used for task-aware
video compression. The detectors predict object locations and classes
directly from the features of compressed video frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyDetector(nn.Module):
    """
    A simple detector model that maps features to detection outputs.
    
    This dummy detector uses a simple convolutional architecture to 
    transform feature inputs into detection outputs (e.g., bounding box
    coordinates or class probabilities).
    """
    def __init__(self, in_channels=128, hidden_channels=64, num_classes=80):
        """
        Initialize the dummy detector.
        
        Args:
            in_channels: Number of input channels (default: 128)
            hidden_channels: Number of hidden channels (default: 64)
            num_classes: Number of output classes (default: 80, COCO dataset)
        """
        super(DummyDetector, self).__init__()
        
        # Simple convolutional architecture
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass of the dummy detector.
        
        Args:
            x: Input feature tensor of shape [B, C, H, W]
            
        Returns:
            Detection output (class heatmap or bounding box predictions)
        """
        # Apply first convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        
        # Apply second convolutional layer to get outputs
        x = self.conv2(x)
        
        return x


class ObjectDetector(nn.Module):
    """
    A more complete object detector with classification and bounding box prediction.
    
    This detector outputs both class probabilities and bounding box coordinates
    for each spatial location, making it suitable for anchor-free detection.
    """
    def __init__(self, in_channels=128, hidden_channels=64, num_classes=80):
        """
        Initialize the object detector.
        
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
            num_classes: Number of object classes to detect
        """
        super(ObjectDetector, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.class_head = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        
        # Bounding box regression head (4 values per location: x, y, w, h)
        self.bbox_head = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        
        # Object confidence head
        self.conf_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass of the object detector.
        
        Args:
            x: Input feature tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary with detection outputs:
            - 'class_logits': Class prediction logits
            - 'bbox_pred': Bounding box predictions
            - 'confidence': Object confidence scores
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Generate predictions
        class_logits = self.class_head(features)
        bbox_pred = self.bbox_head(features)
        confidence = torch.sigmoid(self.conf_head(features))
        
        return {
            'class_logits': class_logits,      # [B, num_classes, H, W]
            'bbox_pred': bbox_pred,            # [B, 4, H, W]
            'confidence': confidence           # [B, 1, H, W]
        }


class VideoObjectDetector(nn.Module):
    """
    Object detector that operates on video frames.
    
    This detector can leverage temporal information across video frames
    to improve detection accuracy and consistency.
    """
    def __init__(self, in_channels=128, hidden_channels=64, num_classes=80, time_steps=5):
        """
        Initialize the video object detector.
        
        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels
            num_classes: Number of object classes to detect
            time_steps: Number of frames to process at once
        """
        super(VideoObjectDetector, self).__init__()
        
        self.time_steps = time_steps
        
        # Temporal aggregation (simple average pooling across time)
        self.temporal_pool = nn.AdaptiveAvgPool3d((hidden_channels, None, None))
        
        # Base detector operating on aggregated features
        self.detector = ObjectDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes
        )
        
    def forward(self, x):
        """
        Forward pass of the video object detector.
        
        Args:
            x: Input feature tensor, can be:
               - Single frame: [B, C, H, W]
               - Multiple frames: [B, C, T, H, W]
            
        Returns:
            Dictionary with detection outputs (same as ObjectDetector)
        """
        # Handle both single frame and multi-frame inputs
        if x.dim() == 4:  # [B, C, H, W]
            # Single-frame input
            return self.detector(x)
        
        elif x.dim() == 5:  # [B, C, T, H, W]
            # Multi-frame input
            batch_size, channels, time_steps, height, width = x.shape
            
            # If only one time step, process as single frame
            if time_steps == 1:
                return self.detector(x.squeeze(2))
            
            # Check if we need to adapt to a different number of time steps
            if time_steps != self.time_steps and time_steps > 1:
                # Simple approach: average over time dimension
                x_pooled = x.mean(dim=2)
            else:
                # Take center frame when handling exactly 'time_steps' frames
                center_idx = time_steps // 2
                x_pooled = x[:, :, center_idx]
            
            # Pass pooled features to the detector
            return self.detector(x_pooled)
        
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}, expected 4 or 5") 