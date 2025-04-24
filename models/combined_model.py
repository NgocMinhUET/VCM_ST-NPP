"""
Task-Aware Video Processor

This module implements the complete task-aware video processing pipeline
that integrates all components of the system:
1. STNPP - Spatio-Temporal Neural Preprocessing
2. QAL - Quantization Adaptation Layer
3. ProxyCodec - Differentiable proxy for video codecs
4. Task Networks - Downstream computer vision tasks

The combined model optimizes for both reconstruction quality and 
task performance simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.st_npp import STNPP
from models.qal import QAL, QALModule
from models.proxy_codec import ProxyCodec
from models.task_networks.detector import DummyDetector, VideoObjectDetector
from models.task_networks.segmenter import DummySegmenter, VideoSegmentationNet
from models.task_networks.tracker import DummyTracker, VideoObjectTracker


class TaskAwareVideoProcessor(nn.Module):
    """
    Complete Task-Aware Video Processing model that optimizes for both
    reconstruction quality and task performance.
    """
    def __init__(
        self,
        task_type='detection',
        channels=3,
        hidden_channels=64,
        num_classes=80,
        time_steps=5,
        qp_levels=51
    ):
        """
        Initialize the task-aware video processor.
        
        Args:
            task_type: Type of downstream task ('detection', 'segmentation', 'tracking')
            channels: Number of input/output channels (default: 3 for RGB)
            hidden_channels: Number of channels in hidden layers
            num_classes: Number of classes for the task
            time_steps: Number of frames to process at once
            qp_levels: Number of QP levels supported
        """
        super(TaskAwareVideoProcessor, self).__init__()
        
        self.task_type = task_type
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.time_steps = time_steps
        
        # Spatio-Temporal Neural Preprocessing
        self.stnpp = STNPP(
            channels=channels,
            hidden_channels=hidden_channels,
            temporal_kernel_size=time_steps
        )
        
        # Quantization Adaptation Layer
        self.qal = QALModule(
            channels=hidden_channels,
            hidden_size=hidden_channels,
            qp_levels=qp_levels,
            temporal_kernel_size=time_steps
        )
        
        # Proxy Codec
        self.proxy_codec = ProxyCodec(
            channels=channels,
            latent_channels=hidden_channels,
            block_size=8
        )
        
        # Task head based on task type
        if task_type == 'detection':
            self.task_head = VideoObjectDetector(
                in_channels=channels,
                hidden_channels=hidden_channels,
                num_classes=num_classes,
                time_steps=time_steps
            )
        elif task_type == 'segmentation':
            self.task_head = VideoSegmentationNet(
                in_channels=channels,
                hidden_channels=hidden_channels,
                num_classes=num_classes,
                time_steps=time_steps
            )
        elif task_type == 'tracking':
            self.task_head = VideoObjectTracker(
                in_channels=channels,
                hidden_channels=hidden_channels,
                num_classes=num_classes,
                time_steps=time_steps
            )
        else:
            # Fallback to dummy detector
            self.task_head = DummyDetector(
                in_channels=channels,
                hidden_channels=hidden_channels,
                num_classes=num_classes
            )
            
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize model weights properly
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, frames, qp=None):
        """
        Forward pass of the task-aware video processor.
        
        Args:
            frames: Input video frames of shape [B, T, C, H, W] or [B, C, T, H, W]
            qp: Quantization parameter (0-51, higher = more compression)
            
        Returns:
            Dictionary containing:
            - 'reconstructed': Reconstructed video frames
            - 'task_output': Output from task-specific head
            - 'bpp': Estimated bits per pixel
        """
        # Ensure input is in the format [B, T, C, H, W]
        # This is more natural for video data
        if frames.size(1) == self.channels and frames.size(2) == self.time_steps:
            # Input is [B, C, T, H, W], transpose to [B, T, C, H, W]
            frames = frames.permute(0, 2, 1, 3, 4)
            
        # Apply STNPP
        preprocessed = self.stnpp(frames)
        
        # Apply QAL modulation
        modulated = self.qal(preprocessed, qp)
        
        # Apply proxy codec
        reconstructed, bpp = self.proxy_codec(modulated, qp)
        
        # Apply task head on reconstructed frames
        task_output = self.task_head(reconstructed)
        
        # Return results
        return {
            'reconstructed': reconstructed,
            'task_output': task_output,
            'bpp': bpp
        }


class CombinedModel(nn.Module):
    """
    Combined model that includes a codec and a task network.
    
    Args:
        codec: Codec model for compression and decompression
        task_network: Task-specific neural network (detection, segmentation, tracking)
        seq_length: Number of frames in a video sequence
        task_type: Type of task to perform ('detection', 'segmentation', 'tracking')
        hidden_channels: Number of hidden channels in model components (default: 64)
    """
    
    def __init__(self, codec=None, task_network=None, seq_length=5, task_type='detection', hidden_channels=64):
        super(CombinedModel, self).__init__()
        
        # Store parameters
        self.task_type = task_type
        self.seq_length = seq_length
        self.hidden_channels = hidden_channels
        
        # Initialize codec
        if codec is None:
            from models.proxy_codec import ProxyCodec
            self.codec = ProxyCodec(hidden_channels=hidden_channels)
        else:
            self.codec = codec
        
        # Initialize task network based on task type if not provided
        if task_network is None:
            if task_type == 'detection':
                from models.task_networks.detector import DummyDetector
                self.task_network = DummyDetector(hidden_channels=hidden_channels)
            elif task_type == 'segmentation':
                from models.task_networks.segmenter import DummySegmenter
                self.task_network = DummySegmenter(hidden_channels=hidden_channels)
            elif task_type == 'tracking':
                from models.task_networks.tracker import DummyTracker
                self.task_network = DummyTracker(hidden_channels=hidden_channels)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
        else:
            self.task_network = task_network
    
    def forward(self, x, qp=None):
        """
        Forward pass through the combined model.
        
        Args:
            x: Input tensor of shape [B, S, C, H, W] where S is sequence length
            qp: Quantization parameter (optional)
            
        Returns:
            Dictionary containing:
                - 'reconstructed': Reconstructed frames after codec
                - 'task_output': Output from task network
                - 'bitrate': Estimated bitrate
        """
        # Remember original shape
        original_shape = x.shape
        batch_size, seq_len, channels, height, width = original_shape
        
        # Process with codec (reshape to [B*S, C, H, W] for the codec)
        x_flat = x.reshape(-1, channels, height, width)
        
        # Pass through codec
        reconstructed, bits = self.codec(x_flat, qp)
        
        # Reshape back to [B, S, C, H, W]
        reconstructed = reconstructed.reshape(batch_size, seq_len, channels, height, width)
        
        # Pass reconstructed frames to task network
        task_output = self.task_network(reconstructed)
        
        # Return results as dictionary
        return {
            'reconstructed': reconstructed,
            'task_output': task_output,
            'bitrate': bits
        }


# Test code
if __name__ == "__main__":
    import time
    
    # Create sample input
    batch_size = 2
    time_steps = 5
    channels = 3
    height = 256
    width = 256
    num_classes = 80
    
    # Create input frames [B, T, C, H, W]
    frames = torch.rand(batch_size, time_steps, channels, height, width)
    qp = 30  # Quantization parameter
    
    # Test CombinedModel
    print("Testing CombinedModel...")
    combined_model = CombinedModel(task_type='detection')
    
    start_time = time.time()
    result = combined_model(frames, qp)
    elapsed = time.time() - start_time
    
    print(f"Input shape: {frames.shape}")
    print(f"Reconstructed shape: {result['reconstructed'].shape}")
    if isinstance(result['task_output'], dict):
        for k, v in result['task_output'].items():
            print(f"Task output '{k}' shape: {v.shape}")
    else:
        print(f"Task output shape: {result['task_output'].shape}")
    print(f"Bitrate: {result['bitrate'].mean().item():.4f} bpp")
    print(f"Forward pass time: {elapsed:.4f} seconds")
    
    # Test TaskAwareVideoProcessor
    print("\nTesting TaskAwareVideoProcessor...")
    full_model = TaskAwareVideoProcessor(
        task_type='detection',
        channels=channels,
        hidden_channels=64,
        num_classes=num_classes,
        time_steps=time_steps
    )
    
    start_time = time.time()
    result = full_model(frames, qp)
    elapsed = time.time() - start_time
    
    print(f"Input shape: {frames.shape}")
    print(f"Reconstructed shape: {result['reconstructed'].shape}")
    if isinstance(result['task_output'], dict):
        for k, v in result['task_output'].items():
            print(f"Task output '{k}' shape: {v.shape}")
    else:
        print(f"Task output shape: {result['task_output'].shape}")
    print(f"Bitrate: {result['bpp'].mean().item():.4f} bpp")
    print(f"Forward pass time: {elapsed:.4f} seconds")
 