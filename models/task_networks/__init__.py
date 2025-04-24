"""
Task network implementations for task-aware video compression.

This module provides neural network models for various downstream tasks:
- Object detection
- Semantic segmentation
- Object tracking
"""

from models.task_networks.detector import ObjectDetector, VideoObjectDetector, DummyDetector
from models.task_networks.segmenter import SegmentationNet, VideoSegmentationNet, AttentionUNet, DeepLabV3Plus, DummySegmenter
from models.task_networks.tracker import ObjectTracker, VideoObjectTracker, DummyTracker

# Dictionary mapping task names to their respective model classes
TASK_MODELS = {
    'detection': VideoObjectDetector,
    'segmentation': VideoSegmentationNet,
    'tracking': VideoObjectTracker
}

def get_task_model(task_name, **kwargs):
    """
    Factory function to get a task model by name.
    
    Args:
        task_name: Name of the task ('detection', 'segmentation', 'tracking')
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Task model instance
    """
    if task_name not in TASK_MODELS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_MODELS.keys())}")
    
    return TASK_MODELS[task_name](**kwargs)

"""
Task networks package initialization.
""" 