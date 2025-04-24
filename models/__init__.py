"""
Models module for Task-Aware Video Processing system.

This module imports the main components of the system:
1. STNPP - Spatio-Temporal Neural Preprocessing
2. QAL - Quantization Adaptation Layer
3. ProxyCodec - Differentiable proxy for video codecs
4. TaskAwareVideoProcessor - Complete end-to-end system
"""

from models.st_npp import STNPP
from models.qal import QAL
from models.proxy_codec import ProxyCodec
from models.combined_model import TaskAwareVideoProcessor

__all__ = ['STNPP', 'QAL', 'ProxyCodec', 'TaskAwareVideoProcessor']

"""
Models package initialization.
""" 