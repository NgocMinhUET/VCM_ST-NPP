#!/usr/bin/env python
"""
Simple test script to verify if HevcCodec can be imported correctly.
"""

import os
import sys

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.codec_utils import HevcCodec
    codec = HevcCodec()
    print("Successfully imported HevcCodec!")
    print(f"Codec configuration: YUV format: {codec.yuv_format}, Preset: {codec.preset}")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Other error: {e}") 