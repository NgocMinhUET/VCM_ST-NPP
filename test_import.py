#!/usr/bin/env python3
"""
Simple test script to verify if HevcCodec can be imported correctly.
"""

import os
import sys

# Add parent directory to path for imports
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

try:
    from models.stnpp import STNPP
    print("Successfully imported STNPP")
except ImportError as e:
    print(f"Error importing STNPP: {e}")

try:
    from models.qal import QAL
    print("Successfully imported QAL")
except ImportError as e:
    print(f"Error importing QAL: {e}")

try:
    from models.proxy_network import ProxyNetwork
    print("Successfully imported ProxyNetwork")
except ImportError as e:
    print(f"Error importing ProxyNetwork: {e}")

try:
    from scripts.train_stnpp import VideoDataset
    print("Successfully imported VideoDataset from train_stnpp")
except ImportError as e:
    print(f"Error importing VideoDataset: {e}") 