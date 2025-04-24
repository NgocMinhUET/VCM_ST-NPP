import torch
import torch.nn as nn
from models.combined_model import TaskAwareVideoProcessor
from models.st_npp import STNPP
from models.qal import QAL
from models.proxy_codec import ProxyCodec
from models.task_networks.detector import VideoObjectDetector

def test_model_forward():
    """Test model forward pass with a single batch of data"""
    print("Testing model forward pass...")
    
    # Parameters
    batch_size = 1
    channels = 3
    time_steps = 4
    height = 128
    width = 128
    feature_channels = 32
    latent_channels = 16
    
    # Create random input
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    # Create model components
    st_npp = STNPP(
        channels=channels,
        latent_channels=feature_channels,
        time_steps=time_steps
    )
    
    qal = QAL(
        channels=feature_channels,
        qp_levels=8
    )
    
    proxy_codec = ProxyCodec(
        channels=latent_channels,
        num_qp_levels=8,
        artifact_simulation=True
    )
    
    task_network = VideoObjectDetector(
        in_channels=feature_channels,
        num_classes=80
    )
    
    # Create combined model
    model = TaskAwareVideoProcessor(
        st_npp=st_npp,
        qal=qal,
        proxy_codec=proxy_codec,
        task_network=task_network,
        task_type="detection",
        use_quantization=True
    )
    
    # Forward pass
    try:
        print(f"Input shape: {x.shape}")
        outputs = model(x)
        print("Forward pass successful!")
        print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
        print(f"Task outputs: {outputs['task_outputs'].keys()}")
        return True
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return False

if __name__ == "__main__":
    success = test_model_forward()
    if success:
        print("\nModel is working correctly!")
    else:
        print("\nModel is still not working. Please check the error messages above.") 