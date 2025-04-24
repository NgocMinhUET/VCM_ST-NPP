import torch
import torch.nn as nn
from models.qal import QAL, QALModule

def test_qal_module():
    print("Testing QALModule...")
    
    # Create a random input tensor
    batch_size = 1
    channels = 32
    height = 64
    width = 64
    x = torch.randn(batch_size, channels, height, width)
    
    # Create QP tensor
    qp = torch.tensor([22] * batch_size)
    
    # Create QALModule
    qal_module = QALModule(channels=channels)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    quantized_x, rate, importance_map = qal_module(x, qp)
    print(f"Output shape: {quantized_x.shape}")
    print(f"Rate: {rate}")
    print(f"Importance map shape: {importance_map.shape}")
    
    print("QALModule test completed successfully!")

def test_qal():
    print("Testing QAL...")
    
    # Create a random input tensor
    batch_size = 1
    channels = 32
    time_steps = 4
    height = 64
    width = 64
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    # Create QP tensor
    qp = torch.tensor([22] * batch_size)
    
    # Create QAL
    qal = QAL(channels=channels)
    
    # Forward pass
    print(f"Input shape: {x.shape}")
    quantized_x, rate, importance_map = qal(x, qp)
    print(f"Output shape: {quantized_x.shape}")
    print(f"Rate: {rate}")
    print(f"Importance map shape: {importance_map.shape}")
    
    print("QAL test completed successfully!")

if __name__ == "__main__":
    test_qal_module()
    print("\n" + "-" * 50 + "\n")
    test_qal() 