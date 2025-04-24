"""
Test script to verify shape handling in metric calculations.
Run this script to check if the fixes to PSNR, SSIM, and BPP calculations work correctly.
"""

import torch
import torch.nn.functional as F
from utils.metric_utils import compute_psnr, compute_ssim, compute_bpp
from utils.loss_utils import compute_total_loss

def test_metric_functions():
    """Test the robustness of metric functions with various input shapes"""
    print("\n===== Testing Metric Functions with Various Input Shapes =====")
    
    # Create test tensors of various shapes
    batch_size = 2
    time_steps = 5
    channels = 3
    height = 64
    width = 64
    
    # Standard case: [B, C, H, W]
    standard_a = torch.rand(batch_size, channels, height, width)
    standard_b = torch.rand(batch_size, channels, height, width)
    
    # Video sequence case: [B, T, C, H, W]
    video_a = torch.rand(batch_size, time_steps, channels, height, width)
    video_b = torch.rand(batch_size, time_steps, channels, height, width)
    
    # Shape mismatch case: different spatial dimensions
    mismatch_a = torch.rand(batch_size, channels, height, width)
    mismatch_b = torch.rand(batch_size, channels, height//2, width//2)
    
    # Shape mismatch case: different batch sizes
    batch_mismatch_a = torch.rand(batch_size, channels, height, width)
    batch_mismatch_b = torch.rand(batch_size+1, channels, height, width)
    
    # Bitrate tensor
    bitrate = torch.tensor(0.15)
    
    # List of test cases
    test_cases = [
        ("Standard case [B, C, H, W]", standard_a, standard_b),
        ("Video case [B, T, C, H, W]", video_a, video_b),
        ("Spatial mismatch case", mismatch_a, mismatch_b),
        ("Batch mismatch case", batch_mismatch_a, batch_mismatch_b)
    ]
    
    # Run tests
    for name, a, b in test_cases:
        print(f"\n----- {name} -----")
        print(f"Input A shape: {a.shape}")
        print(f"Input B shape: {b.shape}")
        
        try:
            # Test PSNR
            print("\nTesting PSNR...")
            psnr = compute_psnr(a, b)
            print(f"PSNR result: {psnr.item():.4f}")
            
            # Test SSIM
            print("\nTesting SSIM...")
            ssim = compute_ssim(a, b)
            print(f"SSIM result: {ssim.item():.4f}")
            
            # Test BPP
            print("\nTesting BPP...")
            bpp = compute_bpp(bitrate, a)
            print(f"BPP result: {bpp.item():.4f}")
            
            # Test compute_total_loss
            print("\nTesting compute_total_loss...")
            # Create dummy task output and labels
            if len(a.shape) == 5:  # video
                task_out = torch.rand(batch_size, 1, height, width)  # single class segmentation
                labels = torch.randint(0, 2, (batch_size, 1, height, width)).float()  # binary mask
            else:
                task_out = torch.rand(batch_size, 1, a.shape[2], a.shape[3])  # single class segmentation
                labels = torch.randint(0, 2, (batch_size, 1, a.shape[2], a.shape[3])).float()  # binary mask
                
            loss = compute_total_loss(
                task_out=task_out,
                labels=labels,
                recon=b,
                raw=a,
                bitrate=bitrate,
                task_weight=1.0,
                recon_weight=1.0,
                bitrate_weight=0.01,
                task_type='segmentation'
            )
            print(f"Total loss result: {loss.item():.4f}")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            
    print("\n===== Tests Completed =====")

def test_edge_cases():
    """Test extreme edge cases that might cause issues"""
    print("\n===== Testing Edge Cases =====")
    
    # Edge cases to test
    edge_cases = [
        ("Empty tensor", torch.empty(0, 3, 64, 64), torch.empty(0, 3, 64, 64)),
        ("Very large values", torch.rand(2, 3, 64, 64) * 1000, torch.rand(2, 3, 64, 64) * 1000),
        ("Very small values", torch.rand(2, 3, 64, 64) * 1e-10, torch.rand(2, 3, 64, 64) * 1e-10),
        ("NaN values", torch.full((2, 3, 64, 64), float('nan')), torch.rand(2, 3, 64, 64)),
        ("Inf values", torch.full((2, 3, 64, 64), float('inf')), torch.rand(2, 3, 64, 64)),
        ("Same content (identical images)", torch.rand(2, 3, 64, 64), None)  # Will copy A to B in the test
    ]
    
    for name, a, b in edge_cases:
        print(f"\n----- {name} -----")
        
        # For identical images test
        if b is None:
            b = a.clone()
        
        try:
            # Test PSNR
            print("\nTesting PSNR...")
            psnr = compute_psnr(a, b)
            print(f"PSNR result: {psnr.item() if isinstance(psnr, torch.Tensor) else psnr}")
            
            # Test SSIM
            print("\nTesting SSIM...")
            ssim = compute_ssim(a, b)
            print(f"SSIM result: {ssim.item() if isinstance(ssim, torch.Tensor) else ssim}")
            
            # Test BPP with scalar
            print("\nTesting BPP with scalar...")
            bpp = compute_bpp(0.2, a)
            print(f"BPP result: {bpp.item() if isinstance(bpp, torch.Tensor) else bpp}")
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
    
    print("\n===== Edge Case Tests Completed =====")

if __name__ == "__main__":
    # Run tests
    test_metric_functions()
    test_edge_cases()
    
    print("\nAll tests completed. Check the output for any errors.") 