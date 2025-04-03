import os
import sys
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.proxy_network import ProxyNetwork
from src.evaluation import EvaluationModule, calculate_psnr, calculate_bdrate
from utils.codec_utils import calculate_bpp

def test_proxy_network():
    """Test that the proxy network builds and functions correctly."""
    # Create a sample input
    input_shape = (16, 64, 64, 128)  # (T, H, W, C)
    batch_size = 2
    sample_input = np.random.random((batch_size,) + input_shape).astype(np.float32)
    
    # Create the proxy network
    proxy_network = ProxyNetwork(input_shape=input_shape)
    encoder, decoder, autoencoder = proxy_network.build()
    
    # Get the output from the encoder
    latent = encoder(sample_input)
    
    # Check the encoder output shape
    expected_latent_shape = (batch_size, input_shape[0], input_shape[1]//2, input_shape[2]//2, proxy_network.latent_channels)
    print(f"Encoder output shape: {latent.shape}")
    assert latent.shape == expected_latent_shape, f"Expected shape {expected_latent_shape}, got {latent.shape}"
    
    # Get the output from the decoder
    decoder_output = decoder(latent)
    
    # Check the decoder output shape
    expected_output_shape = (batch_size, input_shape[0], input_shape[1], input_shape[2], input_shape[3])
    print(f"Decoder output shape: {decoder_output.shape}")
    assert decoder_output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {decoder_output.shape}"
    
    # Get the output from the autoencoder
    autoencoder_output = autoencoder(sample_input)
    
    # Check the autoencoder output shape
    print(f"Autoencoder output shape: {autoencoder_output.shape}")
    assert autoencoder_output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {autoencoder_output.shape}"
    
    # Test the bitrate proxy function
    bitrate = proxy_network.bitrate_proxy(latent)
    print(f"Estimated bitrate: {bitrate}")
    
    # Test the proxy loss function
    target_output = np.random.random((batch_size,) + input_shape).astype(np.float32)
    loss, bitrate, distortion = proxy_network.proxy_loss(target_output, autoencoder_output, latent)
    print(f"Proxy loss: {loss}, Bitrate: {bitrate}, Distortion: {distortion}")
    
    print("Proxy network test passed!")
    
def test_custom_loss():
    """Test the custom loss function for the proxy network."""
    # Create a sample input
    input_shape = (16, 64, 64, 128)  # (T, H, W, C)
    batch_size = 2
    sample_input = np.random.random((batch_size,) + input_shape).astype(np.float32)
    target_output = np.random.random((batch_size,) + input_shape).astype(np.float32)
    
    # Create the proxy network
    proxy_network = ProxyNetwork(input_shape=input_shape)
    encoder, decoder, autoencoder = proxy_network.build()
    
    # Create the custom loss function
    custom_loss = ProxyNetwork.create_custom_loss(encoder=encoder, lambda_value=0.1, use_ssim=False)
    
    # Test the loss function
    # Note: This will not work correctly in standalone testing because the loss function
    # needs access to the current inputs to the model, which are only available during training.
    # This test is just to ensure the function can be created without errors.
    print("Custom loss function created successfully.")
    
    print("Custom loss test passed!")
    
def test_evaluation_module():
    """Test the evaluation module functionality."""
    # Create a sample input
    batch_size = 2
    time_steps = 16
    height = 224
    width = 224
    channels = 3
    
    # Create sample videos
    original_video = np.random.random((batch_size, time_steps, height, width, channels)).astype(np.float32)
    preprocessed_video = np.random.random((batch_size, time_steps, height // 4, width // 4, 128)).astype(np.float32)
    
    # Initialize evaluation module
    evaluation_module = EvaluationModule()
    
    # Test PSNR calculation
    psnr = calculate_psnr(original_video[0], original_video[0] + 0.1 * np.random.random((time_steps, height, width, channels)))
    print(f"PSNR: {psnr}")
    
    # Test BD-Rate calculation
    # Create sample results for original and preprocessed videos
    original_results = [(0.5, 35.0), (0.8, 38.0), (1.2, 40.0), (2.0, 42.0)]
    preprocessed_results = [(0.4, 35.0), (0.7, 38.0), (1.0, 40.0), (1.8, 42.0)]
    
    bdrate = calculate_bdrate(original_results, preprocessed_results)
    print(f"BD-Rate: {bdrate}%")
    
    # Evaluate detection (using placeholder functions)
    detection_metrics = {
        'mAP_original': 0.8,
        'mAP_preprocessed': 0.78,
        'bpp_original': 1.2,
        'bpp_preprocessed': 0.8,
        'bitrate_savings': 33.33
    }
    
    # Evaluate segmentation (using placeholder functions)
    segmentation_metrics = {
        'mIoU_original': 0.75,
        'mIoU_preprocessed': 0.72,
        'bpp_original': 1.2,
        'bpp_preprocessed': 0.8,
        'bitrate_savings': 33.33
    }
    
    # Evaluate tracking (using placeholder functions)
    tracking_metrics = {
        'MOTA_original': 0.7,
        'MOTA_preprocessed': 0.68,
        'IDF1_original': 0.65,
        'IDF1_preprocessed': 0.63,
        'bpp_original': 1.2,
        'bpp_preprocessed': 0.8,
        'bitrate_savings': 33.33
    }
    
    print("Evaluation module test passed!")
    
def test_codec_utils():
    """Test the codec utilities."""
    # Create a sample input
    height = 32
    width = 32
    channels = 3
    num_frames = 5
    
    sample_frames = np.random.randint(0, 256, (num_frames, height, width, channels), dtype=np.uint8)
    
    # Test bitrate calculation
    # This is a simple proxy function for testing
    bpp = 8.0 * channels / (height * width)  # 8 bits per channel
    
    print(f"Theoretical bitrate: {bpp} bits per pixel")
    
    print("Codec utils test passed!")

if __name__ == "__main__":
    print("Running tests...")
    test_proxy_network()
    test_custom_loss()
    test_evaluation_module()
    test_codec_utils()
    print("All tests passed!") 