import os
import sys
import unittest
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import VCMPreprocessingPipeline
from src.models.st_npp import STNPP
from src.models.qal import QAL
from src.models.proxy_network import ProxyNetwork

def test_spatial_branch():
    """Test that the spatial branch produces the correct output shape."""
    from src.models.st_npp import SpatialBranch
    
    # Create a sample input
    input_shape = (224, 224, 3)
    batch_size = 2
    sample_input = np.random.random((batch_size,) + input_shape).astype(np.float32)
    
    # Create the spatial branch
    branch = SpatialBranch(input_shape=input_shape, backbone="resnet50").build()
    
    # Get the output
    output = branch(sample_input)
    
    # Check the output shape
    expected_shape = (batch_size, input_shape[0]//4, input_shape[1]//4, 128)
    print(f"Spatial branch output shape: {output.shape}")
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("Spatial branch test passed!")
    
def test_temporal_branch():
    """Test that the temporal branch produces the correct output shape."""
    from src.models.st_npp import TemporalBranch
    
    # Create a sample input
    input_shape = (16, 224, 224, 3)  # (Time, H, W, C)
    batch_size = 2
    sample_input = np.random.random((batch_size,) + input_shape).astype(np.float32)
    
    # Create the temporal branch
    branch = TemporalBranch(input_shape=input_shape, model_type="3dcnn").build()
    
    # Get the output
    output = branch(sample_input)
    
    # Check the output shape
    expected_shape = (batch_size, input_shape[1]//4, input_shape[2]//4, 128)
    print(f"Temporal branch output shape: {output.shape}")
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("Temporal branch test passed!")
    
def test_qal():
    """Test that the QAL produces the correct output shape."""
    # Create a sample input
    batch_size = 2
    sample_qp = np.array([[23], [30]], dtype=np.float32)
    
    # Create the QAL
    qal = QAL(feature_channels=128).build()
    
    # Get the output
    output = qal(sample_qp)
    
    # Check the output shape
    expected_shape = (batch_size, 128)
    print(f"QAL output shape: {output.shape}")
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print("QAL test passed!")
    
def test_complete_pipeline():
    """Test that the complete pipeline works correctly."""
    # Create sample inputs
    batch_size = 2
    time_steps = 16
    input_shape = (224, 224, 3)
    
    spatial_input = np.random.random((batch_size,) + input_shape).astype(np.float32)
    temporal_input = np.random.random((batch_size, time_steps) + input_shape).astype(np.float32)
    qp_input = np.array([[23], [30]], dtype=np.float32)
    
    # Create the pipeline
    pipeline = VCMPreprocessingPipeline(
        input_shape=(batch_size,) + input_shape,
        time_steps=time_steps,
        spatial_backbone="resnet50",
        temporal_model="3dcnn",
        fusion_type="concatenation"
    )
    
    # Build the models
    st_npp_model, qal_model, combined_model = pipeline.build()
    
    # Test each model separately
    st_npp_output = st_npp_model([spatial_input, temporal_input])
    print(f"ST-NPP output shape: {st_npp_output.shape}")
    
    qal_output = qal_model(qp_input)
    print(f"QAL output shape: {qal_output.shape}")
    
    # Test the combined model
    combined_output = combined_model([spatial_input, temporal_input, qp_input])
    expected_shape = (batch_size, input_shape[0]//4, input_shape[1]//4, 128)
    print(f"Combined model output shape: {combined_output.shape}")
    assert combined_output.shape == expected_shape, f"Expected shape {expected_shape}, got {combined_output.shape}"
    
    # Test the preprocess_video method
    output_features = pipeline.preprocess_video(
        temporal_input, 
        qp=qp_input,
        st_npp_model=st_npp_model,
        qal_model=qal_model
    )
    print(f"Preprocessed video shape: {output_features.shape}")
    assert output_features.shape == expected_shape, f"Expected shape {expected_shape}, got {output_features.shape}"
    
    print("Complete pipeline test passed!")
    
class TestPipeline(unittest.TestCase):
    """Test cases for the VCM preprocessing pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        # Set fixed random seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Create sample inputs
        self.batch_size = 2
        self.time_steps = 16
        self.height = 224
        self.width = 224
        self.channels = 3
        
        # Create sample video data
        self.video_data = np.random.random(
            (self.batch_size, self.time_steps, self.height, self.width, self.channels)
        ).astype(np.float32)
        
        # Initialize the pipeline with default settings
        self.pipeline = VCMPreprocessingPipeline(
            input_shape=(None, self.height, self.width, self.channels),
            time_steps=self.time_steps,
            spatial_backbone="resnet50",
            temporal_model="3dcnn",
            fusion_type="concatenation"
        )
        
        # Build the pipeline models
        self.st_npp_model, self.qal_model, self.combined_model = self.pipeline.build()

    def test_st_npp_model(self):
        """Test the Spatio-Temporal Neural Preprocessing module."""
        # Run the ST-NPP model
        features = self.st_npp_model(self.video_data)
        
        # Check output shape
        expected_shape = (self.batch_size, self.height//4, self.width//4, 128)
        self.assertEqual(features.shape, expected_shape)
        
        # Check that outputs are in a reasonable range
        self.assertTrue(np.all(np.isfinite(features.numpy())))
        
        print("ST-NPP model test passed!")

    def test_qal_model(self):
        """Test the Quantization Adaptation Layer."""
        # Create sample QP values
        qp_values = np.array([23, 27], dtype=np.float32)
        
        # Run the QAL model
        scale_factors = self.qal_model(qp_values)
        
        # Check output shape (should match the feature channels)
        expected_shape = (self.batch_size, 128)
        self.assertEqual(scale_factors.shape, expected_shape)
        
        # Check that scale factors are in [0, 1] range
        self.assertTrue(np.all(scale_factors.numpy() >= 0))
        self.assertTrue(np.all(scale_factors.numpy() <= 1))
        
        print("QAL model test passed!")

    def test_combined_model(self):
        """Test the combined pipeline."""
        # Create sample QP values
        qp_values = np.array([23, 27], dtype=np.float32)
        
        # Run the combined model
        features = self.combined_model([self.video_data, qp_values])
        
        # Check output shape
        expected_shape = (self.batch_size, self.height//4, self.width//4, 128)
        self.assertEqual(features.shape, expected_shape)
        
        # Check that outputs are in a reasonable range
        self.assertTrue(np.all(np.isfinite(features.numpy())))
        
        print("Combined model test passed!")

    def test_preprocess_video(self):
        """Test the preprocessing video function."""
        # Preprocess video
        features = self.pipeline.preprocess_video(
            self.video_data,
            qp=23,
            st_npp_model=self.st_npp_model,
            qal_model=self.qal_model
        )
        
        # Check output shape
        expected_shape = (self.batch_size, self.height//4, self.width//4, 128)
        self.assertEqual(features.shape, expected_shape)
        
        print("Preprocess video function test passed!")

    def test_proxy_network_integration(self):
        """Test the integration with the Proxy Network."""
        # Initialize the proxy network
        proxy_input_shape = (self.time_steps, self.height//4, self.width//4, 128)
        proxy_network = ProxyNetwork(input_shape=proxy_input_shape)
        
        # Build the proxy network
        encoder, decoder, autoencoder = proxy_network.build()
        
        # Preprocess video
        features = self.pipeline.preprocess_video(
            self.video_data,
            qp=23,
            st_npp_model=self.st_npp_model,
            qal_model=self.qal_model
        )
        
        # Pass through proxy network encoder
        latent = encoder(features)
        
        # Check encoder output shape
        expected_latent_shape = (
            self.batch_size, self.time_steps, 
            self.height//8, self.width//8, 
            proxy_network.latent_channels
        )
        self.assertEqual(latent.shape, expected_latent_shape)
        
        # Pass through proxy network decoder
        decoded = decoder(latent)
        
        # Check decoder output shape
        expected_decoded_shape = (
            self.batch_size, self.time_steps, 
            self.height//4, self.width//4, 
            128
        )
        self.assertEqual(decoded.shape, expected_decoded_shape)
        
        print("Proxy network integration test passed!")

if __name__ == "__main__":
    unittest.main() 