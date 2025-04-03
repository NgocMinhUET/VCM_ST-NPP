import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from src.models.st_npp import STNPP
from src.models.qal import QAL

class VCMPreprocessingPipeline:
    """
    Complete preprocessing pipeline for video compression for machine vision (VCM).
    
    This pipeline combines the Spatio-Temporal Neural Preprocessing (ST-NPP) module
    and the Quantization Adaptation Layer (QAL) to preprocess videos before they are
    fed into standard video codecs like HEVC/VVC.
    """
    
    def __init__(self, 
                 input_shape=(None, 224, 224, 3),
                 time_steps=16,
                 spatial_backbone="resnet50",
                 temporal_model="3dcnn",
                 fusion_type="concatenation",
                 feature_channels=128):
        """
        Initialize the VCM preprocessing pipeline.
        
        Args:
            input_shape: Tuple of (Batch, H, W, C) for input frames
            time_steps: Number of time steps for temporal processing
            spatial_backbone: Backbone for spatial branch ("resnet50" or "efficientnet-b4")
            temporal_model: Model type for temporal branch ("3dcnn" or "convlstm")
            fusion_type: Fusion method ("concatenation" or "attention")
            feature_channels: Number of output feature channels
        """
        self.input_shape = input_shape
        self.time_steps = time_steps
        self.spatial_backbone = spatial_backbone
        self.temporal_model = temporal_model
        self.fusion_type = fusion_type
        self.feature_channels = feature_channels
        
        # Initialize sub-modules
        self.st_npp = STNPP(
            input_shape=input_shape,
            time_steps=time_steps,
            spatial_backbone=spatial_backbone,
            temporal_model=temporal_model,
            fusion_type=fusion_type
        )
        
        self.qal = QAL(feature_channels=feature_channels)
        
    def build(self):
        """
        Build the complete pipeline.
        
        Returns:
            A tuple of (st_npp_model, qal_model, combined_model)
        """
        # Build ST-NPP model
        st_npp_model = self.st_npp.build()
        
        # Build QAL model
        qal_model = self.qal.build()
        
        # Define inputs for combined model
        spatial_input = Input(shape=self.input_shape[1:], name="spatial_input")
        temporal_input = Input(shape=(self.time_steps,) + self.input_shape[1:], name="temporal_input")
        qp_input = Input(shape=(1,), name="qp_input")
        
        # Forward pass through ST-NPP
        features = st_npp_model([spatial_input, temporal_input])
        
        # Forward pass through QAL
        scale_vector = qal_model(qp_input)
        
        # Reshape scale vector to apply to feature maps
        B = tf.shape(features)[0]
        scale_vector = tf.reshape(scale_vector, [B, 1, 1, self.feature_channels])
        
        # Apply scaling
        scaled_features = features * scale_vector
        
        # Create combined model
        combined_model = Model(
            inputs=[spatial_input, temporal_input, qp_input],
            outputs=scaled_features,
            name="vcm_preprocessing_pipeline"
        )
        
        return st_npp_model, qal_model, combined_model
    
    def preprocess_video(self, video_frames, qp, st_npp_model=None, qal_model=None):
        """
        Preprocess a video using the pipeline.
        
        Args:
            video_frames: Tensor of shape (B, T, H, W, C) or list of frames
            qp: Quantization Parameter value(s)
            st_npp_model: ST-NPP model (if None, will be built)
            qal_model: QAL model (if None, will be built)
            
        Returns:
            Preprocessed feature maps
        """
        # Build models if not provided
        if st_npp_model is None or qal_model is None:
            st_npp_model, qal_model, _ = self.build()
        
        # Ensure video_frames is a tensor of the right shape
        if isinstance(video_frames, list):
            video_frames = tf.stack(video_frames)
        
        # Extract current frame for spatial branch
        current_frame = video_frames[:, -1]  # Last frame in sequence
        
        # Process through ST-NPP
        features = st_npp_model([current_frame, video_frames])
        
        # Apply QAL
        scaled_features = self.qal.apply_to_features(qal_model, features, qp)
        
        return scaled_features 