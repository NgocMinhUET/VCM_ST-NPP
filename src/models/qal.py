import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model

class QAL:
    """
    Quantization Adaptation Layer (QAL)
    
    This module adapts the feature maps based on the Quantization Parameter (QP)
    by generating a scale vector that is applied channel-wise to the feature maps.
    """
    
    def __init__(self, feature_channels=128):
        """
        Initialize the QAL module.
        
        Args:
            feature_channels: Number of channels in the feature maps to scale
        """
        self.feature_channels = feature_channels
        
    def build(self):
        """
        Build the QAL model.
        
        Returns:
            A Keras model that takes QP as input and outputs a scale vector
        """
        # Input layer for QP (scalar value)
        qp_input = Input(shape=(1,), name="qp_input")
        
        # MLP architecture as specified
        x = Dense(64, activation='relu')(qp_input)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.feature_channels, activation='sigmoid')(x)
        
        # Create and return the model
        model = Model(inputs=qp_input, outputs=x, name="qal")
        
        return model
    
    def apply_to_features(self, qal_model, feature_maps, qp):
        """
        Apply the QAL to scale feature maps based on QP.
        
        Args:
            qal_model: Trained QAL model
            feature_maps: Feature maps from ST-NPP module, shape (B, H, W, C)
            qp: Quantization Parameter value(s)
            
        Returns:
            Scaled feature maps
        """
        # Ensure QP has the right shape
        if isinstance(qp, (int, float)):
            qp = tf.constant([[qp]], dtype=tf.float32)
        else:
            qp = tf.reshape(qp, (-1, 1))
            
        # Generate scale vector from QP
        scale_vector = qal_model(qp)  # Shape: (B, C)
        
        # Reshape scale vector to apply to feature maps
        # From (B, C) to (B, 1, 1, C) for broadcasting
        B = tf.shape(feature_maps)[0]
        H = tf.shape(feature_maps)[1]
        W = tf.shape(feature_maps)[2]
        C = self.feature_channels
        
        scale_vector = tf.reshape(scale_vector, [B, 1, 1, C])
        
        # Apply scaling
        scaled_features = feature_maps * scale_vector
        
        return scaled_features 