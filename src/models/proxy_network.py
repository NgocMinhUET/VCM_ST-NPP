import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPool3D, Conv3DTranspose, UpSampling3D,
    BatchNormalization, Activation
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class ProxyNetwork:
    """
    Differentiable Proxy Network - 3D CNN-based Autoencoder
    
    This module serves as a differentiable proxy for HEVC codec during training,
    allowing end-to-end optimization of the preprocessing pipeline.
    """
    
    def __init__(self, 
                 input_shape=(16, 64, 64, 128),   # (T, H, W, C)
                 latent_channels=64,
                 use_batch_norm=True):
        """
        Initialize the Proxy Network.
        
        Args:
            input_shape: Shape of input feature maps from the ST-NPP module
            latent_channels: Number of channels in the latent representation
            use_batch_norm: Whether to use batch normalization
        """
        self.input_shape = input_shape
        self.latent_channels = latent_channels
        self.use_batch_norm = use_batch_norm
        
    def _conv_bn_act(self, x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
        """Helper function for Conv3D + (optional) BatchNorm + ReLU"""
        x = Conv3D(filters, kernel_size, strides=strides, padding=padding)(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def _convt_bn_act(self, x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
        """Helper function for Conv3DTranspose + (optional) BatchNorm + ReLU"""
        x = Conv3DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
        if self.use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    
    def build_encoder(self):
        """Build the encoder part of the network."""
        inputs = Input(shape=self.input_shape)
        
        # Encoder: Input → 3D Conv → 3D Conv → MaxPool3D → 3D Conv → Latent representation
        x = self._conv_bn_act(inputs, 128)
        x = self._conv_bn_act(x, 96)
        x = MaxPool3D(pool_size=(1, 2, 2))(x)  # Reduce spatial dimensions, preserve temporal
        x = self._conv_bn_act(x, self.latent_channels)
        
        # Create and return encoder model
        encoder = Model(inputs=inputs, outputs=x, name="proxy_encoder")
        return encoder
    
    def build_decoder(self):
        """Build the decoder part of the network."""
        # Input shape for the decoder
        _, T, H, W, C = self.input_shape
        latent_shape = (T, H//2, W//2, self.latent_channels)
        
        inputs = Input(shape=latent_shape)
        
        # Decoder: Latent → 3D ConvTranspose → UpSampling3D → 3D ConvTranspose → 3D Conv → Output
        x = self._convt_bn_act(inputs, 96)
        x = UpSampling3D(size=(1, 2, 2))(x)  # Increase spatial dimensions, preserve temporal
        x = self._convt_bn_act(x, 128)
        x = Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='sigmoid')(x)
        
        # Create and return decoder model
        decoder = Model(inputs=inputs, outputs=x, name="proxy_decoder")
        return decoder
    
    def build(self):
        """
        Build the complete proxy network (encoder + decoder).
        
        Returns:
            A tuple of (encoder_model, decoder_model, full_autoencoder_model)
        """
        # Build encoder and decoder
        encoder = self.build_encoder()
        decoder = self.build_decoder()
        
        # Connect them to form the autoencoder
        inputs = Input(shape=self.input_shape)
        latent = encoder(inputs)
        outputs = decoder(latent)
        
        # Create the full model
        autoencoder = Model(inputs=inputs, outputs=outputs, name="proxy_autoencoder")
        
        return encoder, decoder, autoencoder
    
    def bitrate_proxy(self, latent):
        """
        Proxy function to estimate bitrate based on the latent representation.
        
        Args:
            latent: Latent representation from the encoder
            
        Returns:
            Estimated bitrate
        """
        # A simple proxy for bitrate estimation based on entropy of the latent representation
        # For more accurate estimation, more sophisticated methods should be used
        # This is just a basic approximation
        
        # Normalize values to 0-1 range if not already
        latent_norm = tf.clip_by_value(latent, 0, 1)
        
        # Calculate entropy (as a proxy for bitrate)
        # Higher entropy generally requires more bits to encode
        entropy = -tf.reduce_mean(
            latent_norm * tf.math.log(latent_norm + 1e-10) +
            (1 - latent_norm) * tf.math.log(1 - latent_norm + 1e-10)
        )
        
        # Scale the entropy to a reasonable bitrate range (this is just an example)
        estimated_bitrate = entropy * 10.0
        
        return estimated_bitrate
    
    def proxy_loss(self, y_true, y_pred, latent, lambda_value=0.1, use_ssim=False):
        """
        Proxy loss function to approximate HEVC codec performance.
        
        Args:
            y_true: Target output (HEVC output)
            y_pred: Predicted output (proxy network output)
            latent: Latent representation from the encoder
            lambda_value: Weight for the distortion term
            use_ssim: Whether to use SSIM (True) or MSE (False) for distortion
            
        Returns:
            Loss value
        """
        # Estimate bitrate based on latent representation
        bitrate = self.bitrate_proxy(latent)
        
        # Calculate distortion
        if use_ssim:
            # Use SSIM-based distortion (1 - SSIM)
            ssim_value = tf.image.ssim(
                tf.reshape(y_true, [-1, self.input_shape[1], self.input_shape[2], self.input_shape[3]]),
                tf.reshape(y_pred, [-1, self.input_shape[1], self.input_shape[2], self.input_shape[3]]),
                max_val=1.0
            )
            distortion = 1.0 - tf.reduce_mean(ssim_value)
        else:
            # Use MSE for distortion
            distortion = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Combine bitrate and distortion terms
        loss = bitrate + lambda_value * distortion
        
        return loss, bitrate, distortion
    
    @staticmethod
    def create_custom_loss(encoder, lambda_value=0.1, use_ssim=False):
        """
        Create a custom loss function for the proxy network.
        
        Args:
            encoder: Encoder model to get the latent representation
            lambda_value: Weight for the distortion term
            use_ssim: Whether to use SSIM (True) or MSE (False) for distortion
            
        Returns:
            Loss function
        """
        def loss_function(y_true, y_pred):
            # Get the current inputs to the model
            inputs = encoder.inputs[0]
            # Get the latent representation
            latent = encoder(inputs)
            
            # Create an instance to access the methods
            proxy_net = ProxyNetwork()
            
            # Calculate the proxy loss
            loss, _, _ = proxy_net.proxy_loss(y_true, y_pred, latent, lambda_value, use_ssim)
            
            return loss
        
        return loss_function 