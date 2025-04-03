import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Dense, Concatenate, Reshape
from tensorflow.keras.layers import TimeDistributed, ConvLSTM2D, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, EfficientNetB4

class SpatialBranch:
    """Spatial branch of the ST-NPP module using 2D CNN."""
    
    def __init__(self, input_shape, backbone="resnet50"):
        """
        Initialize the spatial branch.
        
        Args:
            input_shape: Tuple of (H, W, C)
            backbone: String, either "resnet50" or "efficientnet-b4"
        """
        self.input_shape = input_shape
        self.backbone = backbone
        
    def build(self):
        """Build the spatial branch model."""
        inputs = Input(shape=self.input_shape)
        
        # Select backbone
        if self.backbone == "resnet50":
            backbone_model = ResNet50(
                include_top=False, 
                weights='imagenet', 
                input_shape=self.input_shape
            )
        elif self.backbone == "efficientnet-b4":
            backbone_model = EfficientNetB4(
                include_top=False, 
                weights='imagenet', 
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        
        # Get features from an intermediate layer
        x = backbone_model(inputs)
        
        # Ensure output shape is (H/4, W/4, 128)
        x = Conv2D(128, kernel_size=1, padding='same')(x)
        
        return Model(inputs=inputs, outputs=x, name="spatial_branch")


class TemporalBranch:
    """Temporal branch of the ST-NPP module using 3D CNN or ConvLSTM."""
    
    def __init__(self, input_shape, model_type="3dcnn"):
        """
        Initialize the temporal branch.
        
        Args:
            input_shape: Tuple of (Time, H, W, C)
            model_type: String, either "3dcnn" or "convlstm"
        """
        self.input_shape = input_shape
        self.model_type = model_type
        
    def build(self):
        """Build the temporal branch model."""
        inputs = Input(shape=self.input_shape)
        
        if self.model_type == "3dcnn":
            # 3D CNN implementation
            x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu')(inputs)
            x = Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
            # Take the last time step
            x = tf.reduce_mean(x, axis=1)
            
        elif self.model_type == "convlstm":
            # ConvLSTM implementation
            x = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True)(inputs)
            x = ConvLSTM2D(128, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Ensure output shape is (H/4, W/4, 128)
        x = Conv2D(128, kernel_size=1, padding='same')(x)
        
        return Model(inputs=inputs, outputs=x, name="temporal_branch")


class FeatureFusion:
    """Feature fusion module to combine spatial and temporal features."""
    
    def __init__(self, fusion_type="concatenation"):
        """
        Initialize the feature fusion module.
        
        Args:
            fusion_type: String, either "concatenation" or "attention"
        """
        self.fusion_type = fusion_type
        
    def build(self, spatial_output, temporal_output):
        """
        Build the feature fusion model.
        
        Args:
            spatial_output: Output tensor from spatial branch
            temporal_output: Output tensor from temporal branch
            
        Returns:
            Fused feature tensor
        """
        if self.fusion_type == "concatenation":
            # Simple concatenation along channel dimension
            fused_features = Concatenate(axis=-1)([spatial_output, temporal_output])
            # Reduce channels to 128 using 1x1 convolution
            fused_features = Conv2D(128, kernel_size=1, padding='same')(fused_features)
            
        elif self.fusion_type == "attention":
            # Attention-based fusion
            attention_weights = Conv2D(128, kernel_size=1, activation='sigmoid')(
                Concatenate(axis=-1)([spatial_output, temporal_output])
            )
            weighted_spatial = Multiply()([spatial_output, attention_weights])
            weighted_temporal = Multiply()([temporal_output, 1 - attention_weights])
            fused_features = Add()([weighted_spatial, weighted_temporal])
            
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
            
        return fused_features


class STNPP:
    """Spatio-Temporal Neural Preprocessing (ST-NPP) module."""
    
    def __init__(self, 
                 input_shape=(None, 224, 224, 3),
                 time_steps=16,
                 spatial_backbone="resnet50",
                 temporal_model="3dcnn",
                 fusion_type="concatenation"):
        """
        Initialize the ST-NPP module.
        
        Args:
            input_shape: Tuple of (Batch, H, W, C)
            time_steps: Number of time steps for temporal processing
            spatial_backbone: String, backbone for spatial branch
            temporal_model: String, model type for temporal branch
            fusion_type: String, fusion method
        """
        self.input_shape = input_shape[1:]  # (H, W, C)
        self.time_steps = time_steps
        self.temporal_input_shape = (time_steps,) + self.input_shape
        self.spatial_backbone = spatial_backbone
        self.temporal_model = temporal_model
        self.fusion_type = fusion_type
        
    def build(self):
        """Build the complete ST-NPP model."""
        # Define inputs
        spatial_input = Input(shape=self.input_shape, name="spatial_input")
        temporal_input = Input(shape=self.temporal_input_shape, name="temporal_input")
        
        # Build branches
        spatial_branch = SpatialBranch(
            input_shape=self.input_shape,
            backbone=self.spatial_backbone
        ).build()
        
        temporal_branch = TemporalBranch(
            input_shape=self.temporal_input_shape,
            model_type=self.temporal_model
        ).build()
        
        # Get features from each branch
        spatial_features = spatial_branch(spatial_input)
        temporal_features = temporal_branch(temporal_input)
        
        # Fusion module
        fusion_module = FeatureFusion(fusion_type=self.fusion_type)
        fused_features = fusion_module.build(spatial_features, temporal_features)
        
        # Create and return the model
        model = Model(
            inputs=[spatial_input, temporal_input],
            outputs=fused_features,
            name="st_npp"
        )
        
        return model 