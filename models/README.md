# Pre-trained Models

This directory contains pre-trained models for the VCM preprocessing pipeline.

## Model Structure

All models are saved in TensorFlow's HDF5 (.h5) format, which preserves the entire model including:
- Model architecture
- Weight values
- Optimizer state
- Loss function
- Metrics

## Available Models

### Proxy Network

- **Filename**: `proxy_network/proxy_network_final.h5`
- **Description**: 3D CNN-based Autoencoder that approximates the HEVC codec
- **Input Shape**: (16, 64, 64, 128) - (Timesteps, Height, Width, Channels)
- **Output Shape**: Same as input
- **Parameters**: ~8.2 million

### ST-NPP Module

- **Filename**: `st_npp/st_npp_final.h5`
- **Description**: Spatio-Temporal Neural Preprocessing module
- **Input Shape**: (16, 224, 224, 3) - (Timesteps, Height, Width, Channels)
- **Output Shape**: (Height/4, Width/4, 128)
- **Parameters**: ~14.7 million
- **Backbone**: ResNet-50 (spatial) + 3D CNN (temporal)

### QAL Module

- **Filename**: `qal/qal_final.h5`
- **Description**: Quantization Adaptation Layer
- **Input Shape**: QP value (scalar)
- **Output Shape**: (128,) - Scale factors for feature channels
- **Parameters**: ~50,000

## Loading Models

To load these models in TensorFlow:

```python
import tensorflow as tf

# Load a model
model = tf.keras.models.load_model('models/proxy_network/proxy_network_final.h5')

# Get model summary
model.summary()

# Use the model for inference
output = model(input_data)
```

## Model Performance

For detailed performance metrics of these models, please refer to the evaluation report (`evaluation_report.md`) in the root directory. 