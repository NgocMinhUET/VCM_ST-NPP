"""
Differentiable Proxy Network Model

This module implements a 3D CNN-based autoencoder that serves as a differentiable proxy
for the HEVC video codec. The proxy network is trained to approximate the rate-distortion
behavior of HEVC while being fully differentiable, allowing end-to-end training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DBlock(nn.Module):
    """A 3D convolutional block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTranspose3DBlock(nn.Module):
    """A 3D transposed convolutional block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(ConvTranspose3DBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ProxyNetworkEncoder(nn.Module):
    """Encoder part of the Proxy Network."""
    
    def __init__(self, input_channels=128, base_channels=64, latent_channels=32):
        super(ProxyNetworkEncoder, self).__init__()
        
        self.conv1 = Conv3DBlock(input_channels, base_channels)
        self.conv2 = Conv3DBlock(base_channels, base_channels * 2)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv3 = Conv3DBlock(base_channels * 2, latent_channels)
        
    def forward(self, x):
        x = self.conv1(x)      # (B, 64, T, H, W)
        x = self.conv2(x)      # (B, 128, T, H, W)
        x = self.pool(x)       # (B, 128, T, H/2, W/2)
        x = self.conv3(x)      # (B, 32, T, H/2, W/2)
        return x


class ProxyNetworkDecoder(nn.Module):
    """Decoder part of the Proxy Network."""
    
    def __init__(self, output_channels=128, base_channels=64, latent_channels=32):
        super(ProxyNetworkDecoder, self).__init__()
        
        self.conv_transpose1 = ConvTranspose3DBlock(latent_channels, base_channels * 2)
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True)
        self.conv_transpose2 = ConvTranspose3DBlock(base_channels * 2, base_channels)
        self.conv_output = nn.Conv3d(base_channels, output_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv_transpose1(x)  # (B, 128, T, H/2, W/2)
        x = self.upsample(x)         # (B, 128, T, H, W)
        x = self.conv_transpose2(x)  # (B, 64, T, H, W)
        x = self.conv_output(x)      # (B, 128, T, H, W)
        return x


class ProxyNetwork(nn.Module):
    """
    Differentiable Proxy Network that approximates HEVC codec behavior.
    
    The network consists of an encoder-decoder architecture with the latent space
    serving as a proxy for the compressed bitstream. The network is trained to
    minimize a rate-distortion loss function.
    """
    
    def __init__(self, input_channels=128, base_channels=64, latent_channels=32):
        super(ProxyNetwork, self).__init__()
        
        self.encoder = ProxyNetworkEncoder(input_channels, base_channels, latent_channels)
        self.decoder = ProxyNetworkDecoder(input_channels, base_channels, latent_channels)
        
    def forward(self, x):
        """
        Forward pass through the proxy network.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Tuple of (reconstructed_features, latent_representation)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def calculate_bitrate(self, latent):
        """
        Calculate the estimated bitrate from the latent representation.
        This is a proxy for the actual bits that would be used by HEVC.
        
        Args:
            latent: Latent representation from the encoder
        
        Returns:
            Estimated bitrate in bits per pixel
        """
        # Simple proxy for bits per pixel based on latent activation magnitudes
        # In real implementation, this would be a more sophisticated model
        return torch.mean(torch.abs(latent)) * 8.0
    
    def calculate_distortion(self, original, reconstructed, use_ssim=False):
        """
        Calculate the distortion between original and reconstructed features.
        
        Args:
            original: Original input features
            reconstructed: Reconstructed features from the proxy network
            use_ssim: Whether to use SSIM instead of MSE for distortion measurement
        
        Returns:
            Distortion measurement (lower is better)
        """
        if use_ssim:
            # Implement SSIM here if needed
            # This would require a differentiable SSIM implementation
            # For simplicity, we use MSE for now
            return F.mse_loss(reconstructed, original)
        else:
            return F.mse_loss(reconstructed, original)
    
    def calculate_rd_loss(self, original, reconstructed, latent, lambda_value=0.1, use_ssim=False):
        """
        Calculate the rate-distortion loss.
        
        Args:
            original: Original input features
            reconstructed: Reconstructed features from the proxy network
            latent: Latent representation from the encoder
            lambda_value: Weight factor for the distortion term
            use_ssim: Whether to use SSIM instead of MSE for distortion measurement
        
        Returns:
            Rate-distortion loss: rate + lambda * distortion
        """
        rate = self.calculate_bitrate(latent)
        distortion = self.calculate_distortion(original, reconstructed, use_ssim)
        
        # Rate-distortion loss: rate + lambda * distortion
        rd_loss = rate + lambda_value * distortion
        
        return rd_loss, rate, distortion


def test_proxy_network():
    """Test function to verify the proxy network implementation."""
    # Create a random input tensor
    batch_size = 2
    time_steps = 16
    height = 64
    width = 64
    channels = 128
    
    x = torch.randn(batch_size, channels, time_steps, height, width)
    
    # Create the proxy network
    proxy_net = ProxyNetwork(input_channels=channels)
    
    # Forward pass
    reconstructed, latent = proxy_net(x)
    
    # Calculate the rate-distortion loss
    rd_loss, rate, distortion = proxy_net.calculate_rd_loss(x, reconstructed, latent)
    
    # Print the shapes and values
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Rate: {rate.item()}")
    print(f"Distortion: {distortion.item()}")
    print(f"RD Loss: {rd_loss.item()}")
    
    assert x.shape == reconstructed.shape, "Input and reconstructed shapes should match"
    
    return True


if __name__ == "__main__":
    test_proxy_network() 