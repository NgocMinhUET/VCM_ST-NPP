import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImprovedAutoencoder(nn.Module):
    """
    Mô hình autoencoder cải tiến để nén video hiệu quả hơn.
    
    Cải tiến chính:
    1. Giảm số lượng kênh tính năng trong latent space
    2. Thêm nén theo chiều thời gian
    3. Tăng độ nén không gian (spatial)
    4. Thêm lớp lượng tử hóa để giảm dung lượng biểu diễn
    """
    
    def __init__(self, input_channels=3, latent_channels=8, time_reduction=2, num_embeddings=512):
        super(ImprovedAutoencoder, self).__init__()
        
        self.latent_channels = latent_channels
        self.time_reduction = time_reduction
        
        # Encoder - với 3 lớp MaxPool3d thay vì 2 lớp
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(time_reduction, 2, 2), stride=(time_reduction, 2, 2)),  # Giảm cả thời gian
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, latent_channels, kernel_size=3, padding=1),  # Giảm channels từ 64 xuống latent_channels
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Decoder - với 3 lớp Upsample tương ứng
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True),
            
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True),
            
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(time_reduction, 2, 2), mode='trilinear', align_corners=True),
            
            nn.Conv3d(16, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output trong khoảng [0, 1]
        )
        
        # Quantization layer
        self.quantize = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_channels)
        
        # Update bits per index for bitrate calculation
        self.bits_per_index = np.log2(num_embeddings)
    
    def encode(self, x):
        """Mã hóa đầu vào thành biểu diễn tiềm ẩn."""
        return self.encoder(x)
    
    def quantize_latent(self, latent):
        """Áp dụng lượng tử hóa vector cho biểu diễn tiềm ẩn."""
        quantized, loss, _ = self.quantize(latent)
        return quantized, loss
    
    def decode(self, latent):
        """Giải mã biểu diễn tiềm ẩn thành hình ảnh tái tạo."""
        return self.decoder(latent)
    
    def forward(self, x):
        """Tiến hành quá trình mã hóa và giải mã."""
        # Mã hóa
        latent = self.encode(x)
        #print(f"Latent shape after encode: {latent.shape}")
        
        # Lượng tử hóa
        quantized, quant_loss = self.quantize_latent(latent)
        #print(f"Latent shape after quantize: {quantized.shape}")
        
        # Giải mã
        reconstructed = self.decode(quantized)
        #print(f"Output shape after decode: {reconstructed.shape}")
        
        return reconstructed, quant_loss, latent
    
    def calculate_bitrate(self, latent):
        """Tính toán bitrate ước tính."""
        # Áp dụng lượng tử hóa
        _, _, indices = self.quantize(latent)
        
        # Mỗi chỉ số codebook chiếm log2(512) = 9 bits
        # Bước 1: Đếm số lượng chỉ số (số lượng vector được lượng tử hóa)
        num_indices = np.prod(indices.shape)
        
        # Bước 2: Tính tổng số bits
        total_bits = num_indices * self.bits_per_index
        
        # Bước 3: Tính số pixel trong đầu vào gốc
        # Latent shape là (B, C, T/time_reduction, H/8, W/8)
        # Original shape là (B, input_channels, T, H, W)
        batch_size = latent.shape[0]
        original_time = latent.shape[2] * self.time_reduction
        original_height = latent.shape[3] * 8
        original_width = latent.shape[4] * 8
        total_pixels = batch_size * original_time * original_height * original_width
        
        # Bước 4: Tính BPP (bits per pixel)
        bpp = total_bits / total_pixels
        
        return bpp


class VectorQuantizer(nn.Module):
    """
    Lớp lượng tử hóa vector dựa trên kỹ thuật VQ-VAE.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Khởi tạo codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, inputs):
        # Lưu ý: inputs shape là [B, C, T, H, W]
        input_shape = inputs.shape
        #print(f"VQ input shape: {input_shape}")
        
        # Hoán vị để channel đứng cuối cùng để dễ dàng phẳng hóa theo channel
        inputs_perm = inputs.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        #print(f"VQ permuted shape: {inputs_perm.shape}")
        
        # Chuyển thành [B*T*H*W, C] để mỗi vector đặc trưng là 1 hàng
        flat_input = inputs_perm.reshape(-1, self.embedding_dim)
        #print(f"VQ flattened shape: {flat_input.shape}")
        
        # Tính khoảng cách với các embedding
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Tìm embedding gần nhất
        encoding_indices = torch.argmin(distances, dim=1)
        #print(f"VQ indices shape: {encoding_indices.shape}")
        
        # Chuyển lại về định dạng ban đầu
        encoding_indices_reshaped = encoding_indices.reshape(input_shape[0], input_shape[2], input_shape[3], input_shape[4])
        #print(f"VQ reshaped indices: {encoding_indices_reshaped.shape}")
        
        # Lấy các embedding tương ứng
        quantized_flat = self.embedding(encoding_indices)
        #print(f"VQ quantized flat shape: {quantized_flat.shape}")
        
        # Reshape lại về dạng [B, T, H, W, C]
        quantized_reshaped = quantized_flat.reshape(input_shape[0], input_shape[2], input_shape[3], input_shape[4], self.embedding_dim)
        #print(f"VQ quantized reshaped: {quantized_reshaped.shape}")
        
        # Chuyển lại về dạng [B, C, T, H, W]
        quantized = quantized_reshaped.permute(0, 4, 1, 2, 3)
        #print(f"VQ quantized final shape: {quantized.shape}")
        
        # Tính loss commitment
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices_reshaped


def count_parameters(model):
    """Đếm số lượng tham số của mô hình."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Hàm kiểm tra mô hình."""
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tạo mô hình
    model = ImprovedAutoencoder(input_channels=3, latent_channels=8, time_reduction=2).to(device)
    
    # Tạo dữ liệu mẫu
    batch_size = 2
    time_steps = 16
    height = 224
    width = 224
    x = torch.randn(batch_size, 3, time_steps, height, width, device=device)
    
    # In ra kích thước đầu vào
    print(f"Input shape: {x.shape}")
    
    # Chạy mô hình
    with torch.no_grad():  # Disable gradient computation for testing
        reconstructed, quant_loss, latent = model(x)
    
    # Tính tỷ lệ nén (kích thước tensor)
    original_size = np.prod(x.shape) * 4  # 4 bytes for float32
    latent_size = np.prod(latent.shape) * 4  # 4 bytes for float32
    
    print(f"\nCompression statistics:")
    print(f"Original size: {original_size} bytes")
    print(f"Latent size: {latent_size} bytes")
    print(f"Compression ratio (tensors): {original_size / latent_size:.2f}x")
    
    # Tính BPP
    try:
        bpp = model.calculate_bitrate(latent)
        print(f"Bits per pixel: {bpp:.4f}")
    except Exception as e:
        print(f"Error calculating BPP: {e}")
    
    # Số lượng tham số
    print(f"\nModel parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    test_model() 