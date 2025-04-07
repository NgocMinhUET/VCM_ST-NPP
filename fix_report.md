# Báo cáo Sửa lỗi (Fix Report)

## Vấn đề Ban đầu
Lỗi ban đầu khi chạy `scripts/train_proxy.py`:
```
ImportError: cannot import name 'HevcCodec' from 'utils.codec_utils' (/work/u9564043/Minh/Thesis/week_propose/p2/VCM_ST-NPP/utils/codec_utils.py)
```

## Phân tích Vấn đề
1. File `utils/codec_utils.py` không chứa class `HevcCodec`, nhưng tập lệnh `train_proxy.py` cố gắng import nó
2. Sau khi sửa lỗi `HevcCodec`, phát hiện thêm vấn đề phụ thuộc với TensorFlow
3. Vấn đề import với `VideoDataset` và `VideoFrameDataset` từ `utils.video_utils`
4. Tham số command line `--qp_values` được sử dụng nhưng không được định nghĩa

## Sửa đổi đã thực hiện
### 1. Thêm class HevcCodec vào utils/codec_utils.py
```python
class HevcCodec:
    """
    HEVC codec implementation for video encoding and decoding.
    
    This class provides methods to encode and decode video frames using the HEVC codec,
    as well as calculating bitrate and distortion metrics.
    """
    
    def __init__(self, yuv_format='420', preset='medium'):
        # ... implementation ...
    
    def encode_decode(self, frames, qp=23):
        # ... implementation ...
    
    def calculate_bitrate(self, encoded_file_path, frames):
        # ... implementation ...
    
    def cleanup(self):
        # ... implementation ...
```

### 2. Làm cho TensorFlow trở thành phụ thuộc tùy chọn
Trong `utils/codec_utils.py` và `utils/video_utils.py`:
```python
# Make TensorFlow optional
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not found. Some functionality may be limited.")
    
    # Define fallback preprocessing functions
    # ...
```

### 3. Sửa lỗi import trong train_proxy.py
Thay thế:
```python
from utils.video_utils import VideoDataset, VideoFrameDataset
```
Bằng:
```python
# Import utils.video_utils for its other functions
import utils.video_utils
```

Vì class `VideoDataset` đã được định nghĩa trong chính file `train_proxy.py`.

### 4. Thêm hỗ trợ cho tham số --qp_values
Sửa hàm `parse_args()` và thêm xử lý trong mã huấn luyện:
```python
parser.add_argument("--qp_values", type=str, default=None,
                    help="Comma-separated list of QP values to train on (e.g., '22,27,32,37')")

# Process QP values if provided
if args.qp_values:
    args.qp_values = [int(qp) for qp in args.qp_values.split(',')]
```

### 5. Cải tiến xử lý lỗi và thông báo
- Thêm kiểm tra tồn tại cho đường dẫn dataset
- Cung cấp thông tin chi tiết về cấu trúc MOT16 mong đợi
- Thêm kiểm tra an toàn trước khi sử dụng MOTImageSequenceDataset
- Hỗ trợ cho tham số chuyển đổi tên (ví dụ: `--dataset` thay vì `--dataset_path`)

## Vấn đề Còn lại
1. **Lỗi Kích thước Tensor**: Khi chạy huấn luyện, chúng ta gặp lỗi về kích thước tensor không khớp:
   ```
   RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 7 but got size 112 for tensor number 1 in the list.
   ```
   Điều này gợi ý về sự không tương thích giữa kiến trúc mô hình ST-NPP và dữ liệu đầu vào. Kiến trúc mô hình cần được kiểm tra kỹ lưỡng hơn.

2. **Sử dụng Bộ nhớ GPU**: Mô hình yêu cầu một lượng đáng kể bộ nhớ GPU, gây ra lỗi CUDA out of memory. Có thể cần giảm batch size, độ phân giải đầu vào, hoặc độ phức tạp của mô hình.

3. **Phụ thuộc TensorFlow**: Mặc dù chúng ta đã làm cho TensorFlow tùy chọn, một số chức năng có thể vẫn yêu cầu nó cho hiệu suất tối ưu. Cần cài đặt TensorFlow nếu muốn sử dụng đầy đủ chức năng.

4. **Ghi chép Dataset không đầy đủ**: Ghi chép về định dạng dataset mong đợi và các tham số tương ứng không đầy đủ, gây khó khăn cho người dùng.

## Khuyến nghị

1. **Cập nhật Tài liệu Kỹ thuật**: Cung cấp mô tả rõ ràng về yêu cầu dataset và tham số huấn luyện.

2. **Phát triển Kiểm tra Dữ liệu Đầu vào**: Thêm kiểm tra trước khi huấn luyện để xác nhận dữ liệu đầu vào có kích thước và định dạng phù hợp.

3. **Tối ưu hóa Kiến trúc Mô hình**: Đánh giá lại kiến trúc của ST-NPP, đặc biệt là các phương thức fusion, để xử lý đúng các kích thước tensor.

4. **Tham số Tiết kiệm Bộ nhớ**: Thêm tham số dòng lệnh để kiểm soát sử dụng bộ nhớ, như `--half_precision` hoặc cấu hình cụ thể cho CUDA. 