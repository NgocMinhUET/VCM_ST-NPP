import cv2
import numpy as np

# Make TensorFlow optional
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not found. Some functionality may be limited.")

    # Define fallback preprocessing functions if TensorFlow is not available
    def resnet_preprocess(x):
        """Fallback ResNet preprocessing when TensorFlow is not available"""
        # Normalize to [0, 1] and then apply approximate ResNet normalization
        x = x.astype(np.float32) / 255.0
        # Approximate ResNet normalization - RGB channels
        x[:, :, :, 0] -= 0.485  # R mean
        x[:, :, :, 0] /= 0.229  # R std
        x[:, :, :, 1] -= 0.456  # G mean
        x[:, :, :, 1] /= 0.224  # G std
        x[:, :, :, 2] -= 0.406  # B mean
        x[:, :, :, 2] /= 0.225  # B std
        return x

    def efficient_preprocess(x):
        """Fallback EfficientNet preprocessing when TensorFlow is not available"""
        # Normalize to [0, 1] and then scale to [-1, 1]
        x = x.astype(np.float32) / 255.0
        x = x * 2.0 - 1.0
        return x

def load_video(video_path, target_size=(224, 224), max_frames=None):
    """
    Load video frames from a file.
    
    Args:
        video_path: Path to the video file
        target_size: Target size for frames (height, width)
        max_frames: Maximum number of frames to load (None for all)
        
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = cv2.resize(frame, target_size)
        
        frames.append(frame)
        
        if max_frames is not None and len(frames) >= max_frames:
            break
            
    cap.release()
    
    return frames

def preprocess_frames(frames, model_name="resnet50"):
    """
    Preprocess frames for a specific model.
    
    Args:
        frames: List of frames or numpy array of shape (n_frames, height, width, channels)
        model_name: Name of the model for preprocessing
        
    Returns:
        Preprocessed frames as numpy array
    """
    if isinstance(frames, list):
        frames = np.stack(frames)
        
    # Apply specific preprocessing based on model
    if model_name.lower().startswith("resnet"):
        processed_frames = resnet_preprocess(frames.astype(np.float32))
    elif model_name.lower().startswith("efficient"):
        processed_frames = efficient_preprocess(frames.astype(np.float32))
    else:
        # Default preprocessing: scale to [0, 1]
        processed_frames = frames.astype(np.float32) / 255.0
        
    return processed_frames

def create_sliding_windows(frames, window_size, stride=1):
    """
    Create sliding window segments from a sequence of frames.
    
    Args:
        frames: List or array of frames
        window_size: Size of the sliding window
        stride: Step size for the sliding window
        
    Returns:
        List of frame segments, each of length window_size
    """
    if isinstance(frames, list):
        frames = np.stack(frames)
        
    n_frames = len(frames)
    segments = []
    
    for i in range(0, n_frames - window_size + 1, stride):
        segment = frames[i:i+window_size]
        segments.append(segment)
        
    return segments

def prepare_batch(video_segments, batch_size=16):
    """
    Prepare batches for model input.
    
    Args:
        video_segments: List of video segments
        batch_size: Batch size
        
    Returns:
        Generator that yields batches
    """
    n_segments = len(video_segments)
    indices = np.arange(n_segments)
    
    for start_idx in range(0, n_segments, batch_size):
        end_idx = min(start_idx + batch_size, n_segments)
        batch_indices = indices[start_idx:end_idx]
        
        # Extract current frames (last frame in each segment)
        current_frames = np.array([segment[-1] for segment in video_segments[start_idx:end_idx]])
        
        # Extract temporal sequences
        temporal_sequences = np.array(video_segments[start_idx:end_idx])
        
        yield current_frames, temporal_sequences 