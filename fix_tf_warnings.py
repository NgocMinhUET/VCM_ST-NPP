"""
Helper script to suppress TensorFlow warnings.
Import this at the beginning of your main script to suppress warnings.
"""

import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="Could not find cuda drivers")
warnings.filterwarnings('ignore', message="Unable to register")

# Print a message to confirm warnings are being suppressed
print("TensorFlow warnings have been suppressed.")

def disable_tf_gpu():
    """
    Disable TensorFlow GPU usage to avoid GPU-related errors.
    Call this function if you encounter GPU-related errors with TensorFlow.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("TensorFlow GPU usage has been disabled.")

def set_memory_growth():
    """
    Configure TensorFlow to grow memory usage as needed.
    Call this function to avoid TensorFlow claiming all GPU memory.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set for {len(gpus)} GPU(s)")
    except:
        print("Failed to set memory growth. TensorFlow may not be installed correctly.") 