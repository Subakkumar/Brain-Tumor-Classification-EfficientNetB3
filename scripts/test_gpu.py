import tensorflow as tf
import sys

#print("TESTING GPU SETUP")
#print("="*40)

print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs Found: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        name = details.get('device_name', 'Unknown')
        print(f"GPU {i}: {name}")
    
    # Test computation
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"GPU computation test passed!")
        print(f"Result: {c.numpy()}")
else:
    print("No GPU detected")

print("GPU test completed!")