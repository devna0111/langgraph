import tensorflow as tf

print("TensorFlow 버전:", tf.__version__)
print("GPU 사용 가능:", tf.config.list_physical_devices('GPU'))
print("CUDA 빌드:", tf.test.is_built_with_cuda())