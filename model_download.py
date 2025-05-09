import kagglehub
import os
import shutil

# Download latest version
# path = kagglehub.model_download("google/mobilenet-v3/tfLite/small-075-224-feature-vector")

# Download latest version
path = kagglehub.model_download("google/mobilenet-edgetpu-v2/tfLite/l")

tflite_file = os.path.join(path, "1.tflite")
shutil.copy(tflite_file, "./model/mobilenet_v2_l.tflite")
print("✅ Model copied to ./mobilenet_v2_l.tflite")

print("Path to model files:", path)

