import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.model_download("google/mobilenet-v3/tfLite/small-075-224-feature-vector")


tflite_file = os.path.join(path, "1.tflite")
shutil.copy(tflite_file, "./model/mobilenet_v3.tflite")
print("✅ Model copied to ./mobilenet_v3.tflite")

print("Path to model files:", path)