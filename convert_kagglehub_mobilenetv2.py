import os
import tensorflow as tf
import numpy as np
import kagglehub

# === 0. Prepare output directory ===
os.makedirs("./model", exist_ok=True)

# === 1. Download MobileNet-EdgeTPU-v2 feature vector (Large) from KaggleHub ===
print("📥 Downloading model from KaggleHub...")
path = kagglehub.model_download("google/mobilenet-edgetpu-v2/tensorFlow2/feature-vector-l")
print("✅ Model downloaded to:", path)

# === 2. Convert SavedModel to float32 TFLite ===
converter_fp32 = tf.lite.TFLiteConverter.from_saved_model(path)
tflite_model_fp32 = converter_fp32.convert()

fp32_path = "./model/mobilenet_edgetpu_v2_l_fp32.tflite"
with open(fp32_path, "wb") as f:
    f.write(tflite_model_fp32)
print("✅ Saved float32 TFLite model to:", fp32_path)

# === 3. Representative dataset for int8 quantization ===
def representative_dataset():
    for _ in range(100):
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [dummy]

# === 4. Convert to int8 quantized TFLite model ===
converter_int8 = tf.lite.TFLiteConverter.from_saved_model(path)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8

tflite_model_int8 = converter_int8.convert()

int8_path = "./model/mobilenet_edgetpu_v2_l_int8.tflite"
with open(int8_path, "wb") as f:
    f.write(tflite_model_int8)
print("✅ Saved int8 quantized model to:", int8_path)

# === 5. Final summary ===
print("\n🎯 Conversion complete! You can now run:")
print(f"   edgetpu_compiler {int8_path}")