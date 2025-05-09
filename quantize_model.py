# quantize_model.py

import tensorflow as tf
import numpy as np

# --- Select model ---
MODEL_NAME = "mobilenet_v3_small"  # Options: mobilenet_v3_small, mobilenet_v3_large
OUTPUT_PATH = f"{MODEL_NAME}_int8.tflite"

# --- 1. Load pretrained Keras model ---
if MODEL_NAME == "mobilenet_v3_small":
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        weights="imagenet",
        include_top=True
    )
elif MODEL_NAME == "mobilenet_v3_large":
    model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3),
        weights="imagenet",
        include_top=True
    )
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

# --- 2. Save as a SavedModel format (required by converter) ---
model.save("saved_model")

# --- 3. Create representative dataset function ---
def representative_data_gen():
    for _ in range(100):
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [dummy_input]

# --- 4. Create converter and quantize ---
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Force full INT8 quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# --- 5. Save output ---
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ Quantized model saved as: {OUTPUT_PATH}")