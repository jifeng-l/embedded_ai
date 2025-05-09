# export_textcnn_tflite.py
import tensorflow as tf
import numpy as np
import os

# === 1. Build TextCNN model ===
def build_textcnn_encoder(vocab_size=10000, embed_dim=128, maxlen=100):
    """
    A simple, quantization-friendly TextCNN encoder model.
    Input: token IDs [batch, maxlen]
    Output: embedding vector [batch, 64]
    """
    inputs = tf.keras.Input(shape=(maxlen,), dtype="int32", name="input_ids")

    x = tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=maxlen)(inputs)

    convs = []
    for k in [3, 4, 5]:
        c = tf.keras.layers.Conv1D(64, kernel_size=k, activation='relu', padding='same')(x)
        p = tf.keras.layers.GlobalMaxPooling1D()(c)
        convs.append(p)

    x = tf.keras.layers.Concatenate()(convs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, name="output_embedding")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="textcnn_encoder")


# === 2. Save as SavedModel ===
saved_model_dir = "saved_textcnn"
if not os.path.exists(saved_model_dir):
    model = build_textcnn_encoder()
    model.save(saved_model_dir)
    print(f"✅ Saved Keras model to: {saved_model_dir}")
else:
    print(f"✅ Found existing model at: {saved_model_dir}")


# === 3. Representative dataset for quantization ===
def representative_data_gen():
    for _ in range(100):
        dummy_input = np.random.randint(1, 10000, size=(1, 100)).astype(np.int32)
        yield [dummy_input]


# === 4. Convert to quantized TFLite ===
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

# === 5. Save to file ===
output_file = "textcnn_encoder_int8.tflite"
with open(output_file, "wb") as f:
    f.write(tflite_model)

print(f"✅ Quantized model saved to: {output_file}")