import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate

def run_image_encoder(image_path, model_path="./model/mobilenet_v2_l_edgetpu.tflite"):
    """
    Loads an Edge TPU-compiled MobileNetV2 Large TFLite model,
    preprocesses and quantizes the input image, runs inference, and returns the output embedding.
    """

    # Load Edge TPU model with delegate if available
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1")]
        )
    except Exception:
        print("⚠️ Edge TPU delegate not available. Running on CPU.")
        interpreter = Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()[0]
    input_index = input_details["index"]
    height, width = input_details["shape"][1:3]
    dtype = input_details["dtype"]

    # Load and resize image
    image = Image.open(image_path).convert("RGB").resize((width, height))
    image_np = np.asarray(image)

    # Apply quantization if required
    if dtype == np.uint8:
        scale, zero_point = input_details["quantization"]
        image_np = image_np.astype(np.float32) / 255.0
        image_np = image_np / scale + zero_point
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    elif dtype == np.int8:
        scale, zero_point = input_details["quantization"]
        image_np = image_np.astype(np.float32) / 255.0
        image_np = image_np / scale + zero_point
        image_np = np.clip(image_np, -128, 127).astype(np.int8)
    else:
        image_np = image_np.astype(np.float32) / 255.0

    image_np = np.expand_dims(image_np, axis=0)  # shape: [1, H, W, 3]
    interpreter.set_tensor(input_index, image_np)

    # Run inference
    interpreter.invoke()

    # Get output details
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])[0]

    # Optional: dequantize output if it's quantized
    if "quantization" in output_details and output_details["quantization"][0] > 0:
        scale, zero_point = output_details["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    return output
