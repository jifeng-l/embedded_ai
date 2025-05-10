from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

def run_image_encoder(image_np, model_path="./model/mobilenet_edgetpu_v2_l_int8_edgetpu.tflite"):
    """
    Receives a NumPy RGB image, preprocesses and quantizes it,
    and runs it through the Edge TPU-compiled MobileNet encoder.
    """
    try:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate("libedgetpu.so.1")]
        )
    except Exception:
        print("⚠️ Edge TPU delegate not available. Running on CPU.")
        interpreter = Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    input_index = input_details["index"]
    height, width = input_details["shape"][1:3]
    dtype = input_details["dtype"]

    # Resize and normalize image
    image_resized = np.array(Image.fromarray(image_np).resize((width, height)))

    if dtype == np.uint8:
        scale, zero_point = input_details["quantization"]
        image_resized = image_resized.astype(np.float32) / 255.0
        image_resized = image_resized / scale + zero_point
        image_resized = np.clip(image_resized, 0, 255).astype(np.uint8)
    elif dtype == np.int8:
        scale, zero_point = input_details["quantization"]
        image_resized = image_resized.astype(np.float32) / 255.0
        image_resized = image_resized / scale + zero_point
        image_resized = np.clip(image_resized, -128, 127).astype(np.int8)
    else:
        image_resized = image_resized.astype(np.float32) / 255.0

    image_input = np.expand_dims(image_resized, axis=0)
    interpreter.set_tensor(input_index, image_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])[0]

    # Optional: dequantize
    if "quantization" in output_details and output_details["quantization"][0] > 0:
        scale, zero_point = output_details["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    return output