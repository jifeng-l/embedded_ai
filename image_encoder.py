from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input, input_size
from PIL import Image
import numpy as np

def run_image_encoder(image_path, model_path="2_edgetpu.tflite"):
    # 加载模型
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    # 加载图像并调整大小
    image = Image.open(image_path).convert("RGB").resize(input_size(interpreter))
    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    # 设置输入并推理
    set_input(interpreter, image_np)
    interpreter.invoke()

    # 获取输出向量
    output = interpreter.tensor(interpreter.get_output_details()[0]['index'])()[0]
    return output