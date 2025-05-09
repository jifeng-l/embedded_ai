from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_details
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

def run_text_encoder(text, model_path="1_edgetpu.tflite"):
    tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    input_ids = tokens["input_ids"].astype(np.int32)

    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details(interpreter)[0]['index'], input_ids)
    interpreter.invoke()

    output = interpreter.tensor(interpreter.get_output_details()[0]['index'])()[0]
    return output