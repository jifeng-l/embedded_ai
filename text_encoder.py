# from pycoral.utils.edgetpu import make_interpreter
# from pycoral.adapters.common import input_details
# from transformers import AutoTokenizer
# import numpy as np

# tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# def run_text_encoder(text, model_path="1_edgetpu.tflite"):
#     tokens = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
#     input_ids = tokens["input_ids"].astype(np.int32)

#     interpreter = make_interpreter(model_path)
#     interpreter.allocate_tensors()
#     interpreter.set_tensor(input_details(interpreter)[0]['index'], input_ids)
#     interpreter.invoke()

#     output = interpreter.tensor(interpreter.get_output_details()[0]['index'])()[0]
#     return output

import numpy as np
import json
import platform
import tflite_runtime.interpreter as tflite

def input_details(interpreter):
    return interpreter.get_input_details()

def make_interpreter(model_path):
    if platform.system() == "Linux":
        delegate_path = "libedgetpu.so.1"
    elif platform.system() == "Darwin":
        delegate_path = "libedgetpu.1.dylib"
    elif platform.system() == "Windows":
        delegate_path = "edgetpu.dll"
    else:
        raise RuntimeError("Unsupported OS")

    return tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate(delegate_path)]
    )

# Load vocab once (word → id)
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Tokenize input text using your word-level vocab
def tokenize_text(text, vocab, max_length=100):
    tokens = text.lower().strip().split()
    ids = [vocab.get(tok, vocab.get("UNK", 1)) for tok in tokens]
    ids = ids[:max_length] + [vocab.get("PAD", 0)] * (max_length - len(ids))
    return np.array(ids, dtype=np.int32).reshape(1, -1)  # Input shape: [1, 100]

# Main encoder inference function for TextCNN
def run_text_encoder(text, model_path="./model/textcnn_encoder_int8.tflite"):
    input_ids = tokenize_text(text, vocab)
    
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    input_index = input_details(interpreter)[0]['index']
    interpreter.set_tensor(input_index, input_ids)
    interpreter.invoke()

    output_index = interpreter.get_output_details()[0]['index']
    output = interpreter.tensor(output_index)()[0]
    return output

if __name__ == "__main__":
    # Example usage
    text = "This is a test sentence."
    model_path = "model/textcnn_encoder_int8.tflite"
    output = run_text_encoder(text, model_path)
    print("Output shape:", output.shape)
    print("Output:", output)