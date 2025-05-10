# #!/usr/bin/env python3
# # gradio_vlm_service.py

# import cv2
# import numpy as np
# import gradio as gr

# from tflite_runtime.interpreter import Interpreter, load_delegate
# from transformers import AutoTokenizer
# from llama_cpp import Llama  # pip install llama-cpp-python

# # ——— 1) Load 4-bit LLM directly from Hugging Face ———
# llm = Llama.from_pretrained(
#     repo_id="TheBloke/Llama-2-7B-GGUF",              # hosted GGUF model
#     filename="llama-2-7b.Q4_0.gguf",                 # Q4_0 version
#     verbose=False,
#     n_ctx=4096,                             # context length
#     n_batch=64,
#     n_threads=2,                           # number of threads
# )

# def run_llm(prompt: str) -> str:
#     """
#     Send the combined prompt to the 4-bit LLM (loaded from Hugging Face)
#     and return its generated text.
#     """
#     resp = llm(
#         prompt,
#         max_tokens=256,
#         temperature=0.7,
#         top_p=0.9
#     )
#     return resp["choices"][0]["text"]


# # ——— 2) Edge TPU image encoder ———
# def run_image_encoder(image_rgb: np.ndarray,
#                       model_path="1_edgetpu.tflite") -> np.ndarray:
#     """
#     Resize an RGB image to the required shape and type (float32),
#     run through the Edge TPU feature extractor, and return the embedding.
#     """
#     interpreter = Interpreter(
#         model_path=model_path,
#         experimental_delegates=[load_delegate("libedgetpu.so.1")]
#     )
#     interpreter.allocate_tensors()
#     inp_det = interpreter.get_input_details()[0]
#     out_det = interpreter.get_output_details()[0]

#     H, W = inp_det["shape"][1], inp_det["shape"][2]
#     resized = cv2.resize(image_rgb, (W, H))

#     # ✅ Normalize to 0.0 - 1.0 and convert to float32
#     input_data = resized.astype(np.float32) / 255.0
#     input_data = np.expand_dims(input_data, axis=0)  # [1, H, W, 3]

#     interpreter.set_tensor(inp_det["index"], input_data)
#     interpreter.invoke()

#     embedding = interpreter.get_tensor(out_det["index"])[0]
#     print("✅ Image embedding preview:", embedding[:10])
#     return embedding

# # ——— 3) Edge TPU text encoder ———
# tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# def run_text_encoder(text: str,
#                      model_path="2_edgetpu.tflite") -> np.ndarray:
#     """
#     Tokenize input text to max_length=384,
#     run through the Edge TPU TFLite encoder,
#     and return a 1D embedding vector.
#     """
#     tokens = tokenizer(
#         text,
#         return_tensors="np",
#         padding="max_length",
#         truncation=True,
#         max_length=384
#     )
#     input_ids = tokens["input_ids"].astype(np.int32)

#     interp = Interpreter(
#         model_path=model_path,
#         experimental_delegates=[load_delegate("libedgetpu.so.1")]
#     )
#     interp.allocate_tensors()
#     inp_det = interp.get_input_details()[0]
#     out_det = interp.get_output_details()[0]

#     interp.set_tensor(inp_det["index"], input_ids)
#     interp.invoke()
#     return interp.get_tensor(out_det["index"])[0]


# # ——— 4) Combine embeddings into a short LLM prompt ———
# def combine_embedding_prompt(img_vec: np.ndarray,
#                              txt_vec: np.ndarray,
#                              raw_text: str) -> str:
#     """
#     Construct a vision-language instruction prompt using image and text embeddings.
#     Emphasizes that inputs are high-dimensional tensors from pretrained encoders.
#     """
#     # Use more values if you want to increase fidelity (e.g., 10 or 20)
#     img_sample = ", ".join(f"{v:.2f}" for v in img_vec[:])
#     txt_sample = ", ".join(f"{v:.2f}" for v in txt_vec[:])

#     return f"""### System:
# You are a vision-language reasoning assistant. You receive as input:
# - A vector embedding extracted from an image using a pretrained CNN encoder.
# - A vector embedding extracted from a user-provided prompt using a pretrained language model.
# - The raw user prompt text (before encoding).

# You will not see the original image or text, only their numeric embeddings and the raw prompt.

# ### User Prompt:
# {raw_text}

# ### Image Embedding (partial vector, float32):
# [{img_sample}]

# ### Text Embedding (partial vector, float32):
# [{txt_sample}]

# ### Task:
# Analyze the semantic relationship between the visual and textual embeddings. Use the raw prompt as reference when relevant. If the embeddings suggest alignment or contrast, explain why. Be precise, reason through the feature space, and use grounded logic.

# ### Answer:"""
    

# # ——— 5) Per-frame end-to-end inference callback ———
# def process_frame(frame_bgr: np.ndarray, prompt_text: str):
#     """
#     Called on every incoming webcam frame (BGR).
#     1) Convert to RGB
#     2) Encode image & text on Edge TPU
#     3) Build prompt + run 4-bit LLM
#     4) Return RGB frame + LLM’s natural language output
#     """
#     if frame_bgr is None:
#         return None, ""

#     # Convert BGR → RGB for correct color
#     rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

#     # Edge TPU encodings
#     img_emb = run_image_encoder(rgb)
#     txt_emb = run_text_encoder(prompt_text)

#     # Build LLM prompt & run inference
#     llm_prompt = combine_embedding_prompt(img_emb, txt_emb, prompt_text)
#     llm_out = run_llm(llm_prompt)

#     return rgb, llm_out


# # ——— 6) Gradio Blocks + webcam streaming ———
# if __name__ == "__main__":
#     with gr.Blocks(title="Real-time VLM Inference: Edge TPU ↔ HuggingFace 4-bit LLM") as demo:
#         gr.Markdown("### Live webcam + text → Edge TPU embeddings → Hugging Face 4-bit LLM → reasoning")

#         with gr.Row():
#             webcam = gr.Image(
#                 sources=["webcam"],  # not auto-streamed now
#                 type="numpy",
#                 label="Webcam Frame"
#             )
#             textbox = gr.Textbox(
#                 label="Text Prompt",
#                 placeholder="Type your description or question here…"
#             )

#         generate_btn = gr.Button("Generate")

#         with gr.Row():
#             out_img = gr.Image(type="numpy", label="RGB Frame")
#             out_txt = gr.Textbox(label="LLM Output")

#         # Trigger processing only when button is clicked
#         generate_btn.click(
#             fn=process_frame,
#             inputs=[webcam, textbox],
#             outputs=[out_img, out_txt]
#         )

#     demo.launch()
#!/usr/bin/env python3
# gradio_vlm_service.py

import cv2
import numpy as np
import gradio as gr

from tflite_runtime.interpreter import Interpreter, load_delegate
from transformers import AutoTokenizer
from llama_cpp import Llama  # pip install llama-cpp-python
from image_encoder import run_image_encoder
from text_encoder import run_text_encoder

# ——— 0) Compression setup ———
# We project the original embeddings into a lower-dimensional space (e.g., 16 dims)
COMPRESSED_DIM = 16
np.random.seed(0)
# Image: 1408 → COMPRESSED_DIM
W_img = (np.random.randn(COMPRESSED_DIM, 1408) * 0.1).astype(np.float32)
b_img = np.zeros(COMPRESSED_DIM, dtype=np.float32)
# Text: 64 → COMPRESSED_DIM
W_txt = (np.random.randn(COMPRESSED_DIM, 64) * 0.1).astype(np.float32)
b_txt = np.zeros(COMPRESSED_DIM, dtype=np.float32)

def compress_embedding(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Apply a linear projection to reduce dimensionality.
    """
    return x @ W.T + b

# ——— 1) Load 4-bit LLM directly from Hugging Face ———
llm = Llama.from_pretrained(
    repo_id="TheBloke/Llama-2-7B-GGUF",      # hosted GGUF model
    filename="llama-2-7b.Q4_0.gguf",         # Q4_0 version
    verbose=False,
    n_ctx=4096,
    n_batch=64,
    n_threads=2,
)

def run_llm(prompt: str) -> str:
    """
    Send the combined prompt to the 4-bit LLM (loaded from Hugging Face)
    and return its generated text.
    """
    resp = llm(
        prompt,
        max_tokens=32,
        temperature=0.2,
        top_p=0.9,
    )
    return resp["choices"][0]["text"]

# # ——— 2) Edge TPU image encoder ———
# def run_image_encoder(image_rgb: np.ndarray,
#                       model_path="1_edgetpu.tflite") -> np.ndarray:
#     """
#     Resize an RGB image to the required shape and type (float32),
#     run through the Edge TPU feature extractor, and return the embedding.
#     """
#     # interpreter = Interpreter(
#     #     model_path=model_path,
#     #     experimental_delegates=[load_delegate("libedgetpu.so.1")]
#     # )
#     try:
#         interpreter = Interpreter(
#             model_path=model_path,
#             experimental_delegates=[load_delegate("libedgetpu.so.1")]
#         )
#     except Exception:
#         print("⚠️ Edge TPU not available, fallback to CPU.")
#         interpreter = Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()
#     inp_det = interpreter.get_input_details()[0]
#     out_det = interpreter.get_output_details()[0]

#     H, W = inp_det["shape"][1], inp_det["shape"][2]
#     resized = cv2.resize(image_rgb, (W, H))

#     # Normalize to [0,1] float32
#     input_data = resized.astype(np.float32) / 255.0
#     input_data = np.expand_dims(input_data, axis=0)  # [1, H, W, 3]

#     interpreter.set_tensor(inp_det["index"], input_data)
#     interpreter.invoke()

#     return interpreter.get_tensor(out_det["index"])[0]

# # ——— 3) Edge TPU text encoder ———
# tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# def run_text_encoder(text: str,
#                      model_path="2_edgetpu.tflite") -> np.ndarray:
#     """
#     Tokenize input text to max_length=384,
#     run through the Edge TPU TFLite encoder,
#     and return a 1D embedding vector.
#     """
#     tokens = tokenizer(
#         text,
#         return_tensors="np",
#         padding="max_length",
#         truncation=True,
#         max_length=384
#     )
#     input_ids = tokens["input_ids"].astype(np.int32)

#     # interp = Interpreter(
#     #     model_path=model_path,
#     #     experimental_delegates=[load_delegate("libedgetpu.so.1")]
#     # )
#     try:
#         interp = Interpreter(
#             model_path=model_path,
#             experimental_delegates=[load_delegate("libedgetpu.so.1")]
#         )
#     except Exception:
#         print("⚠️ Edge TPU not available, fallback to CPU.")
#         interp = Interpreter(model_path=model_path)
#     interp.allocate_tensors()
#     inp_det = interp.get_input_details()[0]
#     out_det = interp.get_output_details()[0]

#     interp.set_tensor(inp_det["index"], input_ids)
#     interp.invoke()
#     return interp.get_tensor(out_det["index"])[0]

# ——— 4) Combine embeddings into a short LLM prompt ———
def combine_embedding_prompt(img_vec: np.ndarray,
                             txt_vec: np.ndarray,
                             raw_text: str) -> str:
    """
    Construct a vision-language instruction prompt using compressed embeddings.
    """
    # Truncate to full compressed dimension
    img_sample = ", ".join(f"{v:.2f}" for v in img_vec.flatten())
    txt_sample = ", ".join(f"{v:.2f}" for v in txt_vec.flatten())

    return f"""### System:
You are a vision-language reasoning assistant. You receive as input:
- A numeric tensor embedding extracted from an image via a pretrained CNN.
- A numeric tensor embedding extracted from user text via a pretrained language model.
- The raw user prompt text (before encoding).

You will not see the original image or raw text, only these compressed embeddings and the prompt.

### User Prompt:
"{raw_text}"

### Compressed Image Embedding ({len(img_vec)} dims):
[{img_sample}]

### Compressed Text Embedding ({len(txt_vec)} dims):
[{txt_sample}]

### Task:
Based on text input and image embedding

### Answer:"""

# ——— 5) Per-frame end-to-end inference callback ———
def process_frame(frame_bgr: np.ndarray, prompt_text: str):
    """
    Called on every button click:
    1) Convert to RGB
    2) Encode image & text on Edge TPU
    3) Compress each embedding
    4) Build prompt + run 4-bit LLM
    5) Return RGB frame + LLM’s response
    """
    if frame_bgr is None:
        return None, ""

    # BGR → RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 2) Raw embeddings
    img_emb_full = run_image_encoder(rgb)
    txt_emb_full = run_text_encoder(prompt_text)

    # 3) Compress embeddings
    img_emb = compress_embedding(img_emb_full, W_img, b_img)
    txt_emb = compress_embedding(txt_emb_full, W_txt, b_txt)

    # Debug prints
    # print("✅ Compressed image embedding preview:", img_emb)
    # print("✅ Compressed text embedding preview:", txt_emb[:10])

    # 4) Build LLM prompt
    llm_prompt = combine_embedding_prompt(img_emb, txt_emb, prompt_text)
    # 5) Run LLM
    llm_out = run_llm(llm_prompt)

    return rgb, llm_out

# ——— 6) Gradio Blocks + manual trigger ———
if __name__ == "__main__":
    with gr.Blocks(
        title="Real-time VLM Inference: Embedding Compression + 4-bit LLM"
    ) as demo:
        gr.Markdown(
            "### Capture an image, enter text, compress embeddings, and run a 4-bit LLM"
        )

        with gr.Row():
            webcam = gr.Image(
                sources=["webcam"],
                type="numpy",
                label="Webcam Frame"
            )
            textbox = gr.Textbox(
                label="Text Prompt",
                placeholder="Type your query here…"
            )

        generate_btn = gr.Button("Generate")

        with gr.Row():
            out_img = gr.Image(type="numpy", label="RGB Frame")
            out_txt = gr.Textbox(label="LLM Output")

        generate_btn.click(
            fn=process_frame,
            inputs=[webcam, textbox],
            outputs=[out_img, out_txt]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)