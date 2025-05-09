from image_encoder import run_image_encoder
from text_encoder import run_text_encoder

def combine_embedding_prompt(image_vec, text_vec):
    # 示例：拼接为可读 prompt，或送入 LLM 作为嵌入
    prompt = "IMAGE_EMBEDDING: " + " ".join(f"{x:.4f}" for x in image_vec[:10]) + " ...\n"
    prompt += "TEXT_EMBEDDING: " + " ".join(f"{x:.4f}" for x in text_vec[:10]) + " ...\n"
    prompt += "Please describe the relationship or context between the image and text."
    return prompt

if __name__ == "__main__":
    img_embedding = run_image_encoder("example.jpg")
    txt_embedding = run_text_encoder("A person wearing a red jacket is walking a dog.")

    prompt = combine_embedding_prompt(img_embedding, txt_embedding)

    print("🧠 LLM Prompt:\n", prompt)

    # TODO: 可调用本地 LLM（llama.cpp、ollama、GPT4All）处理该 prompt