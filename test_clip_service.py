# test_flickr30k_dataset.py

import cv2
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from datasets import load_dataset
from gradio_vlm_service import process_frame  # Your VLM inference function
from PIL import Image


# --- Parse LLM output to binary label ---
def parse_match_from_llm_output(llm_out: str) -> int:
    output = llm_out.lower()
    if "not match" in output or "not related" in output or "no" in output:
        return 0
    elif "match" in output or "related" in output or "yes" in output:
        return 1
    else:
        return 0  # default fallback

# --- Main evaluation function using HuggingFace dataset ---
def evaluate_flickr30k(n_samples=100):
    dataset = load_dataset("nlphuji/flickr30k", split="test")

    y_true, y_pred = [], []
    total = min(n_samples, len(dataset))

    for _ in tqdm(range(total), desc="Evaluating"):
        sample = random.choice(dataset)
        image_pil = sample["image"]
        image = np.array(image_pil.convert("RGB"))
        text = sample["caption"]
        text = sample["caption"]
        if isinstance(text, list):
            text = random.choice(text)
        label = 1

        if random.random() < 0.5:
            label = 0
            other = random.choice(dataset)
            while other["image"] == sample["image"]:
                other = random.choice(dataset)
            text = other["caption"]
            if isinstance(text, list):
                text = random.choice(text)

        _, llm_out = process_frame(image, text)
        pred = parse_match_from_llm_output(llm_out)

        y_true.append(label)
        y_pred.append(pred)

    # --- Report ---
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=3))
    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Accuracy: {acc * 100:.2f}%")

# --- Entry point ---
if __name__ == "__main__":
    evaluate_flickr30k(n_samples=100)