# test_flickr30k_dataset.py

import cv2
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from datasets import load_dataset
from gradio_vlm_service import process_frame  # Your VLM inference function

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
        # 1. Sample an image-text pair (positive sample)
        sample = random.choice(dataset)
        image_path = sample["image"]["path"]
        text = random.choice(sample["sentences"])["raw"]
        label = 1

        # 2. Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[Warning] Failed to load image: {image_path}")
            continue

        # 3. Occasionally create a negative sample by replacing the text
        if random.random() < 0.5:
            label = 0
            other = random.choice(dataset)
            while other["image"]["path"] == image_path:
                other = random.choice(dataset)
            text = random.choice(other["sentences"])["raw"]

        # 4. Inference
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