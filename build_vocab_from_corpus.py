import json
from collections import Counter
from datasets import load_dataset

# === Configurations ===
corpus_name = "ag_news"           # Hugging Face dataset name
text_field = "text"               # Field to extract sentences from
max_vocab_size = 10000            # Including special tokens
output_vocab_file = "vocab.json"  # Output file

# === Step 1: Load public corpus ===
print(f"📥 Downloading corpus: {corpus_name}")
dataset = load_dataset(corpus_name, split="train")  # Use training split

# === Step 2: Extract and tokenize sentences ===
print("🧹 Tokenizing text...")
counter = Counter()
for item in dataset:
    sentence = item[text_field].lower().strip()
    tokens = sentence.split()  # Simple whitespace tokenizer
    counter.update(tokens)

# === Step 3: Build word → ID mapping ===
vocab = {
    "PAD": 0,
    "UNK": 1
}
for i, (word, _) in enumerate(counter.most_common(max_vocab_size - 2), start=2):
    vocab[word] = i

# === Step 4: Save to vocab.json ===
with open(output_vocab_file, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

print(f"✅ Vocab saved to '{output_vocab_file}' (size: {len(vocab)})")