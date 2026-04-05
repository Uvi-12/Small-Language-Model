import os
import numpy as np
import tiktoken
import tqdm
from datasets import load_dataset


def distillation_filter(example):
    keywords = [
        "artificial intelligence", "neural network", "machine learning",
        "deep learning", "transformer model", "backpropagation",
        "supervised learning", "reinforcement learning", "python code",
        "gradient descent", "activation function"
    ]
    if any(k in example['text'].lower() for k in keywords):
        return example['text']
    return None


def build_dataset(target_tokens=30_000_000, out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)
    print("Streaming Cosmopedia...")

    dataset = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", streaming=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']

    train_ids, val_ids = [], []
    token_count = 0
    pbar = tqdm.tqdm(total=target_tokens, desc="Distilling Tokens")

    for example in dataset:
        text = distillation_filter(example)
        if not text:
            continue

        tokens = enc.encode(text) + [eot]
        if token_count % 10 == 0:
            val_ids.extend(tokens)
        else:
            train_ids.extend(tokens)

        token_count += len(tokens)
        pbar.update(len(tokens))
        if token_count >= target_tokens:
            break

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(out_dir, 'train.bin'))
    val_ids.tofile(os.path.join(out_dir, 'val.bin'))
    print("Data distillation complete and saved to binary.")


if __name__ == "__main__":
    build_dataset()