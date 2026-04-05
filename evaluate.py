import torch
import math
import numpy as np
import os
import statistics
from src.model import GPTModel

config = {
    "vocab_size": 50304, "context_length": 1024, "emb_dim": 768,
    "n_heads": 12, "n_layers": 12, "drop_rate": 0.0, "qkv_bias": False
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate():
    print("Loading Model...")
    model = GPTModel(config).to(device)

    ckpt_path = 'out-NanoLLM/model_final.pt'
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Run train.py first.")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    val_data = np.memmap('data/val.bin', dtype=np.uint16, mode='r')
    losses = []
    accuracies = []

    print("Calculating PPL & Accuracy...")
    with torch.no_grad():
        for _ in range(100):
            ix = torch.randint(len(val_data) - 1024, (1,))
            x = torch.from_numpy((val_data[ix:ix + 1024]).astype(np.int64)).unsqueeze(0).to(device)
            y = torch.from_numpy((val_data[ix + 1:ix + 1 + 1024]).astype(np.int64)).unsqueeze(0).to(device)

            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())

            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            acc = (pred == y).float().mean().item()
            accuracies.append(acc)

    avg_loss = statistics.mean(losses)
    perplexity = math.exp(avg_loss)
    avg_acc = statistics.mean(accuracies) * 100

    print("\nOfficial Benchmark Report")
    print("-" * 30)
    print(f"Perplexity (PPL):     {perplexity:.2f}")
    print(f"Validation Loss:      {avg_loss:.4f}")
    print(f"Next-Token Accuracy:  {avg_acc:.2f}%")


if __name__ == "__main__":
    evaluate()