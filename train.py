import torch
import os
import numpy as np
from src.model import GPTModel

# T4 Optimized Configuration
config = {
    "vocab_size": 50304,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": False
}

batch_size = 8
gradient_accumulation_steps = 20
max_iters = 1000
learning_rate = 6e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(split, data_dir='data'):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config["context_length"], (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + config["context_length"]]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + config["context_length"]]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def train():
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs('out-NanoLLM', exist_ok=True)

    model.train()
    for iter_num in range(max_iters):
        x, y = get_batch('train')

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if iter_num % 10 == 0:
            print(f"Step {iter_num}: Loss {loss.item() * gradient_accumulation_steps:.4f}")

    torch.save(model.state_dict(), 'out-NanoLLM/model_final.pt')
    print("Training complete. Checkpoint saved.")


if __name__ == "__main__":
    train()