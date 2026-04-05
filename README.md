Here is the absolute final, polished `README.md` file, combining your published research, the engineering constraints you solved, and the "System-First" repository structure we built. 

It is perfectly formatted and ready to be copied and pasted directly into GitHub.

```markdown
# 🧠 125M NanoLLM: Domain-Specific Small Language Model

**Research Publication**
The architecture, distillation methodology, and evaluation of this model have been published in *Discover Computing* (Springer Nature).
* **Paper:** Nano large language model of 125 million-parameter for STEM education
* **DOI:** [10.1007/s10791-026-10033-z](https://doi.org/10.1007/s10791-026-10033-z)

---

## 📌 Overview
A lightweight, 125M-parameter Large Language Model engineered and trained entirely from scratch using PyTorch. This repository demonstrates the end-to-end pipeline of building a foundational model, from data distillation to constrained compute optimization. 

Unlike standard ~100M parameter models (like GPT-2 Small or DistilGPT-2) that often hallucinate or default to generic web text on technical prompts, this NanoLLM was built using **Data-Centric Knowledge Distillation**. It is designed to punch above its weight class in domain-specific tasks while running efficiently on consumer hardware.

## 🏗️ Architecture & Engineering
* **Parameters:** ~125M (12 layers, 12 attention heads, 768 embedding dimension)
* **Dataset:** Distilled from a strictly filtered synthetic textbook corpus (Cosmopedia), focusing heavily on STEM, AI, and Machine Learning concepts to inject deep domain knowledge.
* **Hardware Optimization:** Engineered to train efficiently using PyTorch mixed precision (`bf16`/`float16`) and gradient accumulation, stabilizing the training loop on highly constrained environments (Apple M1 and single NVIDIA Tesla T4).
* **Core Mechanics:** Transformer blocks implement standard GPT-2 style auto-regressive mechanics, configured for edge efficiency.

## 📊 Performance vs. Baselines
Benchmarked against standard open-source models on the validation set, this model proves that smaller architectures can achieve expert-level syntax when fine-tuned on targeted data.

| Metric | NanoLLM (125M) | GPT-2 Small (124M) | DistilGPT-2 (82M) |
| :--- | :--- | :--- | :--- |
| **Perplexity (PPL)** | **16.06** | 28.5 | 35.0 |
| **Next-Token Accuracy** | **72.8%** | 29.0% | 25.4% |
| **Inference Speed** | **122 tokens/sec** | 115 tokens/sec | 138 tokens/sec |

*(Tested on a single NVIDIA T4 GPU)*

## 📂 Repository Structure
This repository is structured as a deployable software package rather than a flat list of scripts.

* `src/`: Core neural network definitions (`model.py`) and tokenizer logic.
* `notebooks/`: Exploratory block-by-block prototyping, architecture testing, and data validation.
* `data_pipeline.py`: The token distillation and streaming engine.
* `train.py`: The mixed-precision execution script for hardware-constrained training.
* `evaluate.py`: Industry-standard benchmarking logic (PPL, Accuracy, BLEU, ROUGE).

---

## 🚀 Getting Started

### Option 1: Run via Hugging Face (Recommended)
If you want to test the model without training it from scratch or downloading the raw binaries, the pre-trained weights are hosted on Hugging Face.

```python
# Install the Hugging Face Hub
pip install huggingface_hub

# Download the model weights directly
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Uv-27/NanoLLM-125", filename="model_final.pt")
print(f"Model downloaded to: {model_path}")
```

### Option 2: Fork and Run Locally
To replicate the environment, run the evaluation benchmarks, or train the model yourself:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Small-Language-Model.git](https://github.com/YOUR_USERNAME/Small-Language-Model.git)
   cd Small-Language-Model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Run the Data Pipeline:**
   To stream and distill the dataset locally:
   ```bash
   python data_pipeline.py
   ```

4. **Evaluate the Model:**
   Ensure you have downloaded the weights into the `out-NanoLLM/` directory, then run the evaluation script to calculate Perplexity and Accuracy:
   ```bash
   python evaluate.py
   ```
```

