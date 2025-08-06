# Neural-Networks-for-Data-Science-Applications-
This directory houses my final project in the Neural Networks for Data Science Applications course.


# 🧠 Character-Level RNN with JAX – Text Generation from WikiText-2

This project implements a Recurrent Neural Network (RNN) in **JAX** from scratch to perform **language modeling** and **text generation** on the WikiText-2 dataset.  
I achieved full marks (15/15) on this project in the NNDS course.

---

## 🔍 Project Description

The main goals of this project were:

- Build a basic RNN architecture using **pure JAX** (no PyTorch, no TensorFlow).
- Train it to predict the next word given a sequence of previous words.
- Evaluate the model's loss over time on both training and validation sets.
- Use the trained model for **autoregressive text generation**.
- Improve generation using **beam search** with:
  - ✅ Top-K candidate expansion
  - ✅ Early stopping based on repetition detection
  - ✅ Temperature scaling for diversity

---

## 📦 Dataset

- **[WikiText-2 (raw)](https://huggingface.co/datasets/wikitext)**: a clean corpus of Wikipedia articles, suitable for training small language models.
- Provided by 🤗 Hugging Face `datasets` library.

---

## 🛠️ Technologies Used

- `jax`, `jax.numpy` – for all tensor operations and autodiff
- `huggingface/datasets` – to load and preprocess WikiText-2
- `matplotlib` – for plotting training/validation loss

---

## 🧠 Model Architecture

- **Embedding layer**: maps each word token to a dense vector
- **RNN cell**: simple recurrent unit with:
  - Hidden state update: `h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)`
  - Output: `y_hat = Why * h_t + by`
- **Loss**: cross-entropy over the vocabulary
- **Optimizer**: manual SGD using `jax.grad`

---

## 📈 Training

- Input: sequences of 10 words → model predicts the 11th
- Training size limited to 10,000 samples due to memory
- 10 epochs over batch size 16
- Training and validation loss are plotted to monitor convergence

---

## ✨ Text Generation

Implemented two generation methods:

1. **Greedy Sampling** (default):
   - Autoregressively generates one token at a time using argmax
   - Controlled with **temperature** scaling for randomness

2. **Beam Search (Improved)**:
   - Keeps top-K likely sequences at each step
   - Uses early stopping if repetition is detected
   - Allows exploration with more diversity and coherence

> Sample prompt: `"the king"`  
> Output (with temperature and beam search):  
> `"the king to to to to to..."`  
> (Yes – limited creativity, but shows model is learning transitions!)

---

## 📊 Results

- ✅ Successfully trained an RNN from scratch in JAX
- ✅ Tracked train and validation loss with clean convergence
- ✅ Generated novel text from prompts
- ✅ Implemented advanced decoding with beam search

Despite the limited training data and small embedding/hidden dimensions, the model learns reasonable language patterns.

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -U jax datasets matplotlib huggingface_hub fsspec
