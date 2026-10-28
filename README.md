# Comparative Sentiment Classification: Bidirectional RNN vs. LSTM

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter&logoColor=white)](Code_UAS_NLP_Kelompok1.ipynb)
[![Dataset](https://img.shields.io/badge/Dataset-IMDB%2050K-blue)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
[![Task](https://img.shields.io/badge/Task-Binary%20Sentiment%20Classification-success)]()

---

## Project Overview

This study presents an empirical benchmarking between **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** networks for binary sentiment classification on the large-scale **IMDB Movie Reviews dataset**. 

Beyond standard accuracy metrics, this project investigates underlying deep learning mechanics, specifically:
1. **Vanishing and Exploding Gradients:** Evaluating gradient norm dynamics across layers to observe backpropagation through time (BPTT) stability.
2. **Long-Term Dependency Preservation:** Quantifying sequence length impact on classification error rates, identifying exact thresholds where standard RNN context decays compared to gated LSTM memory cells.

---

## Dataset and Preprocessing Pipeline

* **Dataset:** IMDB Large Movie Review Dataset (50,000 balanced binary reviews: 25,000 positive, 25,000 negative).
* **Text Cleansing:** Stripped HTML tags, URLs, special punctuation characters, redundant whitespace, and enforced lowercasing.
* **Tokenization and Sequencing:**
  * Vocabulary Size: 20,000 most frequent tokens (with Out-Of-Vocabulary `<OOV>` token handling).
  * Sequence Length: Padded and truncated to a uniform fixed sequence length of 260 tokens.
  * Data Partitioning: Stratified 80% Training, 10% Validation, and 10% Testing (4,959 test samples).

---

## Model Architectures and Hyperparameters

Both sequential models were implemented natively in **PyTorch** using identical structural dimensions to isolate architectural differences:

| Parameter / Layer | Bidirectional RNN Configuration | Bidirectional LSTM Configuration |
| :--- | :--- | :--- |
| **Embedding Layer** | Embedding Dimension = 128 | Embedding Dimension = 128 |
| **Recurrent Layer** | Bidirectional Simple RNN (Hidden Dim = 64) | Bidirectional LSTM (Hidden Dim = 64) |
| **Regularization** | Dropout ($p = 0.5$) | Dropout ($p = 0.5$) |
| **Classification Head** | Dense Linear Layer $\rightarrow$ BCEWithLogitsLoss | Dense Linear Layer $\rightarrow$ BCEWithLogitsLoss |
| **Optimizer & Schedule** | Adam ($\text{lr} = 0.001$), 50 Epochs | Adam ($\text{lr} = 0.001$), 50 Epochs |

---

## Comparative Performance Evaluation

Evaluated across 4,959 unseen test samples:

| Architecture | Test Accuracy | Precision (Neg / Pos) | Recall (Neg / Pos) | Macro F1-Score | Training Stability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Bidirectional RNN** | **78.0%** | 0.77 / 0.80 | 0.81 / 0.76 | **0.78** | Overfits after Epoch 6; gradient spikes in early epochs |
| **Bidirectional LSTM** | **87.0%** | 0.87 / 0.87 | 0.86 / 0.87 | **0.87** | Smooth convergence; monotonic gradient norm stabilization |

---

## Deep Learning Diagnostic Analysis

### 1. Gradient Flow and Layer Stability
* **RNN Instability:** Demonstrates erratic gradient norm spikes (notably at Epoch 4 within the recurrent layer), indicating susceptibility to gradient explosion and unstable weight updates during BPTT.
* **LSTM Gating Defense:** Gating units (input, forget, and output gates) regulate internal cell state updates, producing smooth, monotonically decreasing gradient norms across all layers.

### 2. Failure Analysis on Long-Term Dependencies
* **Long Review Fragility:** In misclassified reviews, average sequence length was significantly higher (247–259 words) than in accurately classified reviews (223–224 words).
* **The 685 Reversal Case:** A granular case-by-case comparison identified **685 reviews where LSTM predicted correctly while RNN failed**. The failure distribution peaks sharply between 150 to 300+ words, validating that simple RNN hidden states suffer from severe recency bias, while LSTM successfully preserves early contextual signals across long text sequences.

---

## Summary of Findings

1. **Architectural Superiority:** The Bidirectional LSTM outperformed the standard RNN by **+9.0% test accuracy**, delivering balanced 0.87 precision and recall across both sentiment polarities.
2. **Robustness to Long Text:** LSTM is markedly superior for document-level sentiment tasks where sentiment nuance is distributed across multiple paragraphs.
3. **Engineering Takeaway:** While standard RNNs require fewer floating-point operations per step, their gradient vulnerability and limited memory retention make LSTMs (or Transformer-based self-attention models) the necessary choice for sequence modeling.
