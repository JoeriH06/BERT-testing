# 📄 Manuel

## 📌 Project Overview

This project evaluates multiple NLP models on large text documents to determine which model produces the most effective summaries.

We use the **spaCy** library for preprocessing and experiment with various **BERT** and **BART-based transformer models** for summarization and downstream tasks.

---

## 🧱 Pipeline Structure

The pipeline structure that is being used in this project is the medaillon structure.
This structure divides a datascience project into three different layers, Bronze, Silver, and Gold.
Each layer has its own purpose, where bronze is dataretrieval, silver is cleaning and preprocessing, and gold is modeling.
This structure makes sure that everything is nicely formatted and easily traceable

---

### 🥉 1. Bronze Layer (Data Retrieval)

In the Bronze layer, raw data is collected.

- Input data is uploaded by users in **PDF format**
- PDFs are converted into **plain text (.txt)** files
- This conversion enables easier data extraction, cleaning, and readability

---

### 🥈 2. Silver Layer 1 (Text Cleaning)

In this stage:

- The text data from the Bronze layer is loaded into a dataframe
- Text is cleaned and normalized
- Preparation steps are applied to support NLP preprocessing

The cleaned output is saved for further processing.

---

### 🥈 3. Silver Layer 2 (NLP Preprocessing)

In this layer:

- Advanced NLP preprocessing is performed using **spaCy**
- Tasks may include:
  - Tokenization
  - Named Entity Recognition (NER)
  - Part-of-speech tagging
  - Linguistic annotations

These annotations are later used to enhance model performance.

---

### 🥇 4. Gold Layer (Transformer-Based Processing)

In the Gold layer, transformer models (e.g., **BERT**, **BART**) are used to extract key information and generate summaries.

Outputs include:

- Important text segments
- Named entities (contributors, organizations)
- Publication or release dates
- Semantic representations for downstream analysis

spaCy annotations from the Silver layer are used as supporting signals.

---

## 📊 Model Comparison (Summarization Performance)

| Model          | Summary Quality ⭐ | Entity Preservation | Compression Ratio | Runtime (s) | Strengths                             | Weaknesses                      |
| -------------- | ------------------ | ------------------- | ----------------- | ----------- | ------------------------------------- | ------------------------------- |
| **mBART**      | ⭐⭐⭐⭐⭐         | 0.67                | 0.49              | 8.50        | High-quality, well-balanced summaries | Slower runtime                  |
| **BERT**       | ⭐⭐⭐⭐☆          | 0.67                | 0.68              | 0.18        | Very accurate, extremely fast         | Less concise (weak compression) |
| **BART**       | ⭐⭐☆☆☆            | 0.11                | 0.32              | 3.59        | Produces short summaries              | Loses important details         |
| **distilBART** | ⭐☆☆☆☆             | 0.11                | 0.30              | 1.98        | Fast and lightweight                  | Poor information retention      |

---

## 🤖 Model Descriptions

### **mBART (Multilingual BART)**

A transformer model designed for **multilingual text generation and summarization**.  
It produces fluent, coherent, and high-quality summaries across different languages.

---

### **BERT (Bidirectional Encoder Representations from Transformers)**

Primarily designed for **text understanding**, not generation.  
Used here for **extractive summarization**, selecting the most relevant sentences.

---

### **BART (Bidirectional and Auto-Regressive Transformers)**

A hybrid model combining BERT-style understanding with GPT-style generation.  
Commonly used for **abstractive summarization** (rewriting text concisely).

---

### **distilBART**

A **compressed version of BART** created via knowledge distillation.  
Optimized for speed and efficiency, but with reduced accuracy.

---

## 📐 Evaluation Metrics

### ⭐ Summary Quality

A qualitative rating based on:

- Preservation of key information
- Conciseness
- Readability and coherence

---

### 🧾 Entity Preservation

Measures how well the model retains important entities such as:

- People
- Organizations
- Project names

👉 Higher score = better retention of critical details

---

### 📉 Compression Ratio

- Lower value → shorter summary (more compression)
- Higher value → longer summary (less compression)

👉 Ideal balance: concise but still informative

---

### ⏱ Runtime (seconds)

The time required to generate a summary.

👉 Lower value = faster model
