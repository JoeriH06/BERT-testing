# 🥇 Gold Layer — Multilingual Summarization

The **Gold layer** generates high-quality document summaries from the cleaned and NLP-enriched Silver data.

This stage uses transformer-based sequence-to-sequence models to create summaries, with lightweight language routing to select the most suitable model for each document.

---

## 🎯 Purpose

The Gold layer is responsible for:

- Selecting the best summary input from the document
- Detecting whether the document is primarily **Dutch** or **English**
- Routing the document to an appropriate summarization model
- Generating a summary
- Evaluating summary quality with lightweight heuristics

---

## 🧠 Why mBART?

**mBART** is a multilingual sequence-to-sequence model based on the BART architecture.

Compared with smaller or distilled models, mBART is:

- larger
- slower
- multilingual
- better suited for cross-language or non-English summarization tasks

In this pipeline, mBART is used as part of a **language-aware summarization setup**.

---

## 🔄 Pipeline Before Model Use

```text
Upload PDF
  ↓
Extract to TXT
  ↓
Clean TXT
  ↓
Preprocess for NLP
  ↓
Prepare summary input
  ↓
Generate summary
  ↓
Evaluate summary
```

---

## 🔄 Input / Output

### Input

Silver NLP JSON:

```bash
../../../Data/silver/doc_01_silver_nlp.json
```

Expected fields include:

- `document_id`
- `cleaned_text`
- `sentences`
- `entities`

### Output

Gold summary JSON:

```bash
../../../Data/gold/MBART/multilingual_summary_output.json
```

---

## ⚙️ Model Routing

The pipeline does not use a single summarization model for all documents.

Instead, it routes documents based on detected language:

### Dutch

- Model: `ml6team/mbart-large-cc25-cnn-dailymail-nl`
- Family: `mbart`

### English

- Model: `facebook/bart-large-cnn`
- Family: `bart`

### Fallback

- Model: `csebuetnlp/mT5_multilingual_XLSum`
- Family: `mt5`

This allows the system to use:

- a summarization-tuned Dutch checkpoint for Dutch documents
- a strong English summarization model for English documents
- a multilingual fallback when needed

---

## 🌍 Language Detection

Language detection is done with a lightweight heuristic instead of a separate language ID model.

### Approach

The detector:

- tokenizes the first portion of the text
- scores Dutch hint words
- scores English hint words
- uses small morphology hints
- defaults to English if uncertain

### Example logic

```python
def detect_language(text: str) -> str:
    tokens = simple_tokenize(text[:4000])

    if not tokens:
        return "en"

    dutch_score = sum(1 for t in tokens if t in DUTCH_HINTS)
    english_score = sum(1 for t in tokens if t in ENGLISH_HINTS)

    if dutch_score > english_score + 2:
        return "nl"
    return "en"
```

This keeps the pipeline simple and fast.

---

## 🧹 Pre-Summary Cleanup

Before building model input, the pipeline removes some common document artifacts.

### Examples of removed artifacts

- tracked change remnants
- commented text markers
- page markers like `Page X of Y`
- extra whitespace

### Helper functions

- `remove_editorial_artifacts`
- `normalize_whitespace`
- `safe_string`

These functions make the summarization input more stable and readable.

---

## 🧾 Sentence Filtering

Not every sentence is suitable for summarization.

The pipeline filters out sentences that look like:

- references
- bibliography entries
- table of contents lines
- headings
- external examples
- short or low-information fragments

### Functions involved

- `is_table_of_contents_line`
- `looks_like_reference`
- `looks_like_external_example`
- `is_heading_like`
- `is_good_sentence`

This prevents the model from wasting capacity on noisy or low-value content.

---

## 📊 Sentence Scoring

Candidate sentences are ranked using a heuristic scoring system.

### Factors that increase score

- informative sentence length
- discourse cues such as:
  - `this study`
  - `the findings`
  - `overall`
  - `in conclusion`
- higher lexical variety
- useful document positions

### Factors that decrease score

- short intro-like filler
- date-heavy snippets
- news-like factual fragments

### Functions involved

- `sentence_information_score`
- `sentence_position_score`
- `rank_sentences`

This helps the system focus on content that is more likely to summarize the document well.

---

## 🪟 Window-Based Summary Input Selection

Instead of selecting isolated top sentences, the pipeline builds **adjacent sentence windows**.

This preserves local context and makes the final model input more coherent.

### Why windows?

A single sentence may not contain enough context on its own. By expanding around high-scoring sentences, the pipeline keeps surrounding evidence and improves readability.

### Window logic

- rank sentences
- expand around strong candidates
- keep coverage across:
  - early section
  - middle section
  - late section
- avoid overlapping windows
- merge nearby windows
- keep only high-quality windows

### Main function

```python
build_summary_input_from_windows(...)
```

### Output

- `summary_input` → text sent to the model
- `selected_windows` → debug information showing which sentence ranges were used

---

## ✂️ Length Control

The summary length is dynamically calculated based on document size.

### Strategy

- target summary ≈ **4% of input size**
- convert approximate word count to token count
- clamp output to reasonable min/max limits

### Function

```python
compute_output_length_from_document(original_word_count)
```

This avoids summaries that are too short for long documents or too long for short ones.

---

## 🤖 Summary Generation

The actual summary is generated with Hugging Face transformer models.

### Main function

```python
summarize_text(text, language, original_word_count)
```

### Generation settings

- beam search: `num_beams=5`
- repetition control: `no_repeat_ngram_size=3`
- early stopping enabled
- dynamic `min_length` / `max_length`

### For mBART

When using mBART, the pipeline also sets the target language token so the model generates in the correct language.

---

## 🧼 Summary Cleanup

After generation, the output is lightly cleaned to remove unwanted prefixes and formatting artifacts.

### Examples

- remove leading `summary:`
- remove leading `samenvatting:`
- normalize whitespace
- collapse repeated punctuation

### Function

```python
clean_generated_summary(summary)
```

---

## 📈 Evaluation

The pipeline includes lightweight automatic evaluation metrics.

### 1. Compression Ratio

Measures how much shorter the summary is than the source text.

### 2. Entity Preservation

Checks whether important entities from the source appear in the summary.

Entity types considered include:

- `PERSON`
- `ORG`
- `GPE`
- `DATE`
- `PRODUCT`
- `EVENT`

### 3. Sentence Coverage Score

Estimates how much of the source sentence content is reflected in the summary.

### 4. Summary Redundancy Score

Measures how repetitive the summary is by comparing summary sentences against each other.

### Functions involved

- `evaluate_summary`
- `sentence_coverage_score`
- `summary_redundancy_score`

These metrics are heuristic, but useful for diagnostics.

---

## 🧩 Row-Level Processing

Each document row is summarized through:

```python
summarize_document_row(row)
```

### Steps

1. Read `document_id`, `cleaned_text`, `sentences`, and `entities`
2. Detect language
3. Build summary input from selected sentence windows
4. Fall back to filtered sentences if needed
5. Fall back to raw cleaned text if needed
6. Generate summary
7. Evaluate summary
8. Return structured result

---

## 📦 Output Schema

Example output structure:

```json
{
  "document_id": "doc_01",
  "language_detected": "en",
  "model": "facebook/bart-large-cnn",
  "selected_windows": [],
  "summary_input": "...",
  "summary": "...",
  "evaluation": {
    "original_token_count": 0,
    "summary_token_count": 0,
    "compression_ratio": 0.0,
    "preserved_entities": [],
    "missed_entities": [],
    "entity_preservation_score": 0.0,
    "runtime_seconds": 0.0,
    "sentence_coverage_score": 0.0,
    "summary_redundancy_score": 0.0,
    "model_name": "",
    "model_family": "",
    "input_token_count_to_model": 0,
    "output_token_count_from_model": 0,
    "target_min_tokens": 0,
    "target_max_tokens": 0,
    "original_word_count": 0
  }
}
```

---

## 🧰 Libraries Used

```python
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
```

### Purpose of each library

- `json` → save final summary output
- `re` → cleaning and heuristic rules
- `time` → runtime measurement
- `dataclasses` → sentence window representation
- `pathlib` → file paths
- `pandas` → load Silver NLP JSON
- `torch` → run summarization models
- `transformers` → tokenizer and model loading

---

## 🚀 Main Execution Flow

```python
def main() -> None:
    df = pd.read_json(INPUT_JSON, orient="records")

    if df.empty:
        raise ValueError("Input JSON is empty.")

    row = df.loc[0]
    result = summarize_document_row(row)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
```

This script currently processes the **first document row** in the input file.

---

## ✅ Why This Design Works

This summarization pipeline works well because it combines:

- **document cleanup**
- **language-aware model routing**
- **sentence quality filtering**
- **window-based context selection**
- **dynamic length control**
- **basic evaluation metrics**

Rather than sending the full raw document directly into the model, it first selects the most relevant and coherent content.

This improves both efficiency and summary quality.

---

## ⚠️ Limitations

This approach is useful, but it has several limitations.

### 1. Heuristic Language Detection

The language detector is lightweight and may fail on:

- mixed-language documents
- short texts
- domain-specific terminology

### 2. Sentence Filtering Is Rule-Based

Good sentences may be filtered out, and noisy sentences may occasionally pass through.

### 3. Window Selection Is Heuristic

The selected context is not guaranteed to be globally optimal.

### 4. Model Limits Still Apply

Even with input selection, summarization is still constrained by:

- token limits
- model bias
- hallucination risk
- imperfect multilingual support

### 5. Evaluation Is Approximate

The current metrics are heuristic diagnostics, not formal benchmark metrics like ROUGE or BERTScore.

---

## 🔧 Future Improvements

Possible next steps for this layer include:

- add true language identification
- support more languages beyond Dutch and English
- compare summaries with ROUGE / BERTScore
- process all rows instead of only `df.loc[0]`
- add batching for faster inference
- improve entity-aware sentence selection
- try larger or more specialized summarization models
- add GPU/CPU performance logging
- save more debugging artifacts for analysis

---

## 📝 Full Workflow Summary

```text
Silver NLP JSON
  ↓
Language detection
  ↓
Sentence filtering and scoring
  ↓
Window selection
  ↓
Model-specific summarization
  ↓
Summary cleanup
  ↓
Evaluation
  ↓
Gold summary JSON
```

---

## 🔗 Position in the Full Pipeline

```text
Bronze → Silver → Silver NLP → Gold Summary
```

The Gold layer is the stage where cleaned and structured document content becomes a final **usable summary output**.