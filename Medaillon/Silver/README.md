# 🥈 Silver Layer

The **Silver layer** transforms raw text from the Bronze layer into clean, structured, and NLP-ready formats.

It consists of two stages:

1. **Text Cleaning & Normalization** → `silver.ipynb`
2. **NLP Enrichment** → `silver_nlp.ipynb`

---

# 📄 silver.ipynb — Text Cleaning Pipeline

This step focuses on converting noisy PDF-extracted text into usable formats.

## 🔄 Input / Output

- **Input:** Bronze JSON (`raw_text`)
- **Output:** Silver JSON with:
  - `cleaned_text` (reading-friendly)
  - `nlp_text` (structure-preserving)

---

## 🧹 Cleaning Strategy

The pipeline applies a sequence of transformations:

### Core Cleaning Steps

- Normalize line endings
- Replace tabs and non-breaking spaces
- Remove zero-width characters
- Fix hyphenated line breaks
- Remove trailing spaces
- Remove repeated spaces
- Normalize spacing around line breaks
- Merge broken PDF lines (heuristic-based)
- Remove PDF page markers
- Remove reference-like lines
- Remove duplicate lines
- Reduce excessive blank lines

---

## ✨ Two Output Variants

### 1. `cleaned_text` (Reading-Friendly)

Optimized for:
- Summarization
- LLM input
- Human readability

Includes:
- Line merging
- Aggressive cleanup
- Noise reduction

---

### 2. `nlp_text` (Structure-Preserving)

Optimized for:
- NLP pipelines
- Entity extraction
- Metadata extraction (front-matter logic)

Keeps:
- Line structure
- Sentence boundaries

---

## ⚙️ Key Functions

### Safety Wrapper

```python
def safe_string(text):
    return "" if pd.isna(text) else str(text)
```

---

### Cleaning Pipeline (Reading)

```python
def clean_text_for_reading(text):
    text = normalize_line_endings(text)
    text = replace_special_spaces(text)
    text = remove_zero_width_characters(text)
    text = fix_hyphenated_linebreaks(text)
    text = remove_pdf_page_markers(text)
    text = remove_urls_and_emails(text)
    text = remove_trailing_spaces(text)
    text = remove_repeated_spaces(text)
    text = normalize_linebreak_spacing(text)
    text = merge_broken_lines(text)
    text = remove_reference_like_lines(text)
    text = remove_duplicate_lines(text)
    text = remove_many_blank_lines(text)
    return text.strip()
```

---

### Cleaning Pipeline (NLP)

```python
def clean_text_for_nlp(text):
    text = normalize_line_endings(text)
    text = replace_special_spaces(text)
    text = remove_zero_width_characters(text)
    text = fix_hyphenated_linebreaks(text)
    text = remove_pdf_page_markers(text)
    text = remove_urls_and_emails(text)
    text = remove_trailing_spaces(text)
    text = remove_repeated_spaces(text)
    text = normalize_linebreak_spacing_soft(text)
    text = remove_reference_like_lines(text)
    text = remove_many_blank_lines(text)
    return text.strip()
```

---

## 🧪 Quality & Sanity Checks

- Preview original vs cleaned text
- Track:
  - `raw_length`
  - `cleaned_length`
  - `nlp_length`
- Compute ratios:
  - `cleaned_ratio`
  - `nlp_ratio`
- Flag unusually short outputs

---

## 💾 Output

Saved as:

```bash
../../Data/silver/doc_01_silver.json
```

---

# 🧠 silver_nlp.ipynb — NLP Enrichment

This step enriches cleaned text with NLP features using **spaCy**.

---

## 🔄 Input / Output

- **Input:** Silver JSON (`cleaned_text`, `nlp_text`)
- **Output:** Silver NLP JSON with:
  - tokens
  - sentences
  - named entities

---

## ⚙️ NLP Pipeline

### Library

```python
import spacy
nlp = spacy.load("en_core_web_sm")
```

Provides:
- Tokenization
- Lemmatization
- POS tagging
- Named Entity Recognition (NER)

---

## 🔤 Token Processing

```python
def preprocess_tokens(doc):
    tokens = []
    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.is_alpha
            and len(token.text) > 2
        ):
            tokens.append(token.lemma_.lower())
    return tokens
```

Filters:
- Stopwords
- Punctuation
- Short/noisy tokens

---

## 🧾 Sentence Extraction

```python
def extract_sentences(doc):
    return [sent.text.strip() for sent in doc.sents if valid_sentence(sent.text)]
```

Valid sentence criteria:
- ≥ 40 characters
- ≥ 8 words

---

## 🏷️ Named Entity Recognition

```python
def extract_entities(doc):
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]
```

Examples:
- PERSON
- ORG
- DATE
- GPE

---

## 🔁 Full Row Processing

```python
def process_nlp_row(row):
    text_for_nlp = str(row.get("nlp_text") or row.get("cleaned_text", ""))
    cleaned_text = str(row.get("cleaned_text", ""))

    doc = nlp(text_for_nlp)

    return {
        "document_id": row.get("document_id"),
        "cleaned_text": cleaned_text,
        "nlp_text": text_for_nlp,
        "tokens": preprocess_tokens(doc),
        "sentences": extract_sentences(doc),
        "entities": extract_entities(doc)
    }
```

---

## 📊 Metrics & Diagnostics

Tracks per document:
- `n_tokens`
- `n_sentences`
- `n_entities`

Provides:
- Distribution stats
- Row-level previews

---

## 💾 Output

Saved as:

```bash
../../Data/silver/doc_01_silver_nlp.json
```

---

# ✅ Summary

| Layer        | Purpose                         | Output                     |
|-------------|----------------------------------|----------------------------|
| `silver`     | Clean & normalize text          | cleaned_text, nlp_text     |
| `silver_nlp` | Add NLP features                | tokens, sentences, entities|

---

# 🚀 Key Design Principles

- **Separation of concerns**
  - Cleaning vs NLP processing

- **Dual text strategy**
  - Reading vs structure-preserving

- **Robust preprocessing**
  - Handles noisy PDF artifacts

- **Reusable NLP features**
  - Enables downstream metadata extraction

---

# ⚠️ Limitations

- Dependent on PDF extraction quality
- Heuristics may fail on:
  - Complex layouts
  - Tables
  - Multi-column PDFs
- spaCy small model:
  - Limited accuracy vs larger models

---

# 🔧 Future Improvements

- Upgrade to larger spaCy model (`en_core_web_trf`)
- Add language detection
- Improve sentence segmentation
- Add custom NER models
- Handle multi-column layouts better
- Improve reference detection
