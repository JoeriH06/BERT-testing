## 📊 Batch Reports

    ../../Data/silver/reports/silver_validation_report.json
    ../../Data/silver/reports/silver_manifest.json

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
- Remove URLs and emails  
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
- Metadata extraction  

Keeps:

- Line structure  
- Sentence boundaries  

---

## ⚙️ Key Functions

### Safety Wrapper

    def safe_string(text):
        return "" if pd.isna(text) else str(text)

---

### Cleaning Pipeline (Reading)

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

---

### Cleaning Pipeline (NLP)

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

---

## 🧪 Quality & Sanity Checks

### Per document:

- raw_length  
- cleaned_length  
- nlp_length  
- cleaned_ratio  
- nlp_ratio  

### Quality Flags

Automatically detects:

- very short outputs  
- aggressive cleaning  
- empty inputs  

### Diagnostics

- Preview original vs cleaned text  
- Batch-level statistics  
- Validation report generation  

---

## 📊 Output Schema (Silver)

    {
      "document_id": "doc_01",
      "source_file_name": "example.pdf",
      "raw_text": "...",
      "cleaned_text": "...",
      "nlp_text": "...",
      "pdf_metadata": {},
      "page_count": 40,
      "cleaned_length": 12000,
      "nlp_length": 11800,
      "cleaned_ratio": 0.92,
      "nlp_ratio": 0.90,
      "quality_flags": []
    }

---

## 🧠 silver_nlp.ipynb — NLP Enrichment

This step enriches cleaned text with NLP features using spaCy.

---

## 🔄 Input / Output

### Input

Silver JSON:

- cleaned_text  
- nlp_text  
- metadata  

### Output

Silver NLP JSON:

- tokens  
- sentences  
- named entities  
- NLP metrics  

Saved as:

    ../../Data/silver/doc_XX_silver_nlp.json

Reports:

    ../../Data/silver/reports/silver_nlp_validation_report.json
    ../../Data/silver/reports/silver_nlp_manifest.json

---

## ⚙️ NLP Pipeline

### Library

    import spacy
    nlp = spacy.load("en_core_web_sm")

Provides:

- Tokenization  
- Lemmatization  
- POS tagging  
- Named Entity Recognition (NER)  

---

## 🔤 Token Processing

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

---

## 🧾 Sentence Extraction

    def extract_sentences(doc):
        return [sent.text.strip() for sent in doc.sents if valid_sentence(sent.text)]

Valid sentence criteria:

- ≥ 40 characters  
- ≥ 8 words  

---

## 🏷️ Named Entity Recognition

    def extract_entities(doc):
        return [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]

---

## 📊 NLP Metrics

Per document:

- n_tokens  
- n_sentences  
- n_entities  
- entity_label_counts  

### NLP Quality Flags

- low token count  
- no valid sentences  
- no entities detected  

---

## 📦 Output Schema (Silver NLP)

    {
      "document_id": "doc_01",
      "tokens": ["arbeidsmarkt", "ontwikkeling"],
      "sentences": ["The labor market is evolving rapidly..."],
      "entities": [
        {"text": "Zeeland", "label": "GPE"}
      ],
      "n_tokens": 1200,
      "n_sentences": 85,
      "n_entities": 42,
      "entity_label_counts": {
        "ORG": 10,
        "DATE": 5
      },
      "nlp_quality_flags": []
    }

---

## ✅ Summary

| Layer        | Purpose                    | Output                              |
|-------------|----------------------------|-------------------------------------|
| silver       | Clean & normalize text     | cleaned_text, nlp_text, metrics     |
| silver_nlp   | Add NLP features           | tokens, sentences, entities, metrics|

---

## 🚀 Key Design Principles

- Separation of concerns (cleaning vs NLP)  
- Batch-first architecture  
- Traceability across layers  
- Dual text strategy  
- Quality monitoring  

---

## ⚠️ Limitations

- Dependent on PDF extraction quality  
- Heuristics may fail on complex layouts  
- Tables and multi-column PDFs  
- spaCy small model limitations (especially for Dutch)  

---

## 🔧 Future Improvements

- Add language detection  
- Support Dutch model (`nl_core_news_sm`)  
- Upgrade to transformer models  
- Improve sentence segmentation  
- Add custom NER  
- Remove raw_text from Silver  
- Improve layout-aware cleaning  