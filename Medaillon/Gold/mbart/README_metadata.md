# 📄 Metadata Generation Pipeline

This pipeline generates structured metadata from documents **after a PDF has been converted into cleaned text**.

It combines rule-based extraction with lightweight NLP to produce consistent metadata across a wide range of document types.

---

## 🔄 Pipeline Flow

```
PDF → Text Extraction → Cleaning → NLP Preprocessing → Metadata Extraction → Classification → JSON Output
```

---

## 🧠 What This Pipeline Actually Uses

Despite references to *“mBART metadata extraction”*, the system is primarily **rule-based**.

### Model Usage

- **Zero-shot classification only**
  - Model: `facebook/bart-large-mnli`
  - Purpose: Assign document to a **research group**

### Rule-Based Extraction

Used for:
- Title
- Contributors
- Dates
- Description

---

## ⚙️ How Metadata Is Generated

### 1. Select Working Text

The pipeline chooses the best available text:

```python
working_text = nlp_text or cleaned_text
```

---

### 2. Extract Contributors

**Function:** `extract_contributors(row)`

- Focuses on the first lines of the document
- Removes:
  - Table of contents
  - Section headers
  - Organization names
- Detects patterns like:
  - `Author:`
  - `Authors:`
  - `Written by`
- Splits names on:
  - commas, `and`, `&`, `;`
- Validates person-like names
- Optionally includes NER `PERSON` entities (if near title page)

---

### 3. Extract Title

**Function:** `extract_title(row, contributors)`

- Reads probable title-page lines
- Merges short consecutive lines
- Stops at:
  - Dates
  - Author lines
  - Organizations
  - TOC markers
  - Section headers
- Fallback: scores and selects best candidate line

---

### 4. Extract Dates

**Function:** `extract_document_dates(working_text)`

#### Supported formats:

- `12/03/2024`
- `2024-03-12`
- `12 March 2024`
- `March 2024`
- `2024`

#### Scoring logic

**Higher score if:**
- Appears early in document
- Near keywords:
  - publication date
  - published
  - report date
  - submission date
  - final report

**Lower score if:**
- In references/bibliography
- Near DOI, URL, journal, volume, pages
- Standalone year without context

#### Outputs:

- `start_date`
- `end_date`
- `dates_found`
- `date_confidence`

---

### 5. Extract Description

```python
description = abstract_text or extract_body_preview(working_text)
```

- Uses abstract if available
- Otherwise uses first meaningful body text

---

### 6. Classify Research Group

- Model: `facebook/bart-large-mnli`
- Method: zero-shot classification

#### Input:

- Title
- Description
- Body excerpt

#### Template:

```
"This document best fits the research group {}."
```

#### Threshold:

```python
RESEARCH_GROUP_CONFIDENCE_THRESHOLD = 0.20
```

- Below threshold → no label assigned

---

## ✅ Why This Works Well

### 1. Uses Front-Matter Structure

Extracts from:
- Title
- Authors
- Dates
- Abstract

---

### 2. Filters Noise

Removes:
- Table of contents
- Section headers
- Bibliography
- Page numbers
- Institutions
- Non-title lines

---

### 3. Combines Rules + NLP

- Regex + heuristics → structure
- NER → contributor validation
- Zero-shot → classification

---

### 4. Built-in Fallbacks

- `nlp_text → cleaned_text`
- `abstract → body preview`
- `title block → scored line`
- `authors → +NER support`

---

## 📦 Output Schema

```json
{
  "id": "",
  "title": "",
  "contributors": [],
  "start_date": "",
  "end_date": "",
  "dates_found": [],
  "date_confidence": 0.0,
  "description": "",
  "research_group": "",
  "research_group_confidence": 0.0,
  "research_group_top3": []
}
```

Includes:
- `metadata_debug` (front lines, date candidates, etc.)

---

## ⚠️ Limitations

Performs worse on:
- Scanned PDFs (poor OCR)
- Complex layouts
- Slide decks
- Multi-column PDFs
- Metadata only in headers/footers/images

Also:
- Conservative → may miss edge cases rather than guess

---

# 🚧 Improvements Roadmap

## 🔴 High Priority

### 1. Improve PDF → Text Extraction

- Evaluate tools: PyMuPDF, pdfplumber, Tika
- Fix line merging issues
- Handle multi-column layouts
- Normalize encoding
- Add OCR fallback (Tesseract)

---

### 2. Improve Title Extraction

- Better multi-line support
- Handle ALL CAPS
- Detect subtitles (`:`, `-`)
- Fallback: best front-line candidate

---

### 3. Improve Contributor Extraction

- Support initials (J. Smith)
- Support non-Western names
- Allow longer names
- Add confidence scores
- Log rejected candidates

---

### 4. Improve Date Extraction

- Distinguish:
  - publication vs submission vs project dates
- Filter:
  - copyright years
  - academic years
- Handle ranges (2022–2024)
- Add primary date field

---

## 🟡 Medium Priority

### 5. Improve Description

- Clean abstract extraction
- Better sentence segmentation
- Optional summarization model
- Reduce noise

---

### 6. Improve Classification

- Improve label wording
- Try `multi_label=True`
- Tune thresholds
- Log misclassifications

---

### 7. Metadata Confidence Scoring

- Per-field confidence:
  - title
  - contributors
  - dates
- Aggregate score

---

## 🟢 Low Priority

### 8. Performance & Scaling

- Batch processing
- Progress logging
- Multiprocessing
- Cache predictions

---

### 9. Debugging & Observability

- Expand debug output:
  - title candidates
  - contributor candidates
  - rejected lines
- Save intermediate states
- Replace print with logging

---

### 10. Evaluation Framework

- Create labeled dataset
- Measure:
  - title accuracy
  - contributor precision/recall
  - date correctness
  - classification accuracy
- Add regression tests

---

### 11. Configurability

- Externalize thresholds
- Toggle features (NER, classification)
- Config-driven regex patterns

---

## 🚀 Advanced Improvements

### 12. LLM Fallback Extraction

- Use LLM only when rules fail

**Prompt:**
```
Extract title, authors, and date from this text
```

- Combine with rule-based output

---

### 13. Layout-Aware Extraction

- Use:
  - LayoutLM
  - pdfplumber layout features
- Detect:
  - font size
  - position
  - blocks

---

### 14. Language Support

- Detect language
- Adapt:
  - month names
  - author formats
  - section labels
  