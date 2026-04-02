# 🥇 Gold Layer — Metadata Extraction Pipeline
The **Gold layer** generates structured metadata from documents after they have passed through the Bronze, Silver, and Silver NLP layers. It combines **rule-based extraction**, **NLP signals**, and **zero-shot classification** to produce consistent, traceable metadata across documents.
---

## 🔄 Pipeline Flow
PDF → Bronze → Silver → Silver NLP → Gold → Classification → JSON Output
---

## 🎯 Purpose
The Gold layer is responsible for:
- Extracting title, contributors, dates, description
- Classifying documents into research groups
- Preserving upstream metadata
- Generating per-document JSON + batch reports
---

## 📥 Input / Output
### Input
../../../Data/silver/doc_XX_silver_nlp.json
Contains:
- document_id
- cleaned_text
- nlp_text
- entities
- metadata

### Output
../../../Data/gold/metadata/doc_XX_gold_metadata.json
Reports:
../../../Data/gold/metadata/reports/gold_metadata_manifest.json
../../../Data/gold/metadata/reports/gold_metadata_validation_report.json
---

## 🧠 What This Pipeline Uses
### Rule-Based Extraction
- Title
- Contributors
- Dates
- Description

### NLP Signals
- spaCy entities (PERSON)

### Model Usage
- facebook/bart-large-mnli
- Zero-shot classification for research group
---

## ⚙️ Metadata Generation
### 1. Working Text
working_text = nlp_text or cleaned_text

### 2. Contributors
- Uses title-page lines
- Removes TOC, sections, institutions
- Cleans suffixes (e.g. Uitgave: #1)
- Splits on commas, and, &, ;
- Validates names
- Uses NER fallback

### 3. Title
- Uses front lines
- Stops at TOC (Inhoudsopgave), authors, dates
- Merges short lines
- Fallback scoring

### 4. Dates
Supported:
- 12/03/2024
- 2024-03-12
- 12 March 2024
- March 2024
- 2024
Features:
- Front-matter only
- Context scoring
- Dutch + English support
Outputs:
- start_date
- end_date
- dates_found
- date_confidence

### 5. Description
description = abstract_text or extract_body_preview(working_text)

### 6. Research Group Classification
Model: facebook/bart-large-mnli
Template:
Deze publicatie past het best binnen onderzoeksgroep {}.
Threshold:
RESEARCH_GROUP_CONFIDENCE_THRESHOLD = 0.20
---

## 📦 Output Schema
{
  "id": "doc_01",
  "document_id": "doc_01",
  "source_file_name": "...",
  "source_file_path": "...",
  "pdf_metadata": {},
  "page_count": 40,
  "title": "...",
  "title_found": true,
  "contributors": [],
  "contributors_found": true,
  "start_date": "",
  "end_date": "",
  "dates_found": [],
  "date_confidence": 0.0,
  "description": "",
  "research_group": "",
  "research_group_confidence": 0.0,
  "research_group_top3": [],
  "metadata_quality_flags": [],
  "metadata_debug": {}
}
---

## 📊 Batch Processing
- Processes all Silver NLP files
- One output per document
- Generates:
  - manifest
  - validation report
---

## 🧪 Quality Flags
- missing_title
- missing_contributors
- missing_date
- missing_description
- missing_research_group
---

## ✅ Strengths
- Front-matter focused extraction
- Strong noise filtering (TOC, references, headers)
- Hybrid approach (rules + NLP + classification)
- Full metadata traceability
- Batch-first architecture
---

## ⚠️ Limitations
- Depends on PDF extraction quality
- Sensitive to layout issues
- Heuristic-based contributor extraction
- English model on Dutch text
---

## 🚧 Improvements
### High Priority
- Improve contributor extraction
- Improve Dutch handling
- Improve classification accuracy
- Add confidence scoring

### Medium
- Language detection
- Dutch spaCy model
- Better descriptions

### Low
- Logging
- Parallel processing
- Config-driven rules
---

## 🚀 Advanced
### LLM fallback
Extract title, authors, and date from this text

### Layout-aware extraction
- pdfplumber
- LayoutLM

### Multilingual support
- Detect language
- Adapt rules