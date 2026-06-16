# 🥇 Gold Metadata Layer

The **Gold Metadata layer** converts enriched NLP output into structured, KMP-ready metadata.

This layer is designed to be generic for Dutch and English PDF documents and avoids document-specific hardcoded extraction rules.

---

## 🎯 Purpose

The Gold Metadata layer is responsible for:

- extracting structured metadata from document evidence;
- extracting contributors/authors where evidence is available;
- generating 15–20 keywords where enough evidence exists;
- creating KMP-compatible metadata fields;
- preventing hallucinated metadata by validating outputs against local evidence;
- preparing outputs for indexing, review, and future KMP integration.

---

## 🔄 Input / Output

### Input

```bash
../../Data/silver
../../Data/silver_nlp
../../Data/gold
```

### Output

```bash
../../Data/gold_meta
```

---

## 📄 Main Output Schema

```json
{
  "title": "Automating Knowledge Extraction",
  "authors": ["Example Person"],
  "date": "2026-06-14",
  "document_type": "portfolio",
  "language": "en",
  "research_or_project_topic": "NLP-based metadata extraction",
  "research_question_or_goal": null,
  "short_summary": "Short summary of the document.",
  "keywords": [
    "knowledge management",
    "metadata extraction",
    "NLP pipeline"
  ],
  "tools_technologies_or_models": [
    "Qwen",
    "Ollama",
    "Streamlit"
  ],
  "main_outputs_or_results": [],
  "suitable_kmp_fields": {
    "title": "Automating Knowledge Extraction",
    "description": "Short summary of the document.",
    "keywords": [],
    "contributors": ["Example Person"],
    "date": "2026-06-14",
    "language": "en",
    "document_type": "portfolio"
  },
  "contact": {
    "name": "Example Person",
    "email": null,
    "phone": null
  },
  "confidence_notes": []
}
```

---

## ⚙️ Main Processing Steps

1. Load Silver, Silver NLP, and Gold outputs.
2. Build a compact evidence package from title page, summary, top terms, and text excerpts.
3. Extract local candidates for title, contributors, dates, contact details, document type, and keywords.
4. Ask the local Ollama model to produce structured metadata.
5. Validate model output against available evidence.
6. Merge local contributor candidates with supported model output.
7. Save the final Gold Metadata JSON.

---

## ✅ Notebook Quality Rules

The notebook has been structured based on project feedback:

- one function per code cell;
- every function contains a docstring;
- contributor extraction is generic and evidence-based;
- organizations are not stored as contributors;
- unsupported or uncertain fields are set to `null` or `[]`;
- keywords are expanded to 15–20 terms when enough evidence exists;
- the output is suitable for human review before KMP upload.

---

## 📦 Technologies Used

- Python
- JSON
- Regular expressions
- Local Ollama models
- Qwen model family
- Evidence-based metadata validation

---

## ⚠️ Limitations

Possible limitations include:

- contributor extraction depends on how clearly the document lists authors;
- scanned PDFs require OCR before this layer can work reliably;
- LLM output can vary, therefore validation and human review remain necessary;
- complex layouts may hide metadata in tables or images.

---

## 🚀 Future Improvements

- add confidence scores per metadata field;
- connect metadata directly to KMP upload fields;
- improve contributor detection using NER and layout-aware parsing;
- add ontology-based topic classification;
- include human-in-the-loop approval in the Streamlit interface.
