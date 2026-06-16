# 🥇 Gold Layer — Final Knowledge Outputs

The **Gold layer** creates the final semantic document intelligence used by the Proof of Concept for the Kennis Management Platform (KMP).

This notebook is designed to run after the Bronze, Silver, and Silver NLP layers.

---

## 🎯 Purpose

The Gold layer is responsible for:

- generating a final document summary;
- generating **15–20 useful top terms** for search and indexing;
- extracting suggested entities such as people, organizations, dates, projects, tools, and models;
- calculating lightweight evaluation metrics;
- creating structured JSON outputs that can be used by the Streamlit interface and future KMP integration.

---

## 🔄 Input / Output

### Input

```bash
../../Data/silver
../../Data/silver_nlp
```

### Output

```bash
../../Data/gold
```

---

## 📄 Main Output Schema

```json
{
  "document_id": "doc_01",
  "document_summary": "Short factual document summary.",
  "top_terms": [
    {
      "rank": 1,
      "term": "knowledge management",
      "context": "Why the term matters in the document.",
      "evidence": ["Short phrase from the source text."]
    }
  ],
  "suggested_entities": {
    "people": [],
    "organizations": [],
    "locations": [],
    "dates": [],
    "projects": [],
    "models_or_tools": []
  },
  "main_topics": [],
  "results_or_conclusions": [],
  "possible_value_for_knowledge_platform": "...",
  "evaluation": {
    "runtime_seconds": 37.67,
    "original_token_count": 990,
    "summary_token_count": 75,
    "compression_ratio": 0.0753,
    "top_term_coverage": 0.18,
    "top_term_count": 20
  }
}
```

---

## ⚙️ Main Processing Steps

1. Load Silver and Silver NLP outputs.
2. Select representative document text from the beginning, middle, and end of the document.
3. Generate summary, top terms, entities, and KMP value using a local Ollama model.
4. Use generic fallback extraction when Ollama is unavailable.
5. Normalize the top terms so the output contains **15–20 terms**.
6. Calculate lightweight evaluation metrics.
7. Save the final Gold JSON output.

---

## ✅ Notebook Quality Rules

The notebook has been structured based on project feedback:

- one function per code cell;
- every function contains a docstring;
- the output is generic and not hardcoded for one document;
- uncertain values are left empty instead of guessed;
- top terms are generated in a range of 15–20;
- local deployment through Ollama is supported.

---

## 📦 Technologies Used

- Python
- JSON
- Regular expressions
- Local Ollama models
- Qwen model family
- Lightweight NLP fallback logic

---

## ⚠️ Limitations

Possible limitations include:

- LLM variability between runs;
- limited accuracy when the Silver layer contains poor text extraction;
- dependency on local hardware performance;
- no manually validated benchmark summaries.

---

## 🚀 Future Improvements

- integrate direct KMP API export;
- add confidence scoring for generated terms;
- improve entity classification;
- add vector search integration;
- evaluate larger local Qwen models on HZ infrastructure.
