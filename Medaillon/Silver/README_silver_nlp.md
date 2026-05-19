# 🥈 Silver NLP Layer — Local LLM Processing

The **Silver NLP layer** enriches the structured Silver documents using local Large Language Models (LLMs).

This layer introduces semantic processing and lightweight NLP tasks while still focusing on intermediate document enrichment.

The implementation uses locally hosted models through Ollama.

---

## 🎯 Purpose

The Silver NLP layer is responsible for:

- Generating document summaries
- Extracting keywords
- Identifying important topics
- Generating semantic metadata
- Enriching document structure
- Preparing information for Gold-level outputs

---

## 🔄 Input / Output

### Input

```bash
../../Data/silver
```

### Output

```bash
../../Data/silver_nlp
```

---

## 📄 Example Output Schema

```python
{
  "document_id": "doc_01",
  "summary": "This document discusses...",
  "keywords": [
    "NLP",
    "Knowledge Management",
    "AI"
  ],
  "topics": [
    "Automation",
    "Document Processing"
  ],
  "processed_with_model": "qwen2.5:3b-instruct"
}
```

---

## ⚙️ Main Processing Steps

1. Load Silver structured documents
2. Chunk long documents
3. Send chunks to local LLM
4. Generate summaries and keywords
5. Combine chunk outputs
6. Save enriched NLP results

---

## 📦 Technologies Used

- Python
- Ollama
- Qwen2.5 Instruct
- LangChain
- JSON

---

## ✅ Why This Layer Exists

- Adds semantic understanding
- Creates AI-enriched metadata
- Improves document searchability
- Enables downstream knowledge extraction
- Supports automated analysis workflows

---

## ⚠️ Limitations

Possible issues include:

- Hallucinations from LLMs
- Slow inference on CPU-only systems
- Context window limitations
- Inconsistent summarization quality
- Large document processing overhead

---

## 🚀 Future Improvements

- GPU acceleration
- Better prompt engineering
- Vector database integration
- Hybrid NLP + rule-based extraction
- Improved chunking strategies