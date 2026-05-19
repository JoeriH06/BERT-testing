# 🥉 Bronze Layer

The **Bronze layer** is the raw ingestion stage of the pipeline.

Its purpose is to collect source PDF documents, extract their textual content, and preserve that information in a structured format for downstream processing.

This layer focuses entirely on ingestion and traceability. No advanced NLP or AI processing is performed at this stage.

---

## 🎯 Purpose

The Bronze layer is responsible for:

- Reading raw PDF files
- Extracting text page by page
- Assigning consistent document IDs
- Preserving original extracted content
- Saving extracted text into structured outputs
- Storing document metadata
- Creating a stable base for downstream processing

This layer acts as the foundation of the medallion architecture.

---

## 🔄 Input / Output

### Input

```bash
../../Data/raw
```

### Output

```bash
../../Data/bronze
```

---

## 📄 Example Output Schema

```python
{
  "document_id": "doc_01",
  "source_file_name": "example.pdf",
  "raw_text": "Full extracted text from the PDF...",
  "page_count": 40,
  "processed_at_utc": "2026-04-02T08:56:13.212101+00:00"
}
```

---

## ⚙️ Main Processing Steps

1. Load PDF files from the raw directory
2. Extract text using PyPDF
3. Combine page text into a single document string
4. Save extracted text and metadata
5. Generate consistent document identifiers

---

## 📦 Technologies Used

- Python
- PyPDF
- JSON
- Pathlib

---

## ✅ Why This Layer Exists

- Keeps ingestion separate from processing
- Preserves original extracted text
- Makes extraction reproducible
- Standardizes document handling
- Simplifies downstream NLP workflows

---

## ⚠️ Limitations

The extraction quality depends heavily on PDF quality.

Possible issues include:

- Missing text
- Broken reading order
- Multi-column extraction problems
- OCR limitations
- Flattened tables
- Headers and footers mixed into text

---

## 🚀 Future Improvements

- Add OCR fallback support
- Improve metadata extraction
- Add extraction quality scoring
- Store relative instead of absolute paths
- Add document validation reports
- Improve error handling