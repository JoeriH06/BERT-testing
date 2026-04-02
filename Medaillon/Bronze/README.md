# 🥉 Bronze Layer

The **Bronze layer** is the raw ingestion stage of the pipeline.

Its purpose is to collect source documents, extract their text, and store that text in a structured JSON format for downstream processing.

In this project, the source files are PDFs stored in the raw data folder. These are converted into plain text and then packaged into Bronze JSON records, along with extraction metadata and logs.

---

## 🎯 Purpose

The Bronze layer is responsible for:

- Reading raw PDF files
- Extracting text page by page
- Assigning consistent document IDs
- Capturing PDF metadata when available
- Logging extraction results per page
- Saving extracted text into structured JSON records
- Generating validation and manifest reports

This layer does **not** perform advanced cleaning or NLP. Its job is simply to preserve the extracted document text and basic metadata in a structured way.

---

## 🔄 Input / Output

### Input

- Raw PDF files in:

```bash
../../Data/raw
```
### Output

```bash
../../Data/bronze
```
```bash
../../Data/bronze/logs
```
```bash
../../Data/bronze/reports
```

### Output schema

```python
{
  "document_id": "doc_01",
  "source_file_name": "example.pdf",
  "source_file_path": "D:\\path\\to\\Data\\raw\\example.pdf",
  "raw_text": "Full extracted text from the PDF...",
  "pdf_metadata": {
    "Author": "Author Name",
    "Title": "Document Title"
  },
  "page_count": 40,
  "page_extraction_log": [
    {
      "page_number": 1,
      "extraction_method": "pypdf",
      "characters_extracted": 207,
      "status": "success",
      "warning": null
    }
  ],
  "extraction_status": "success",
  "used_ocr_fallback": false,
  "processed_at_utc": "2026-04-02T08:56:13.212101+00:00"
}
```

### Why This Layer Exists
- Keeps ingestion separate from cleaning
- Preserves original extracted text
- Adds traceability via metadata and logs
- Makes extraction quality inspectable
- Standardizes document IDs
- Simplifies downstream processing

### Limitations
- Depends on PDF extraction quality
Possible issues:
- Broken reading order
- Missing characters
- Multi-column errors
- Headers/footers mixed in
- Tables flattened into text

### Future Improvements
- Improve OCR accuracy
- Store relative paths instead of absolute
- Add file size + timestamps
- Add text length metrics
- Improve error handling
- Add data quality scoring