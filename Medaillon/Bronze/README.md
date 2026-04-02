# 🥉 Bronze Layer

The **Bronze layer** is the raw ingestion stage of the pipeline.

Its purpose is to collect source documents, extract their text, and store that text in a simple structured JSON format for downstream processing.

In this project, the source files are PDFs stored in the raw data folder. These are converted into plain text and then packaged into Bronze JSON records.

---

## 🎯 Purpose

The Bronze layer is responsible for:

- Reading raw PDF files
- Extracting text page by page
- Assigning consistent document IDs
- Saving extracted text into JSON records

This layer does **not** perform advanced cleaning or NLP. Its job is simply to preserve the extracted document text in a structured way.

---

## 🔄 Input / Output

### Input

- Raw PDF files in:

```bash
../../Data/raw
```

### Output

- Bronze JSON files in:

```bash
../../Data/bronze
```

Each output file contains:

- `document_id`
- `raw_text`

---

## 📦 Output Schema

```json
{
  "document_id": "doc_01",
  "raw_text": "Full extracted text from the PDF..."
}
```

---

## ⚙️ Processing Steps

### 1. Reset Existing Files

Before processing begins, the pipeline clears previously generated intermediate files:

- Deletes old `doc_*.txt` files from the raw folder
- Deletes and recreates the Bronze output folder

This ensures a clean run each time.

#### Functions

```python
def clean_raw_txt_files(folder):
    for f in os.listdir(folder):
        if f.startswith("doc_") and f.endswith(".txt"):
            os.remove(os.path.join(folder, f))
```

```python
def clean_bronze_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
```

---

### 2. Generate Document IDs

Each processed document receives a sequential ID:

```python
def get_next_doc_id(counter):
    return f"doc_{counter:02d}"
```

Examples:
- `doc_01`
- `doc_02`
- `doc_03`

This makes downstream file naming consistent across the pipeline.

---

### 3. Convert PDFs to TXT

Each PDF in the raw folder is processed with `pypdf.PdfReader`.

#### Logic

- Open PDF
- Iterate through all pages
- Extract text from each page
- Concatenate extracted text
- Save as `.txt`

#### Core code

```python
reader = PdfReader(pdf_path, strict=False)
text = ""

for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted + "\n"
```

The resulting text file is saved as:

```bash
../../Data/raw/doc_XX.txt
```

---

### 4. Convert TXT to Bronze JSON

Each generated text file is wrapped into a JSON record using pandas.

#### Schema per file

```python
bronze_data = pd.DataFrame([{
    "document_id": document_id,
    "raw_text": text
}])
```

Each file is saved as:

```bash
../../Data/bronze/doc_XX_bronze.json
```

---

## 🧰 Libraries Used

```python
import os
import shutil
import pandas as pd
from pypdf import PdfReader
```

### Purpose of each library

- `os` → file and folder handling
- `shutil` → deleting and recreating folders
- `pandas` → writing structured JSON
- `pypdf` → extracting text from PDFs

---

## 📝 Full Workflow Summary

```text
PDF files
  ↓
Text extraction with pypdf
  ↓
Intermediate .txt files
  ↓
Bronze JSON records
```

---

## ✅ Why This Layer Exists

The Bronze layer creates a stable handoff between raw files and downstream processing.

Benefits:

- Keeps ingestion separate from cleaning
- Makes extracted text easy to inspect
- Standardizes document IDs
- Simplifies later Silver/NLP/metadata steps

---

## ⚠️ Limitations

This layer depends entirely on the quality of PDF text extraction.

Potential issues include:

- Scanned PDFs with no embedded text
- Broken reading order
- Missing characters
- Multi-column extraction errors
- Headers/footers mixed into body text

Because of this, Bronze output should be treated as **raw extracted text**, not final clean content.

---

## 🔧 Future Improvements

Possible improvements for this layer:

- Add OCR fallback for scanned PDFs
- Extract PDF metadata when available
- Preserve original file name in JSON
- Add page-level extraction logs
- Detect failed or empty text extraction
- Track source file path
- Add batch validation reports

---

## 📁 Example Folder Structure

```text
Data/
├── raw/
│   ├── report_1.pdf
│   ├── report_2.pdf
│   ├── doc_01.txt
│   └── doc_02.txt
└── bronze/
    ├── doc_01_bronze.json
    └── doc_02_bronze.json
```

---

## 🚀 Next Step

The Bronze output is used as input for the **Silver layer**, where the raw extracted text is cleaned, normalized, and prepared for NLP and metadata extraction.