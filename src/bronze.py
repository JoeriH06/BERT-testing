
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def extract_pdf_text(pdf_path: str | Path) -> str:
    """Bronze layer: extract raw text from PDF, page by page."""
    if PdfReader is None:
        raise ImportError("pypdf is required. Install with: pip install pypdf")
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            pages.append(page_text.strip())
    return "\n\n".join(pages).strip()


def save_bronze_output(document_id: str, file_path: str | Path, text: str, data_dir: str | Path = "Data") -> dict:
    data_dir = Path(data_dir)
    bronze_dir = data_dir / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    file_path = Path(file_path)

    txt_output = bronze_dir / f"{document_id}.txt"
    json_output = bronze_dir / f"{document_id}_metadata.json"

    txt_output.write_text(text or "", encoding="utf-8")
    metadata = {
        "document_id": document_id,
        "original_file": file_path.name,
        "file_type": file_path.suffix.lower(),
        "text_file": str(txt_output),
        "characters": len(text or ""),
        "processed_at": datetime.now().isoformat(),
        "processing_layer": "bronze",
    }
    json_output.write_text(json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8")
    return metadata


def run_bronze_for_file(pdf_path: str | Path, document_id: str, data_dir: str | Path = "Data") -> dict:
    text = extract_pdf_text(pdf_path)
    meta = save_bronze_output(document_id, pdf_path, text, data_dir=data_dir)
    meta["raw_text"] = text
    return meta


def run_bronze_layer(data_dir: str | Path = "Data") -> list[dict]:
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    outputs = []
    for index, file_path in enumerate(sorted(raw_dir.glob("*.pdf")), start=1):
        document_id = f"doc_{index:02d}"
        outputs.append(run_bronze_for_file(file_path, document_id, data_dir=data_dir))
    return outputs
