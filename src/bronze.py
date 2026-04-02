import os
import json
from pathlib import Path
from pypdf import PdfReader


RAW_FOLDER = Path("Data/raw")
BRONZE_FOLDER = Path("Data/bronze")

RAW_FOLDER.mkdir(parents=True, exist_ok=True)
BRONZE_FOLDER.mkdir(parents=True, exist_ok=True)


def get_next_doc_id(folder: str | Path) -> str:
    """
    Find the next sequential document ID based on existing .txt files.
    Example: doc_01, doc_02, doc_03
    """
    folder = Path(folder)
    existing_numbers = []

    for file_path in folder.iterdir():
        if file_path.name.startswith("doc_") and file_path.suffix.lower() == ".txt":
            try:
                num = int(file_path.stem.split("_")[1])
                existing_numbers.append(num)
            except (IndexError, ValueError):
                pass

    next_num = max(existing_numbers, default=0) + 1
    return f"doc_{next_num:02d}"


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF using pypdf.
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path), strict=False)

    text_parts = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text_parts.append(extracted)

    return "\n".join(text_parts).strip()


def save_txt(document_id: str, text: str, raw_folder: str | Path = RAW_FOLDER) -> str:
    """
    Save extracted text as a .txt file in the raw folder.
    Returns the txt file path.
    """
    raw_folder = Path(raw_folder)
    raw_folder.mkdir(parents=True, exist_ok=True)

    txt_path = raw_folder / f"{document_id}.txt"
    txt_path.write_text(text, encoding="utf-8")
    return str(txt_path)


def save_bronze_json(document_id: str, text: str, bronze_folder: str | Path = BRONZE_FOLDER) -> str:
    """
    Save bronze output as JSON.
    Returns the bronze JSON path.
    """
    bronze_folder = Path(bronze_folder)
    bronze_folder.mkdir(parents=True, exist_ok=True)

    bronze_output_path = bronze_folder / f"{document_id}_bronze.json"

    bronze_data = [
        {
            "document_id": document_id,
            "raw_text": text
        }
    ]

    with open(bronze_output_path, "w", encoding="utf-8") as f:
        json.dump(bronze_data, f, ensure_ascii=False, indent=4)

    return str(bronze_output_path)


def run_bronze(pdf_path: str | Path, raw_folder: str | Path = RAW_FOLDER, bronze_folder: str | Path = BRONZE_FOLDER) -> dict:
    """
    Process one PDF:
    1. Generate next doc ID
    2. Extract text from PDF
    3. Save TXT to raw folder
    4. Save bronze JSON to bronze folder

    Returns:
        {
            "document_id": ...,
            "pdf_path": ...,
            "txt_path": ...,
            "bronze_json_path": ...
        }
    """
    pdf_path = Path(pdf_path)
    raw_folder = Path(raw_folder)
    bronze_folder = Path(bronze_folder)

    raw_folder.mkdir(parents=True, exist_ok=True)
    bronze_folder.mkdir(parents=True, exist_ok=True)

    document_id = get_next_doc_id(raw_folder)

    text = extract_text_from_pdf(pdf_path)
    txt_path = save_txt(document_id, text, raw_folder)
    bronze_json_path = save_bronze_json(document_id, text, bronze_folder)

    return {
        "document_id": document_id,
        "pdf_path": str(pdf_path),
        "txt_path": txt_path,
        "bronze_json_path": bronze_json_path
    }


def process_all_pdfs(raw_folder: str | Path = RAW_FOLDER, bronze_folder: str | Path = BRONZE_FOLDER) -> list[dict]:
    """
    Optional batch mode:
    Process all PDFs in the raw folder that do not yet have matching TXT output.
    """
    raw_folder = Path(raw_folder)
    bronze_folder = Path(bronze_folder)

    results = []

    for file_path in raw_folder.iterdir():
        if file_path.suffix.lower() != ".pdf":
            continue

        document_id = get_next_doc_id(raw_folder)
        txt_path = raw_folder / f"{document_id}.txt"

        if txt_path.exists():
            print(f"Skipping {file_path.name}, already converted.")
            continue

        try:
            text = extract_text_from_pdf(file_path)
            saved_txt_path = save_txt(document_id, text, raw_folder)
            bronze_json_path = save_bronze_json(document_id, text, bronze_folder)

            result = {
                "document_id": document_id,
                "pdf_path": str(file_path),
                "txt_path": saved_txt_path,
                "bronze_json_path": bronze_json_path
            }
            results.append(result)

            print(f"Processed {file_path.name} -> {document_id}")

        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    return results