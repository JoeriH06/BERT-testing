from pathlib import Path
import json
from datetime import datetime
from pypdf import PdfReader


RAW_FOLDER = Path("Data/raw")
BRONZE_FOLDER = Path("Data/bronze")

BRONZE_FOLDER.mkdir(parents=True, exist_ok=True)


def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"

    return text.strip()

def save_bronze_output(document_id, file_path, text):
    txt_output = BRONZE_FOLDER / f"{document_id}.txt"
    json_output = BRONZE_FOLDER / f"{document_id}_metadata.json"

    txt_output.write_text(text, encoding="utf-8")

    metadata = {
        "document_id": document_id,
        "original_file": file_path.name,
        "file_type": file_path.suffix.lower(),
        "text_file": str(txt_output),
        "characters": len(text),
        "processed_at": datetime.now().isoformat()
    }

    json_output.write_text(json.dumps(metadata, indent=4), encoding="utf-8")



def run_bronze_layer():
    files = list(RAW_FOLDER.glob("*.pdf"))

    for index, file_path in enumerate(files, start=1):
        document_id = f"doc_{index:02d}"

        print(f"Processing {file_path.name}...")

        text = extract_pdf_text(file_path)

        save_bronze_output(document_id, file_path, text)

        print(f"Saved {document_id} to bronze layer")
