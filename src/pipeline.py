from pathlib import Path
import json

from src.bronze import run_bronze
from src.silver import run_silver
from src.silver_nlp import run_silver_nlp
from src.gold import run_gold


def clear_previous_outputs():
    folders = [
        Path("Data/bronze"),
        Path("Data/silver"),
        Path("Data/gold/MBART"),
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        for file_path in folder.iterdir():
            if file_path.is_file():
                file_path.unlink()

    # optional: remove generated txt files but keep uploaded pdfs
    raw_folder = Path("Data/raw")
    raw_folder.mkdir(parents=True, exist_ok=True)
    for file_path in raw_folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".txt":
            file_path.unlink()


def run_pipeline(pdf_path: str, gold_resources=None) -> dict:
    clear_previous_outputs()

    bronze_result = run_bronze(pdf_path)
    silver_result = run_silver(bronze_result["bronze_json_path"])
    silver_nlp_result = run_silver_nlp(silver_result["silver_json_path"])
    gold_result = run_gold(
        silver_nlp_result["silver_nlp_json_path"],
        resources=gold_resources
    )

    with open(gold_result["gold_json_path"], "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    top_terms = gold_data.get("metadata", {}).get("top_terms", [])

    return {
        "document_id": bronze_result["document_id"],
        "pdf_path": bronze_result["pdf_path"],
        "txt_path": bronze_result["txt_path"],
        "bronze_json_path": bronze_result["bronze_json_path"],
        "silver_json_path": silver_result["silver_json_path"],
        "silver_nlp_json_path": silver_nlp_result["silver_nlp_json_path"],
        "gold_json_path": gold_result["gold_json_path"],
        "top_terms": top_terms,
    }