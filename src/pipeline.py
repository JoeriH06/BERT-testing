from pathlib import Path
import json
import re
import time

from src import bronze, silver, silver_nlp, gold_meta, gold

DATA_DIRS = [
    Path("Data/raw"),
    Path("Data/bronze"),
    Path("Data/silver"),
    Path("Data/silver_nlp"),
    Path("Data/gold_meta"),
    Path("Data/gold"),
]

def ensure_data_dirs():
    for folder in DATA_DIRS:
        folder.mkdir(parents=True, exist_ok=True)

def make_document_id(pdf_path):
    stem = Path(pdf_path).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:32] or "upload"
    return f"doc_{stem}_{int(time.time())}"

def run_pipeline(pdf_path, gold_resources=None):
    """Run bronze -> silver -> silver_nlp -> gold_meta -> gold for one PDF."""
    ensure_data_dirs()
    pdf_path = Path(pdf_path)
    document_id = make_document_id(pdf_path)

    # Bronze
    text = bronze.extract_pdf_text(pdf_path)
    bronze.save_bronze_output(document_id, pdf_path, text)

    # Silver
    cleaned_text = silver.clean_text(text, mode="modeling")
    metadata_text = silver.clean_text(text, mode="metadata")
    chunks = silver.split_into_semantic_chunks(cleaned_text, max_words=900, min_words=250)
    chunk_records = silver.build_chunk_records(chunks)
    quality_report = silver.build_quality_report(cleaned_text, chunks)
    language = silver.detect_language(cleaned_text)
    silver.save_silver_output(
        document_id=document_id,
        cleaned_text=cleaned_text,
        metadata_text=metadata_text,
        chunk_records=chunk_records,
        quality_report=quality_report,
        language=language,
    )

    # Silver NLP
    nlp_paths = silver_nlp.run_silver_nlp_layer(document_ids=[document_id])

    # Gold metadata
    metadata = gold_meta.extract_metadata(document_id, metadata_text)
    gold_meta.save_metadata(document_id, metadata)
    meta_json_path = Path("Data/gold_meta") / f"{document_id}_meta.json"

    # Gold summary
    gold_paths = gold.run_gold_layer(
        document_ids=[document_id],
        gold_resources=gold_resources,
    )
    gold_json_path = Path(gold_paths[0]) if gold_paths else Path("Data/gold") / f"{document_id}_gold.json"

    # Merge metadata into gold output so app.py can display a single result.
    result = json.loads(gold_json_path.read_text(encoding="utf-8"))
    dates = metadata.get("dates", {}) or {}
    top_terms = []
    nlp_json_path = Path("Data/silver_nlp") / f"{document_id}_nlp.json"

    if nlp_json_path.exists():
        nlp_data = json.loads(nlp_json_path.read_text(encoding="utf-8"))

        raw_keywords = nlp_data.get("keywords", [])

        for item in raw_keywords[:12]:
            if isinstance(item, dict):
                keyword = item.get("keyword") or item.get("text") or item.get("term")
            else:
                keyword = str(item)

            keyword = keyword.strip()

            if keyword:
                top_terms.append(keyword)
    result["metadata"] = {
        "id": document_id,
        "title": metadata.get("title"),
        "subtitle": metadata.get("subtitle"),
        "description": metadata.get("description"),
        "contributors": metadata.get("contributors", []),
        "contributors_structured": metadata.get("contributors_structured", {}),
        "contributors_confidence": metadata.get("contributors_confidence"),
        "contact_person": metadata.get("contact_person"),
        "publication_date": dates.get("publication_date"),
        "start_date": dates.get("start_date"),
        "end_date": dates.get("end_date"),
        "dates_found": dates.get("dates_found", []),
        "top_terms": top_terms,
    }
    model_value = result.get("model", "Unknown model")

    if isinstance(model_value, dict):
        result["model"] = model_value.get("name", "Unknown model")
    else:
        result["model"] = str(model_value)
    gold_json_path.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")

    return {
        "document_id": document_id,
        "bronze_text_path": str(Path("Data/bronze") / f"{document_id}.txt"),
        "silver_json_path": str(Path("Data/silver") / f"{document_id}_silver.json"),
        "silver_nlp_json_path": str(nlp_json_path),
        "gold_meta_json_path": str(meta_json_path),
        "gold_json_path": str(gold_json_path),
        "quality_report": quality_report,
        "language": language,
    }
