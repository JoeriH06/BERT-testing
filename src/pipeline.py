
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import shutil
import time

from src import bronze, silver, silver_nlp, gold, gold_meta

DATA_DIRS = ["raw", "bronze", "silver", "silver_nlp", "gold", "gold_meta"]


def ensure_data_dirs(data_dir: str | Path = "Data") -> None:
    root=Path(data_dir)
    for d in DATA_DIRS:
        (root/d).mkdir(parents=True, exist_ok=True)


def clear_data_layers(data_dir: str | Path = "Data", keep_raw_file: str | None = None) -> None:
    root=Path(data_dir)
    ensure_data_dirs(root)
    for d in DATA_DIRS:
        folder=root/d
        for item in folder.iterdir():
            if d == "raw" and keep_raw_file and item.name == keep_raw_file:
                continue
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item)


def make_document_id(pdf_path: str | Path) -> str:
    p=Path(pdf_path)
    digest=hashlib.sha1((p.name + str(p.stat().st_size) + str(time.time())).encode()).hexdigest()[:8]
    stem=re.sub(r"[^a-zA-Z0-9]+", "_", p.stem).strip("_").lower()[:28] or "upload"
    return f"doc_{stem}_{digest}"


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merge_result(document_id: str, data_dir: str | Path = "Data") -> dict:
    root=Path(data_dir)
    gold_data=load_json(root/"gold"/f"{document_id}_gold.json")
    meta=load_json(root/"gold_meta"/f"{document_id}_gold_metadata.json")
    silver_data=load_json(root/"silver"/f"{document_id}_silver.json")

    metadata = {
        "id": document_id,
        "title": meta.get("title"),
        "contributors": meta.get("authors", []),
        "publication_date": meta.get("date"),
        "language": meta.get("language"),
        "document_type": meta.get("document_type"),
        "research_or_project_topic": meta.get("research_or_project_topic"),
        "research_question_or_goal": meta.get("research_question_or_goal"),
        "description": meta.get("short_summary"),
        "keywords": meta.get("keywords", []),
        "top_terms": [t.get("term") for t in gold_data.get("top_terms", []) if isinstance(t, dict) and t.get("term")],
        "contact": meta.get("contact", {}),
        "suitable_kmp_fields": meta.get("suitable_kmp_fields", {}),
    }

    result = {
        "document_id": document_id,
        "document_summary": gold_data.get("document_summary") or meta.get("short_summary"),
        "summary": gold_data.get("document_summary") or meta.get("short_summary"),  # compatibility
        "top_terms": gold_data.get("top_terms", []),
        "suggested_entities": gold_data.get("suggested_entities", {}),
        "main_topics": gold_data.get("main_topics", []),
        "results_or_conclusions": gold_data.get("results_or_conclusions", []),
        "possible_value_for_knowledge_platform": gold_data.get("possible_value_for_knowledge_platform"),
        "metadata": metadata,
        "model": gold_data.get("@pipeline", {}).get("model") or meta.get("@pipeline", {}).get("model"),
        "language": metadata.get("language"),
        "quality": silver_data.get("quality", {}),
        "statistics": silver_data.get("statistics", {}),
        "@pipeline": {
            "created_at": datetime.now().isoformat(),
            "bronze": "completed",
            "silver": silver_data.get("processing_version"),
            "silver_nlp": (load_json(root/"silver_nlp"/f"{document_id}_silver_nlp.json")).get("processing_version"),
            "gold": gold_data.get("@pipeline", {}).get("processing_version"),
            "gold_meta": meta.get("@pipeline", {}).get("processing_version"),
        }
    }
    out=root/"gold"/f"{document_id}_result.json"
    out.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    # also update gold json for old app
    (root/"gold"/f"{document_id}_gold.json").write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    return result


def run_pipeline(
    pdf_path: str | Path,
    model: str = "qwen2.5:3b-instruct",
    data_dir: str | Path = "Data",
    require_ollama: bool = True,
    clear_previous: bool = True,
    progress_callback=None,
    gold_resources=None,
) -> dict:
    """Run bronze -> silver -> silver_nlp -> gold -> gold_meta for one PDF.

    progress_callback: optional function(step_name: str, progress: float)
    """
    root=Path(data_dir)
    ensure_data_dirs(root)
    pdf_path=Path(pdf_path)
    if clear_previous:
        clear_data_layers(root, keep_raw_file=pdf_path.name)

    document_id=make_document_id(pdf_path)

    def progress(step, value):
        if progress_callback:
            progress_callback(step, value)

    progress("Bronze: extracting PDF text", 0.08)
    bronze_out=bronze.run_bronze_for_file(pdf_path, document_id, data_dir=root)

    progress("Silver: cleaning and structuring document", 0.25)
    silver_out=silver.process_text(bronze_out["raw_text"], document_id, original_file=pdf_path.name, data_dir=root)

    progress("Silver NLP: extracting local keyword/entity suggestions", 0.45)
    silver_nlp.process_document(document_id, data_dir=root)

    progress("Gold: local Ollama document analysis", 0.68)
    effective_model=model
    if gold_resources and isinstance(gold_resources, dict) and gold_resources.get("model"):
        effective_model=gold_resources["model"]
    gold.process_document(document_id, data_dir=root, model=effective_model, require_ollama=require_ollama)

    progress("Gold Meta: local Ollama metadata extraction", 0.86)
    gold_meta.process_document(document_id, data_dir=root, model=effective_model, require_ollama=require_ollama)

    progress("Finalizing result", 0.95)
    result=merge_result(document_id, root)

    paths={
        "document_id": document_id,
        "bronze_text_path": str(root/"bronze"/f"{document_id}.txt"),
        "silver_json_path": str(root/"silver"/f"{document_id}_silver.json"),
        "silver_nlp_json_path": str(root/"silver_nlp"/f"{document_id}_silver_nlp.json"),
        "gold_json_path": str(root/"gold"/f"{document_id}_gold.json"),
        "gold_result_json_path": str(root/"gold"/f"{document_id}_result.json"),
        "gold_meta_json_path": str(root/"gold_meta"/f"{document_id}_gold_metadata.json"),
        "quality_report": silver_out.get("quality", {}),
        "statistics": silver_out.get("statistics", {}),
        "language": silver_out.get("detected_language"),
        "model": effective_model,
    }
    progress("Done", 1.0)
    return paths


def load_result(gold_json_path: str | Path) -> dict:
    return load_json(gold_json_path)
