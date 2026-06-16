from __future__ import annotations

from datetime import datetime
from pathlib import Path
import hashlib
import inspect
import json
import re
import shutil
import time

import requests

DATA_DIRS = ["raw", "bronze", "silver", "silver_nlp", "gold", "gold_meta"]

NOTEBOOKS = {
    "bronze": Path("Medaillon/Bronze/bronze.ipynb"),
    "silver": Path("Medaillon/Silver/silver_local_llm.ipynb"),
    "silver_nlp": Path("Medaillon/Silver/silver_nlp_local_llm.ipynb"),
    "gold": Path("Medaillon/Gold/gold_local_llm.ipynb"),
    "gold_meta": Path("Medaillon/Gold/gold_meta_local_llm.ipynb"),
}


def ensure_data_dirs(data_dir: str | Path = "Data") -> None:
    root = Path(data_dir)
    for folder in DATA_DIRS:
        (root / folder).mkdir(parents=True, exist_ok=True)


def clear_data_layers(data_dir: str | Path = "Data", keep_raw_file: str | None = None) -> None:
    root = Path(data_dir)
    ensure_data_dirs(root)
    for folder_name in DATA_DIRS:
        folder = root / folder_name
        for item in folder.iterdir():
            if folder_name == "raw" and keep_raw_file and item.name == keep_raw_file:
                continue
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item)


def make_document_id(pdf_path: str | Path) -> str:
    path = Path(pdf_path)
    digest = hashlib.sha1((path.name + str(path.stat().st_size) + str(time.time())).encode()).hexdigest()[:8]
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower()[:28] or "upload"
    return f"doc_{stem}_{digest}"


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_result(gold_json_path: str | Path) -> dict:
    return load_json(gold_json_path)


def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False


def _is_notebook_run_cell(source: str) -> bool:
    stripped = source.lstrip()
    if stripped.startswith("# Run"):
        return True
    return stripped in {"run_bronze_layer()"} or stripped.startswith("run_bronze_layer()")


def _rewrite_notebook_paths(source: str, data_dir: Path) -> str:
    data_literal = repr(str(data_dir))
    replacements = {
        'Path("../../Data/raw")': f'Path({data_literal}) / "raw"',
        'Path("../../Data/bronze")': f'Path({data_literal}) / "bronze"',
        'Path("../../Data/silver")': f'Path({data_literal}) / "silver"',
        'Path("../../Data/silver_nlp")': f'Path({data_literal}) / "silver_nlp"',
        'Path("../../Data/gold")': f'Path({data_literal}) / "gold"',
        'Path("../../Data/gold_meta")': f'Path({data_literal}) / "gold_meta"',
        'Path("../../Data")': f'Path({data_literal})',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def load_notebook_namespace(notebook_path: str | Path, data_dir: str | Path) -> dict:
    """Load notebook code cells as callable Python without running the final notebook run cell."""
    notebook_path = Path(notebook_path)
    root = Path(data_dir)
    nb = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    namespace = {"__name__": f"_notebook_{notebook_path.stem}"}
    code_cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if _is_notebook_run_cell(source):
            continue
        code_cells.append(_rewrite_notebook_paths(source, root))
    exec(compile("\n\n".join(code_cells), str(notebook_path), "exec"), namespace)

    namespace["BASE_DIR"] = root
    namespace["RAW_DIR"] = root / "raw"
    namespace["RAW_FOLDER"] = root / "raw"
    namespace["BRONZE_DIR"] = root / "bronze"
    namespace["BRONZE_FOLDER"] = root / "bronze"
    namespace["SILVER_DIR"] = root / "silver"
    namespace["SILVER_NLP_DIR"] = root / "silver_nlp"
    namespace["GOLD_DIR"] = root / "gold"
    namespace["GOLD_META_DIR"] = root / "gold_meta"
    namespace["META_DIR"] = root / "gold_meta"
    ensure_data_dirs(root)
    return namespace


def call_notebook_func(namespace: dict, name: str, *args, **kwargs):
    func = namespace[name]
    accepted = inspect.signature(func).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return func(*args, **filtered_kwargs)


def run_notebook_bronze(pdf_path: Path, document_id: str, data_dir: Path) -> dict:
    namespace = load_notebook_namespace(NOTEBOOKS["bronze"], data_dir)
    text = namespace["extract_pdf_text"](pdf_path)
    namespace["save_bronze_output"](document_id, pdf_path, text)
    return {"document_id": document_id, "raw_text": text}


def run_notebook_layer(layer: str, function_name: str, document_id: str, data_dir: Path, **kwargs):
    namespace = load_notebook_namespace(NOTEBOOKS[layer], data_dir)
    return call_notebook_func(namespace, function_name, document_id, data_dir=data_dir, **kwargs)


def merge_result(document_id: str, data_dir: str | Path = "Data") -> dict:
    root = Path(data_dir)
    gold_data = load_json(root / "gold" / f"{document_id}_gold.json")
    meta = load_json(root / "gold_meta" / f"{document_id}_gold_metadata.json")
    silver_data = load_json(root / "silver" / f"{document_id}_silver.json")
    silver_nlp_data = load_json(root / "silver_nlp" / f"{document_id}_silver_nlp.json")

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
        "keyword_suggestions": meta.get("keyword_suggestions", []),
        "keyword_target_count": meta.get("keyword_target_count"),
        "keyword_word_count": meta.get("keyword_word_count"),
        "field_suggestions": meta.get("field_suggestions", {}),
        "kmp_context": meta.get("kmp_context", {}),
        "top_terms": [item.get("term") for item in gold_data.get("top_terms", []) if isinstance(item, dict) and item.get("term")],
        "contact": meta.get("contact", {}),
        "suitable_kmp_fields": meta.get("suitable_kmp_fields", {}),
    }

    result = {
        "document_id": document_id,
        "document_summary": meta.get("short_summary") or gold_data.get("document_summary"),
        "summary": meta.get("short_summary") or gold_data.get("document_summary"),
        "top_terms": gold_data.get("top_terms", []),
        "suggested_entities": gold_data.get("suggested_entities", silver_nlp_data.get("entities", {})),
        "main_topics": gold_data.get("main_topics", []),
        "results_or_conclusions": gold_data.get("results_or_conclusions", []),
        "possible_value_for_knowledge_platform": gold_data.get("possible_value_for_knowledge_platform"),
        "metadata": metadata,
        "model": gold_data.get("@pipeline", {}).get("model") or meta.get("@pipeline", {}).get("model"),
        "evaluation": meta.get("evaluation") or gold_data.get("evaluation", {}),
        "quality": {
            **silver_data.get("quality", {}),
            **gold_data.get("quality", {}),
            **meta.get("quality", {}),
        },
        "statistics": {
            **silver_data.get("statistics", {}),
            **gold_data.get("statistics", {}),
            **meta.get("statistics", {}),
        },
        "@pipeline": {
            "created_at": datetime.now().isoformat(),
            "runtime_source": "Medaillon notebooks",
            "bronze": "completed",
            "silver": silver_data.get("@pipeline", {}).get("processing_version") or silver_data.get("processing_version"),
            "silver_nlp": silver_nlp_data.get("@pipeline", {}).get("processing_version") or silver_nlp_data.get("processing_version"),
            "gold": gold_data.get("@pipeline", {}).get("processing_version"),
            "gold_meta": meta.get("@pipeline", {}).get("processing_version"),
        },
    }
    out = root / "gold" / f"{document_id}_result.json"
    out.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
    (root / "gold" / f"{document_id}_gold.json").write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
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
    """Run Bronze -> Silver -> Silver NLP -> Gold -> Gold Meta from the Medaillon notebooks."""
    root = Path(data_dir)
    ensure_data_dirs(root)
    pdf_path = Path(pdf_path)
    if clear_previous:
        clear_data_layers(root, keep_raw_file=pdf_path.name)

    document_id = make_document_id(pdf_path)

    def progress(step: str, value: float) -> None:
        if progress_callback:
            progress_callback(step, value)

    progress("Bronze notebook: extracting PDF text", 0.08)
    run_notebook_bronze(pdf_path, document_id, root)

    progress("Silver notebook: cleaning and structuring document", 0.25)
    silver_path = run_notebook_layer("silver", "process_silver_document", document_id, root)
    silver_out = load_json(silver_path)

    progress("Silver NLP notebook: extracting local keyword/entity suggestions", 0.45)
    run_notebook_layer("silver_nlp", "process_silver_nlp_document", document_id, root)

    effective_model = model
    if gold_resources and isinstance(gold_resources, dict) and gold_resources.get("model"):
        effective_model = gold_resources["model"]

    progress("Gold notebook: local Ollama document analysis", 0.68)
    run_notebook_layer("gold", "process_gold_document", document_id, root, model=effective_model, require_ollama=require_ollama)

    progress("Gold Meta notebook: metadata extraction", 0.86)
    run_notebook_layer("gold_meta", "process_metadata", document_id, root, model=effective_model, require_ollama=require_ollama)

    progress("Finalizing result", 0.95)
    merge_result(document_id, root)

    paths = {
        "document_id": document_id,
        "bronze_text_path": str(root / "bronze" / f"{document_id}.txt"),
        "silver_json_path": str(root / "silver" / f"{document_id}_silver.json"),
        "silver_nlp_json_path": str(root / "silver_nlp" / f"{document_id}_silver_nlp.json"),
        "gold_json_path": str(root / "gold" / f"{document_id}_gold.json"),
        "gold_result_json_path": str(root / "gold" / f"{document_id}_result.json"),
        "gold_meta_json_path": str(root / "gold_meta" / f"{document_id}_gold_metadata.json"),
        "quality_report": silver_out.get("quality", {}),
        "statistics": silver_out.get("statistics", {}),
        "language": silver_out.get("detected_language"),
        "model": effective_model,
        "runtime_source": "Medaillon notebooks",
    }
    progress("Done", 1.0)
    return paths
