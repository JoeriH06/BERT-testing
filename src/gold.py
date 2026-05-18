
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import re
import time
import requests

LOCAL_MODEL = "qwen2.5:3b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_EVIDENCE_CHARS = 18000
MAX_SELECTED_CHUNKS = 12
REQUEST_TIMEOUT_SECONDS = 900
TEMPERATURE = 0.1
NUM_CTX = 8192

STOPWORDS = {
    "de","het","een","en","of","in","op","te","van","voor","met","dat","dit","die","als","aan","om","er",
    "is","zijn","wordt","worden","was","waren","heeft","hebben","had","door","bij","uit","naar","ook",
    "maar","dan","dus","kan","kunnen","zal","zullen","moet","moeten","waar","wat","hoe","welke","wie",
    "wanneer","tijdens","alleen","nieuwe","gemaakt","gebruikt","maken","gebruik","bevat","basis",
    "manier","twee","tijd","team","onderzoek","ontwikkelen","ontwikkeld","plaatsen","kaart",
    "the","a","an","and","or","in","on","to","of","for","with","that","this","these","those","as","by",
    "from","at","it","its","be","is","are","was","were","has","have","had","can","could","should","would",
    "will","may","might","what","how","which","who","when","during","only","new","made","used","use",
    "using","make","contains","basis","way","two","time","team","research","develop","developed"
}
BAD_MARKERS = {"figure_caption", "table_start", "table_end", "appendix", "bijlage", "references", "bibliografie"}


def load_gold_models(*args, **kwargs):
    """Backward compatible with older app.py. Ollama models are served, not preloaded."""
    model = kwargs.get("model") or kwargs.get("local_model") or LOCAL_MODEL
    return {"model": model, "backend": "ollama"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists(): return []
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=4), encoding="utf-8")


def clean_value(v) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def is_bad_term(term: str) -> bool:
    low=clean_value(term).lower()
    if not low or low in STOPWORDS: return True
    if any(m in low for m in BAD_MARKERS): return True
    if re.fullmatch(r"\d+(?:[.,]\d+)?", low): return True
    return False


def clean_term(term: str) -> str | None:
    term=clean_value(term)
    if is_bad_term(term): return None
    if len(term.split()) > 7: return None
    return term


def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    try:
        r=requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False


def ollama_generate(prompt: str, model: str = LOCAL_MODEL, format_json: bool = True, timeout: int = REQUEST_TIMEOUT_SECONDS) -> str:
    payload={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }
    if format_json:
        payload["format"]="json"
    r=requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response","")


def parse_json_safely(text: str) -> dict:
    text=str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m=re.search(r"\{.*\}", text, flags=re.S)
        if m: return json.loads(m.group(0))
    raise ValueError("Could not parse JSON from local LLM output")


def get_top_silver_terms(silver_nlp: dict, limit: int = 25) -> list[str]:
    terms=[]
    for item in silver_nlp.get("keyword_suggestions", [])[:limit*2]:
        term=item.get("term") if isinstance(item, dict) else str(item)
        term=clean_term(term)
        if term and term.lower() not in [t.lower() for t in terms]:
            terms.append(term)
        if len(terms)>=limit: break
    return terms


def chunk_score(chunk: dict, terms: list[str]) -> float:
    text=chunk.get("text","")
    low=text.lower()
    score=0.0
    score += min(50, len(text.split())/30)
    for i,term in enumerate(terms[:25]):
        if term.lower() in low:
            score += max(1, 10-i*0.25)
    if re.search(r"\b(summary|samenvatting|conclusion|conclusie|results|resultaten|research question|onderzoeksvraag|main question|hoofdvraag)\b", low):
        score += 15
    return score


def select_evidence_chunks(chunks: list[dict], terms: list[str], max_chunks: int = MAX_SELECTED_CHUNKS) -> list[dict]:
    if not chunks: return []
    scored=[(chunk_score(ch, terms), idx, ch) for idx,ch in enumerate(chunks)]
    scored.sort(reverse=True, key=lambda x: x[0])
    selected=sorted(scored[:max_chunks], key=lambda x:x[1])
    return [ch for _,_,ch in selected]


def build_evidence(silver: dict, silver_nlp: dict) -> tuple[str, list[str], list[dict]]:
    chunks=silver.get("chunks", [])
    if not chunks:
        chunks=read_jsonl(Path("nonexistent"))  # harmless
    terms=get_top_silver_terms(silver_nlp)
    selected=select_evidence_chunks(chunks, terms)
    parts={
        "detected_language": silver.get("detected_language") or silver.get("language"),
        "titlepage_candidates": silver.get("titlepage_candidates", {}),
        "document_statistics": silver.get("statistics", {}),
        "silver_keyword_suggestions": terms[:25],
        "silver_entity_suggestions": silver_nlp.get("entity_suggestions", {}),
        "selected_document_chunks": [
            {
                "chunk_id": c.get("chunk_id"),
                "source_section": c.get("source_section_heading"),
                "text": clean_value(c.get("text"))[:2500]
            } for c in selected
        ]
    }
    evidence=json.dumps(parts, ensure_ascii=False, indent=2)
    if len(evidence) > MAX_EVIDENCE_CHARS:
        evidence=evidence[:MAX_EVIDENCE_CHARS]
    return evidence, terms, selected


def fallback_gold(silver: dict, silver_nlp: dict) -> dict:
    main=silver.get("document_parts",{}).get("main_text","")
    sentences=re.split(r"(?<=[.!?])\s+", clean_value(main))
    summary=" ".join([s for s in sentences if len(s.split())>12][:4])[:1200]
    terms=get_top_silver_terms(silver_nlp, 15)
    top=[{"rank":i+1,"term":t,"context":"","evidence":"Silver NLP keyword suggestion"} for i,t in enumerate(terms[:15])]
    return {
        "document_summary": summary,
        "top_terms": top,
        "suggested_entities": silver_nlp.get("entity_suggestions", {}),
        "main_topics": terms[:5],
        "results_or_conclusions": [],
        "possible_value_for_knowledge_platform": summary[:500],
        "confidence_notes": ["Fallback extractive Gold output used because Ollama was not reachable or failed."],
    }


def process_document(document_id: str, data_dir: str | Path = "Data", model: str = LOCAL_MODEL, require_ollama: bool = True) -> dict:
    data_dir=Path(data_dir)
    silver_dir=data_dir/"silver"; nlp_dir=data_dir/"silver_nlp"; gold_dir=data_dir/"gold"; gold_dir.mkdir(parents=True, exist_ok=True)
    silver=read_json(silver_dir/f"{document_id}_silver.json")
    silver_nlp_path=nlp_dir/f"{document_id}_silver_nlp.json"
    if not silver_nlp_path.exists(): silver_nlp_path=nlp_dir/f"{document_id}_nlp.json"
    silver_nlp=read_json(silver_nlp_path)
    detected_language=silver.get("detected_language") or silver.get("language") or "unknown"
    evidence, terms, selected = build_evidence(silver, silver_nlp)

    schema={
        "document_summary": "",
        "top_terms": [{"rank": 1, "term": "", "context": "", "evidence": ""}],
        "suggested_entities": {"people": [], "organizations": [], "locations": [], "dates": [], "methods_tools_or_products": []},
        "main_topics": [],
        "results_or_conclusions": [],
        "possible_value_for_knowledge_platform": "",
        "confidence_notes": []
    }
    lang_rule = (
        "Write document_summary, main_topics, context, results_or_conclusions, possible_value_for_knowledge_platform and confidence_notes in Dutch."
        if str(detected_language).lower().startswith("nl")
        else "Write document_summary, main_topics, context, results_or_conclusions, possible_value_for_knowledge_platform and confidence_notes in English."
    )
    prompt=f"""
You are a strict document analysis system for a knowledge management platform.
Return valid JSON only.

Language rule:
- Detected document language: {detected_language}
- {lang_rule}
- Keep technical terms in their original form when appropriate.

Rules:
- Use only evidence from the provided text.
- Do not invent facts, names, dates, tools, or results.
- Silver NLP terms/entities are suggestions only; verify them against evidence.
- Return 10 to 15 top_terms when enough evidence exists.
- Each top_term must have a short context sentence that actually mentions or explains the term.
- Suggested entities are suggestions, not final metadata.
- Return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

EVIDENCE:
{evidence}
"""
    start=time.time()
    if require_ollama and not check_ollama():
        raise RuntimeError("Ollama is not reachable. Start Ollama with 'ollama serve' and pull the selected model.")
    try:
        raw=ollama_generate(prompt, model=model, format_json=True)
        result=parse_json_safely(raw)
    except Exception:
        if require_ollama:
            raise
        result=fallback_gold(silver, silver_nlp)
    # Normalize
    result.setdefault("document_summary", "")
    result.setdefault("top_terms", [])
    fallback_terms = terms
    cleaned_terms=[]; seen=set()
    for item in result.get("top_terms", []):
        if not isinstance(item, dict): item={"term": str(item), "context": "", "evidence": ""}
        term=clean_term(item.get("term"))
        if not term or term.lower() in seen: continue
        seen.add(term.lower())
        item["term"]=term
        item["rank"]=len(cleaned_terms)+1
        cleaned_terms.append(item)
    for t in fallback_terms:
        if len(cleaned_terms)>=15: break
        if t.lower() not in seen:
            seen.add(t.lower())
            cleaned_terms.append({"rank":len(cleaned_terms)+1,"term":t,"context":"","evidence":"Silver NLP keyword suggestion"})
    result["top_terms"]=cleaned_terms[:15]
    result["@pipeline"]={
        "document_id": document_id,
        "processing_layer": "gold",
        "processing_version": "gold_local_llm_streamlit_v1_from_notebook",
        "created_at": datetime.now().isoformat(),
        "runtime_seconds": round(time.time()-start,2),
        "local_execution_note": "This layer uses a local Ollama model. It can be upgraded by changing model.",
        "model": model,
        "source_silver_version": silver.get("processing_version"),
        "source_silver_nlp_version": silver_nlp.get("processing_version"),
        "selected_chunks": [c.get("chunk_id") for c in selected],
    }
    write_json(result, gold_dir/f"{document_id}_gold.json")
    # Human readable txt
    lines=[f"# Gold analysis - {document_id}\n", "## Samenvatting / Summary\n", result.get("document_summary",""), "\n## Top termen / Top terms\n"]
    for t in result["top_terms"]:
        lines.append(f"{t.get('rank')}. **{t.get('term')}** — {t.get('context','')}")
    (gold_dir/f"{document_id}_gold_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (gold_dir/f"{document_id}_gold_chunk_outputs.jsonl").write_text("", encoding="utf-8")
    return result


def run_gold_layer(document_ids: list[str] | None = None, data_dir: str | Path = "Data", model: str = LOCAL_MODEL, require_ollama: bool = True, gold_resources=None) -> list[str]:
    silver_dir=Path(data_dir)/"silver"
    if gold_resources and isinstance(gold_resources, dict) and gold_resources.get("model"):
        model=gold_resources["model"]
    if document_ids is None:
        document_ids=[p.name.replace("_silver.json","") for p in sorted(silver_dir.glob("*_silver.json"))]
    paths=[]
    for doc_id in document_ids:
        process_document(doc_id, data_dir=data_dir, model=model, require_ollama=require_ollama)
        paths.append(str(Path(data_dir)/"gold"/f"{doc_id}_gold.json"))
    return paths
