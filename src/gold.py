
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import re
import time
import requests
from collections import Counter

# =========================
# Configuration
# =========================

SILVER_FOLDER = Path("Data/silver")
SILVER_NLP_FOLDER = Path("Data/silver_nlp")
GOLD_FOLDER = Path("Data/gold")
GOLD_FOLDER.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL = "qwen2.5:3b-instruct"

REQUEST_TIMEOUT = 240
MAX_SELECTED_CHUNKS = 8
MAX_EVIDENCE_CHARS = 16000
MIN_CHUNK_WORDS_FOR_LLM = 80

STOPWORDS = {
    "nl": set("""
        de het een en van voor in op aan met als is zijn was wordt worden door dat die dit deze naar om bij uit over ook niet meer maar kan kunnen er hun haar hij zij we wij je u ze of tot dan dus omdat waarin wanneer wie wat waar hoe welke hebben heeft had onder boven tussen per tijdens
    """.split()),
    "en": set("""
        the a an and of for in on to with as is are was were be been being by that this these those from at or not but can could should would have has had it its they them their we our you your he she his her which who what when where how according during between within about around also more such
    """.split()),
}

BAD_KEYWORDS = {
    "pdf", "https", "http", "www", "page", "pagina", "figure", "figuur", "table", "tabel",
    "chapter", "hoofdstuk", "section", "sectie", "appendix", "bijlage", "contents", "inhoudsopgave"
}


# =========================
# Helpers
# =========================

def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=4, ensure_ascii=False), encoding="utf-8")


def clean_space(text: Any) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()
    return text


def clean_llm_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json|text|markdown)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    text = clean_space(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9])", text)
    return [p.strip() for p in parts if 8 <= len(p.split()) <= 90]


def is_noise_sentence(sentence: str) -> bool:
    s = clean_space(sentence)
    if len(s) < 35:
        return True
    low = s.lower()
    if re.search(r"https?://|www\.|__data/assets|\.pdf\b", low):
        return True
    if re.match(r"^\d+\s+[A-Z].{0,80}(?:19|20)\d{2}", s):
        return True
    tokens = s.split()
    if tokens and sum(1 for t in tokens if len(re.sub(r"\W", "", t)) <= 2) / len(tokens) > 0.35:
        return True
    alpha = sum(ch.isalpha() for ch in s)
    if alpha / max(1, len(s)) < 0.55:
        return True
    return False


def normalize_keyword(term: Any) -> str:
    term = clean_space(term).strip(" .,:;|/-")
    term = re.sub(r"[_\[\]{}<>]+", " ", term)
    term = re.sub(r"\s+", " ", term)
    return term


def is_good_keyword(term: str, language: str = "unknown") -> bool:
    term = normalize_keyword(term)
    low = term.lower()
    if not term or len(low) < 3 or low in BAD_KEYWORDS:
        return False
    if any(x in low for x in ["http", "www", ".pdf", "__data", "figure_caption"]):
        return False
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-]+", low)
    if not words:
        return False
    stop = STOPWORDS.get(language, STOPWORDS["en"] | STOPWORDS["nl"])
    if len(words) == 1 and words[0] in stop:
        return False
    if sum(1 for w in words if w in stop) / max(1, len(words)) > 0.55:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)*", low):
        return False
    return True


def load_silver(document_id: str) -> Dict[str, Any]:
    path = SILVER_FOLDER / f"{document_id}_silver.json"
    data = read_json(path)
    if not isinstance(data, dict):
        raise FileNotFoundError(f"Missing or invalid silver file: {path}")
    return data


def load_silver_nlp(document_id: str) -> Dict[str, Any]:
    data = read_json(SILVER_NLP_FOLDER / f"{document_id}_nlp.json", {})
    return data if isinstance(data, dict) else {}


def get_main_text(silver: Dict[str, Any], document_id: str) -> str:
    parts = silver.get("document_parts") if isinstance(silver.get("document_parts"), dict) else {}
    candidates = [
        parts.get("main_text"),
        silver.get("main_text"),
        silver.get("cleaned_text"),
        silver.get("clean_text"),
        silver.get("text"),
    ]
    for c in candidates:
        if isinstance(c, str) and len(c.split()) >= 30:
            return c

    txt_candidates = [
        SILVER_FOLDER / f"{document_id}_clean_main_text.txt",
        SILVER_FOLDER / f"{document_id}_clean.txt",
    ]
    for path in txt_candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text.split()) >= 30:
                return text
    return ""


def get_chunks(silver: Dict[str, Any], document_id: str) -> List[Dict[str, Any]]:
    chunks = silver.get("chunks")
    if isinstance(chunks, list) and chunks:
        normalized = []
        for i, ch in enumerate(chunks, 1):
            if isinstance(ch, dict):
                text = ch.get("text") or ch.get("chunk_text") or ch.get("content") or ""
                normalized.append({
                    "chunk_id": ch.get("chunk_id", f"chunk_{i:03d}"),
                    "source_section_heading": ch.get("source_section_heading") or ch.get("heading") or "",
                    "word_count": len(str(text).split()),
                    "text": str(text),
                })
        if normalized:
            return normalized

    jsonl = SILVER_FOLDER / f"{document_id}_chunks.jsonl"
    if jsonl.exists():
        rows = []
        for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and row.get("text"):
                    rows.append(row)
            except Exception:
                pass
        if rows:
            return rows

    main_text = get_main_text(silver, document_id)
    if main_text:
        return build_fallback_chunks(main_text)
    return []


def build_fallback_chunks(text: str, target_words: int = 900, overlap_words: int = 150) -> List[Dict[str, Any]]:
    words = str(text or "").split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "chunk_id": f"chunk_{len(chunks)+1:03d}",
            "source_section_heading": "fallback_main_text",
            "word_count": len(chunk_text.split()),
            "text": chunk_text,
        })
        if end >= len(words):
            break
        start = max(start + 1, end - overlap_words)
    return chunks


def keyword_hints(silver_nlp: Dict[str, Any], language: str) -> List[str]:
    raw = silver_nlp.get("keyword_suggestions") or silver_nlp.get("keywords") or []
    out = []
    for item in raw:
        if isinstance(item, dict):
            term = item.get("term") or item.get("keyword") or item.get("text")
        else:
            term = item
        term = normalize_keyword(term)
        if is_good_keyword(term, language):
            out.append(term)
    return list(dict.fromkeys(out))[:20]


def score_chunk(chunk: Dict[str, Any], keywords: List[str]) -> float:
    text = clean_space(chunk.get("text", ""))
    low = text.lower()
    if len(text.split()) < 30:
        return -999
    score = min(len(text.split()) / 100, 10)
    score += sum(2 for kw in keywords[:20] if kw.lower() in low)
    if re.search(r"\b(summary|samenvatting|abstract|conclusion|conclusie|results|resultaten|recommendations|aanbevelingen|method|methode)\b", low):
        score += 5
    if re.search(r"https?://|www\.|\.pdf\b|references|bibliography|bronnen|bibliografie", low):
        score -= 4
    return score


def select_chunks(chunks: List[Dict[str, Any]], keywords: List[str], max_chunks: int = MAX_SELECTED_CHUNKS) -> List[Dict[str, Any]]:
    good = [c for c in chunks if len(str(c.get("text", "")).split()) >= MIN_CHUNK_WORDS_FOR_LLM]
    if not good:
        good = [c for c in chunks if len(str(c.get("text", "")).split()) >= 30]
    if not good:
        return []
    scored = sorted(((score_chunk(c, keywords), i, c) for i, c in enumerate(good)), reverse=True)
    picked = sorted(scored[:max_chunks], key=lambda x: x[1])
    return [c for _, _, c in picked]


def extractive_summary(text: str, language: str = "unknown", max_sentences: int = 5) -> str:
    sentences = [s for s in split_sentences(text) if not is_noise_sentence(s)]
    if not sentences:
        return clean_space(text)[:900]

    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{4,}", text.lower())
    stop = STOPWORDS.get(language, STOPWORDS["en"] | STOPWORDS["nl"])
    freqs = Counter(w for w in words if w not in stop and w not in BAD_KEYWORDS)
    top = {w for w, _ in freqs.most_common(30)}

    scored = []
    total = len(sentences)
    for i, s in enumerate(sentences):
        low = s.lower()
        score = sum(1.5 for w in top if w in low)
        score += 2.5 * (1 - i / max(1, total))
        if re.search(r"\b(aim|goal|purpose|result|conclusion|recommendation|doel|resultaat|conclusie|aanbeveling|onderzoek)\b", low):
            score += 4
        scored.append((score, i, s))

    chosen = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in chosen).strip()


def top_terms_from_text(text: str, language: str, existing: Optional[List[str]] = None, max_terms: int = 15) -> List[Dict[str, Any]]:
    terms = []
    seen = set()
    for term in existing or []:
        term = normalize_keyword(term)
        if is_good_keyword(term, language) and term.lower() not in seen:
            seen.add(term.lower())
            terms.append(term)

    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\-]{3,}", text.lower())
    stop = STOPWORDS.get(language, STOPWORDS["en"] | STOPWORDS["nl"])
    for word, _ in Counter(w for w in words if w not in stop and w not in BAD_KEYWORDS).most_common(50):
        if word not in seen:
            seen.add(word)
            terms.append(word)
        if len(terms) >= max_terms:
            break

    sentences = split_sentences(text)
    out = []
    for i, term in enumerate(terms[:max_terms], 1):
        context = ""
        for s in sentences:
            if term.lower() in s.lower() and not is_noise_sentence(s):
                context = s[:450]
                break
        out.append({"rank": i, "term": term, "context": context, "evidence": [context] if context else []})
    return out


def build_evidence_pack(chunks: List[Dict[str, Any]], keywords: List[str], language: str) -> str:
    parts = [
        f"Detected language: {language}",
        "Keyword hints: " + ", ".join(keywords[:20]),
        "",
        "Selected document evidence:"
    ]
    used = 0
    for ch in chunks:
        text = clean_space(ch.get("text", ""))
        if not text:
            continue
        heading = clean_space(ch.get("source_section_heading", ""))
        block = f"\n[{ch.get('chunk_id', 'chunk')}] {heading}\n{text}\n"
        if used + len(block) > MAX_EVIDENCE_CHARS:
            remaining = MAX_EVIDENCE_CHARS - used
            if remaining > 500:
                parts.append(block[:remaining])
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def call_ollama(prompt: str, model: str = LOCAL_MODEL, temperature: float = 0.1, num_ctx: int = 8192) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "top_p": 0.9,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return clean_llm_text(response.json().get("response", ""))


def make_gold_prompt(evidence_pack: str, language: str) -> str:
    output_language = "Dutch" if language == "nl" else "English"
    return f"""You are extracting a reliable knowledge-platform summary from document evidence.

Write all natural-language fields in {output_language}.

Return ONLY valid JSON with this schema:
{{
  "document_summary": "4-6 sentence factual summary",
  "top_terms": [
    {{"rank": 1, "term": "important term", "context": "short context sentence from the evidence"}}
  ],
  "main_topics": ["topic 1", "topic 2", "topic 3"],
  "results_or_conclusions": ["result/conclusion if present"],
  "possible_value_for_knowledge_platform": "why this document is useful"
}}

Rules:
- Use only the evidence.
- Do not invent facts.
- Include 10 to 15 top_terms if possible.
- Every top term should have a non-empty context when possible.
- If evidence is limited, still return the best possible JSON.

EVIDENCE:
{evidence_pack}
"""


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = clean_llm_text(text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def fallback_gold(main_text: str, language: str, keywords: List[str], reason: str) -> Dict[str, Any]:
    summary = extractive_summary(main_text, language, max_sentences=5)
    return {
        "document_summary": summary,
        "summary": summary,
        "top_terms": top_terms_from_text(main_text, language, keywords, max_terms=15),
        "main_topics": keywords[:5],
        "results_or_conclusions": [],
        "possible_value_for_knowledge_platform": summary[:500],
        "fallback_used": True,
        "fallback_reason": reason,
    }


def normalize_gold_result(result: Dict[str, Any], main_text: str, language: str, keywords: List[str]) -> Dict[str, Any]:
    summary = clean_space(result.get("document_summary") or result.get("summary") or "")
    if len(summary.split()) < 25:
        summary = extractive_summary(main_text, language, max_sentences=5)

    raw_terms = result.get("top_terms")
    terms = []
    if isinstance(raw_terms, list):
        for i, item in enumerate(raw_terms, 1):
            if isinstance(item, dict):
                term = normalize_keyword(item.get("term"))
                context = clean_space(item.get("context"))
            else:
                term = normalize_keyword(item)
                context = ""
            if is_good_keyword(term, language):
                if not context:
                    for s in split_sentences(main_text):
                        if term.lower() in s.lower() and not is_noise_sentence(s):
                            context = s[:450]
                            break
                terms.append({"rank": len(terms)+1, "term": term, "context": context, "evidence": [context] if context else []})
    if len(terms) < 10:
        existing = [t["term"] for t in terms] + keywords
        auto_terms = top_terms_from_text(main_text, language, existing, max_terms=15)
        seen = {t["term"].lower() for t in terms}
        for t in auto_terms:
            if t["term"].lower() not in seen:
                terms.append({**t, "rank": len(terms)+1})
            if len(terms) >= 15:
                break

    result["document_summary"] = summary
    result["summary"] = summary
    result["top_terms"] = terms[:15]
    result.setdefault("main_topics", keywords[:5])
    result.setdefault("results_or_conclusions", [])
    result.setdefault("possible_value_for_knowledge_platform", summary[:500])
    return result


def summarize_document(document_id: str, model: str = LOCAL_MODEL) -> Dict[str, Any]:
    start = time.time()
    silver = load_silver(document_id)
    silver_nlp = load_silver_nlp(document_id)
    language = silver.get("language") or silver.get("detected_language") or "unknown"
    main_text = get_main_text(silver, document_id)

    if len(main_text.split()) < 30:
        result = fallback_gold(main_text, language, [], "No usable main_text found in Silver output.")
    else:
        chunks = get_chunks(silver, document_id)
        keywords = keyword_hints(silver_nlp, language)
        selected_chunks = select_chunks(chunks, keywords)

        if not selected_chunks:
            selected_chunks = build_fallback_chunks(main_text)

        evidence_pack = build_evidence_pack(selected_chunks, keywords, language)

        try:
            raw = call_ollama(make_gold_prompt(evidence_pack, language), model=model)
            parsed = parse_json_object(raw)
            if not parsed:
                result = fallback_gold(main_text, language, keywords, "Ollama returned non-JSON or empty output.")
                result["raw_ollama_output"] = raw[:2000]
            else:
                result = normalize_gold_result(parsed, main_text, language, keywords)
                result["fallback_used"] = False
        except requests.exceptions.RequestException as e:
            result = fallback_gold(main_text, language, keywords, f"Ollama request failed: {type(e).__name__}: {e}")

        result["selected_chunk_count"] = len(selected_chunks)
        result["available_chunk_count"] = len(chunks)
        result["main_text_words"] = len(main_text.split())

    result.update({
        "document_id": document_id,
        "language": language,
        "model": model,
        "processing_layer": "gold",
        "processing_version": "gold_local_llm_summary_safe_v2",
        "runtime_seconds": round(time.time() - start, 2),
        "processed_at": datetime.now().isoformat(),
        "local_execution_note": "Gold uses local Ollama when available. If Ollama fails or returns unusable output, an extractive fallback summary is created so the app always has a summary.",
    })
    return result


def save_gold_output(document_id: str, result: Dict[str, Any], data_dir: str | Path = "Data") -> str:
    gold_dir = Path(data_dir) / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    json_path = gold_dir / f"{document_id}_gold.json"
    txt_path = gold_dir / f"{document_id}_gold_summary.txt"
    write_json(result, json_path)

    lines = [
        f"# Gold analysis - {document_id}",
        "",
        "## Samenvatting / Summary",
        "",
        result.get("document_summary") or result.get("summary") or "",
        "",
        "## Top termen / Top terms",
        "",
    ]
    for item in result.get("top_terms", []):
        if isinstance(item, dict):
            lines.append(f"{item.get('rank')}. **{item.get('term')}** — {item.get('context', '')}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return str(json_path)


def process_single_document(document_id: str, model: str = LOCAL_MODEL, data_dir: str | Path = "Data") -> str:
    global SILVER_FOLDER, SILVER_NLP_FOLDER, GOLD_FOLDER
    data_dir = Path(data_dir)
    SILVER_FOLDER = data_dir / "silver"
    SILVER_NLP_FOLDER = data_dir / "silver_nlp"
    GOLD_FOLDER = data_dir / "gold"
    GOLD_FOLDER.mkdir(parents=True, exist_ok=True)

    result = summarize_document(document_id, model=model)
    return save_gold_output(document_id, result, data_dir=data_dir)


def run_gold_layer(document_ids: Optional[List[str]] = None, model: str = LOCAL_MODEL, data_dir: str | Path = "Data") -> List[str]:
    data_dir = Path(data_dir)
    silver_dir = data_dir / "silver"
    if document_ids is None:
        document_ids = [p.name.replace("_silver.json", "") for p in silver_dir.glob("*_silver.json")]
    paths = []
    for document_id in document_ids:
        paths.append(process_single_document(document_id, model=model, data_dir=data_dir))
    return paths

def check_ollama(model: str = LOCAL_MODEL, base_url: str = "http://localhost:11434") -> dict:
    """
    Check whether the local Ollama server is reachable and whether the selected model exists.
    Used by Streamlit before running Gold/Gold Meta.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()

        data = response.json()
        installed_models = []

        for item in data.get("models", []):
            name = item.get("name") or item.get("model")
            if name:
                installed_models.append(name)

        model_available = (
            model in installed_models
            or any(m.startswith(model + ":") for m in installed_models)
            or any(model.startswith(m.split(":")[0]) for m in installed_models)
        )

        return {
            "ok": True,
            "server_running": True,
            "model": model,
            "model_available": model_available,
            "installed_models": installed_models,
            "message": (
                f"Ollama is running and model '{model}' is available."
                if model_available
                else f"Ollama is running, but model '{model}' was not found."
            ),
        }

    except requests.exceptions.RequestException as e:
        return {
            "ok": False,
            "server_running": False,
            "model": model,
            "model_available": False,
            "installed_models": [],
            "message": f"Ollama is not reachable at {base_url}. Start it with: ollama serve",
            "error": f"{type(e).__name__}: {e}",
        }

def process_document(
    document_id: str,
    data_dir: str | Path = "Data",
    model: str = LOCAL_MODEL,
    require_ollama: bool = False,
) -> dict:
    """
    Compatibility wrapper for pipeline.py.
    require_ollama is accepted for compatibility.
    Gold already falls back to extractive summary if Ollama fails.
    """
    json_path = process_single_document(
        document_id=document_id,
        model=model,
        data_dir=data_dir,
    )

    return read_json(Path(json_path), default={})

# Backward-compatible no-op for older app.py versions.
def load_gold_models(*args, **kwargs) -> Dict[str, str]:
    return {"backend": "ollama", "model": kwargs.get("model", LOCAL_MODEL)}
