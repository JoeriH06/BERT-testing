from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

SILVER_FOLDER = Path("Data/silver")
SILVER_NLP_FOLDER = Path("Data/silver_nlp")
GOLD_FOLDER = Path("Data/gold")
GOLD_FOLDER.mkdir(parents=True, exist_ok=True)

SUMMARY_MODELS = {
    "nl": "yhavinga/t5-v1.1-base-dutch-cnn-test",
    "en": "facebook/bart-large-cnn",
    "default": "google/flan-t5-base",
}

STOPWORDS = {
    "nl": set("de het een en van voor in op aan met als is zijn was wordt worden door dat die dit deze naar om bij uit over ook niet meer maar kan kunnen er hun haar hij zij we wij je u ze of tot dan dus omdat waarin wanneer wie wat waar hoe welke hebben heeft had onder boven tussen per aan bij uit rond volgens vanaf binnen tijdens naar over door waarover hierover daarvan hierin daarbij deze".split()),
    "en": set("the a an and of for in on to with as is are was were be been being by that this these those from at or not but can could should would have has had it its they them their we our you your he she his her which who what when where how according during between within about around".split()),
}

BAD_SUMMARY_PATTERNS = [
    "summarize the", "vat de onderstaande", "use only information", "gebruik alleen informatie",
    "do not invent", "verzin geen", "text:", "tekst:", "http://", "https://", "www.",
]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\u00a0", " ")).strip()


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9•])", text)
    return [clean_space(s) for s in pieces if 7 <= len(s.split()) <= 85]


def choose_summary_model(language: str) -> str:
    return SUMMARY_MODELS.get(language, SUMMARY_MODELS["default"])


def load_model(language: str) -> Tuple[str, Any, Any]:
    model_name = choose_summary_model(language)
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        return "extractive_fallback", None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return model_name, tokenizer, model
    except Exception:
        return "extractive_fallback", None, None


def load_gold_models(languages: Optional[Iterable[str]] = None) -> Dict[str, Tuple[str, Any, Any]]:
    languages = list(languages or ["nl", "en", "default"])
    return {language: load_model(language) for language in languages}


def clean_summary_text(summary: str) -> str:
    summary = clean_space(summary)
    summary = re.sub(r"(?:TEXT|TEKST)\s*:\s*", "", summary, flags=re.I)
    summary = re.sub(r"https?://\S+|www\.\S+", "", summary)
    summary = re.sub(r"\s+([.,;:!?])", r"\1", summary)
    return clean_space(summary)


def is_bad_summary(summary: str) -> bool:
    low = (summary or "").lower()
    return len(summary.split()) < 18 or any(p in low for p in BAD_SUMMARY_PATTERNS) or is_noisy_sentence(summary)


def get_summary_length(word_count: int) -> Tuple[int, int]:
    if word_count < 700:
        return 80, 180
    if word_count < 2500:
        return 120, 260
    return 160, 360


def keywords_from_text(text: str, language: str, max_terms: int = 30) -> List[str]:
    stop = STOPWORDS.get(language, STOPWORDS["en"] | STOPWORDS["nl"])
    words = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", text)]
    words = [w for w in words if w not in stop]
    return [w for w, _ in Counter(words).most_common(max_terms)]


def load_modeling_hints(document_id: str) -> Dict[str, Any]:
    path = SILVER_NLP_FOLDER / f"{document_id}_nlp.json"
    if not path.exists():
        return {"main_topics": [], "important_entities": []}
    data = read_json(path)
    if not isinstance(data, dict):
        return {"main_topics": [], "important_entities": []}

    topics = []
    for item in data.get("keywords", []):
        if isinstance(item, dict):
            value = item.get("keyword") or item.get("text") or item.get("term")
        else:
            value = str(item)
        if value:
            topics.append(clean_space(value))

    entities = []
    raw_entities = data.get("entities", {})
    groups = raw_entities.values() if isinstance(raw_entities, dict) else [raw_entities] if isinstance(raw_entities, list) else []
    for group in groups:
        if isinstance(group, list):
            for item in group:
                if isinstance(item, dict):
                    value = item.get("text") or item.get("label") or item.get("value")
                else:
                    value = str(item)
                if value:
                    entities.append(clean_space(value))

    return {
        "main_topics": list(dict.fromkeys([x for x in topics if x]))[:12],
        "important_entities": list(dict.fromkeys([x for x in entities if x]))[:16],
    }


def is_noisy_sentence(sentence: str) -> bool:
    s = clean_space(sentence)
    low = s.lower()
    if len(s) < 25:
        return True
    if re.search(r"\b(?:bron|source|pagina|page|tabel|table|grafiek|figure|figuur|copyright|isbn)\b", low):
        return True
    if re.search(r"\b(?:aantal|totaal)\s+%\s+aantal\b", low):
        return True
    # OCR/table garbage: too many isolated tokens, symbols, or digit fragments.
    tokens = s.split()
    short_tokens = sum(1 for t in tokens if len(re.sub(r"\W", "", t)) <= 2)
    if tokens and short_tokens / len(tokens) > 0.32:
        return True
    if len(re.findall(r"[()*/_]|\b\d\s+\d\b", s)) > 10:
        return True
    alpha = sum(ch.isalpha() for ch in s)
    if alpha / max(len(s), 1) < 0.55:
        return True
    return False


def section_text(text: str, headings: List[str], max_chars: int = 4500) -> str:
    lines = str(text or "").splitlines()
    starts = []
    for i, line in enumerate(lines):
        low = clean_space(line).lower()
        if any(h in low for h in headings):
            starts.append(i)
    if not starts:
        return ""
    start = starts[-1]
    collected = []
    for line in lines[start:start + 80]:
        cleaned = clean_space(line)
        if cleaned:
            collected.append(cleaned)
    return " ".join(collected)[:max_chars]


def sentence_score(sentence: str, keywords: List[str], position: int, total: int) -> float:
    if is_noisy_sentence(sentence):
        return -1000
    low = sentence.lower()
    score = 0.0
    score += sum(2.0 for kw in keywords[:25] if kw.lower() in low)
    score += min(10, len(re.findall(r"\b(?:19|20)\d{2}\b|\b\d+(?:[,.]\d+)?%\b|\b\d+\b", sentence)) * 1.3)
    score += 3 * (1 - position / max(total, 1))
    if re.search(r"\b(conclusion|concludes|summary|samenvatting|conclusie|conclusies|blijkt|result|results|resultaten|aanbeveling|recommendation|scoort|scoren|highest|lowest|hoogste|laagste)\b", low):
        score += 7
    return score


def extractive_summary(text: str, language: str, max_sentences: int = 5) -> str:
    sentences = [s for s in split_sentences(text) if not is_noisy_sentence(s)]
    if not sentences:
        return clean_space(text)[:900]
    keywords = keywords_from_text(text, language)
    scored = [(sentence_score(s, keywords, i, len(sentences)), i, s) for i, s in enumerate(sentences)]
    picked = sorted(scored, reverse=True)[:max_sentences]
    picked = sorted([x for x in picked if x[0] > -100], key=lambda x: x[1])
    return clean_summary_text(" ".join(s for _, _, s in picked))


def summarize_text(text: str, language: str, resources: Optional[Dict[str, Tuple[str, Any, Any]]] = None) -> str:
    text = clean_space(text)
    if not text:
        return ""
    min_len, max_len = get_summary_length(len(text.split()))
    model_name, tokenizer, model = (resources or {}).get(language) or (resources or {}).get("default") or load_model(language)
    if tokenizer is not None and model is not None:
        prompt = "Vat deze tekst feitelijk en neutraal samen in het Nederlands:\n" if language == "nl" else "Summarize this text factually and neutrally:\n"
        try:
            inputs = tokenizer(prompt + text[:6000], return_tensors="pt", truncation=True, max_length=1024)
            output = model.generate(**inputs, min_length=min_len, max_length=max_len, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
            summary = clean_summary_text(tokenizer.decode(output[0], skip_special_tokens=True))
            if not is_bad_summary(summary):
                return summary
        except Exception:
            pass
    return extractive_summary(text, language, max_sentences=5)


def normalize_chunk(chunk: Any, idx: int) -> Dict[str, str]:
    if isinstance(chunk, dict):
        text = chunk.get("text") or chunk.get("chunk_text") or chunk.get("content") or chunk.get("cleaned_text") or ""
        chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk_{idx:03d}"
        return {"chunk_id": str(chunk_id), "text": str(text)}
    return {"chunk_id": f"chunk_{idx:03d}", "text": str(chunk or "")}


def summarize_chunks(chunks: List[Any], language: str, resources: Optional[Dict[str, Tuple[str, Any, Any]]] = None) -> List[Dict[str, str]]:
    results = []
    for idx, raw_chunk in enumerate(chunks, start=1):
        chunk = normalize_chunk(raw_chunk, idx)
        text = chunk["text"]
        if len(text.split()) < 40:
            continue
        summary = summarize_text(text, language, resources)
        if summary and not is_bad_summary(summary):
            results.append({"chunk_id": chunk["chunk_id"], "summary": summary})
    return results


def create_final_summary(text: str, chunk_summaries: List[Dict[str, str]], language: str) -> str:
    # Prefer an explicit conclusions/summary section when present. This is generic and works for reports in NL/EN.
    conclusion = section_text(text, ["conclusies", "conclusie", "conclusions", "conclusion", "samenvatting", "summary"])
    if len(conclusion.split()) >= 40:
        return postprocess_final_summary(extractive_summary(conclusion, language, max_sentences=5), language)
    combined = " ".join(x.get("summary", "") for x in chunk_summaries if x.get("summary"))
    source = combined if len(combined.split()) >= 80 else text
    return postprocess_final_summary(extractive_summary(source, language, max_sentences=5), language)


def postprocess_final_summary(summary: str, language: str) -> str:
    summary = clean_summary_text(summary)
    sentences = split_sentences(summary)
    kept, seen = [], set()
    for sent in sentences:
        if is_noisy_sentence(sent):
            continue
        key = re.sub(r"\W+", " ", sent.lower())[:100]
        if key not in seen:
            seen.add(key)
            kept.append(sent)
    return " ".join(kept[:5]) or summary


def read_silver_document(document_id: str) -> Dict[str, Any]:
    path = SILVER_FOLDER / f"{document_id}_silver.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing silver file: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Silver file must contain a JSON object: {path}")
    return data


def save_gold_output(document_id: str, data: Dict[str, Any]) -> str:
    GOLD_FOLDER.mkdir(parents=True, exist_ok=True)
    out = GOLD_FOLDER / f"{document_id}_gold.json"
    out.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    return str(out)


def process_single_document(document_id: str, gold_resources: Optional[Dict[str, Tuple[str, Any, Any]]] = None) -> str:
    silver = read_silver_document(document_id)
    language = silver.get("language", "unknown")
    text = silver.get("cleaned_text") or silver.get("text") or silver.get("content") or ""
    raw_chunks = silver.get("chunks") or silver.get("chunk_records") or silver.get("cleaned_chunks") or []
    if isinstance(raw_chunks, str):
        raw_chunks = [raw_chunks]
    if not isinstance(raw_chunks, list):
        raw_chunks = []
    chunks = [normalize_chunk(chunk, idx) for idx, chunk in enumerate(raw_chunks, start=1)] or [{"chunk_id": "chunk_001", "text": text}]

    hints = load_modeling_hints(document_id)
    chunk_summaries = summarize_chunks(chunks, language, gold_resources)
    final_summary = create_final_summary(text, chunk_summaries, language)
    model_tuple = (gold_resources or {}).get(language) or (gold_resources or {}).get("default") or ("extractive_fallback", None, None)
    output = {
        "document_id": document_id,
        "language": language,
        "summary": final_summary,
        "chunk_summaries": chunk_summaries,
        "modeling_hints_used": hints,
        "model": model_tuple[0],
        "processed_at": datetime.now().isoformat(),
    }
    return save_gold_output(document_id, output)


def run_gold_layer(document_ids: Optional[List[str]] = None, gold_resources: Optional[Dict[str, Tuple[str, Any, Any]]] = None) -> List[str]:
    if document_ids is None:
        document_ids = [p.name.replace("_silver.json", "") for p in SILVER_FOLDER.glob("*_silver.json")]
    return [process_single_document(document_id, gold_resources) for document_id in document_ids]
