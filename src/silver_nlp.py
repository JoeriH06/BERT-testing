
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import json
import math
import re
import unicodedata

try:
    import spacy
except Exception:
    spacy = None

SPACY_MODELS = {"nl": ["nl_core_news_sm", "xx_ent_wiki_sm"], "en": ["en_core_web_sm", "xx_ent_wiki_sm"], "unknown": ["xx_ent_wiki_sm", "en_core_web_sm", "nl_core_news_sm"]}
_MODEL_CACHE = {}

STOPWORDS = {
    "nl": set("de het een en of in op te van voor met dat dit die als aan om er is zijn wordt worden was waren heeft hebben had door bij uit naar ook maar dan dus kan kunnen zal zullen moet moeten waar wat hoe welke wie wanneer tijdens alleen nieuwe gemaakt gebruikt maken gebruik bevat basis manier twee tijd team onderzoek ontwikkelen ontwikkeld plaatsen kaart hoofdstuk conclusie resultaten methode".split()),
    "en": set("the a an and or in on to of for with that this these those as by from at it its be is are was were has have had can could should would will may might what how which who when during only new made used use using make contains basis way two time team research develop developed chapter conclusion results methodology".split()),
}
GENERIC_STOPWORDS = STOPWORDS["nl"] | STOPWORDS["en"]

BAD_MARKERS = {"figure_caption","table_start","table_end","appendix","bijlage","references","bibliografie","contents","inhoudsopgave"}


def clean_text_value(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\u00a0", " ")).strip()


def load_spacy_model(language: str):
    if spacy is None:
        return None, None
    key = language if language in SPACY_MODELS else "unknown"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    for name in SPACY_MODELS[key]:
        try:
            nlp = spacy.load(name)
            _MODEL_CACHE[key] = (nlp, name)
            return nlp, name
        except Exception:
            continue
    _MODEL_CACHE[key] = (None, None)
    return None, None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists(): return []
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalize_token(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").strip().lower()
    return re.sub(r"^[^\wÀ-ÿ]+|[^\wÀ-ÿ]+$", "", t)


def is_bad_term(term: str) -> bool:
    term = clean_text_value(term)
    low = term.lower()
    if not term or low in GENERIC_STOPWORDS:
        return True
    if any(m in low for m in BAD_MARKERS):
        return True
    if re.fullmatch(r"\d+(?:[.,]\d+)?", low):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+", low):
        return True
    words = re.findall(r"[A-Za-zÀ-ÿ0-9\-]+", low)
    if not words or len(words) > 6:
        return True
    stop_ratio = sum(1 for w in words if w in GENERIC_STOPWORDS) / max(1, len(words))
    if stop_ratio > 0.55:
        return True
    return False


def candidate_phrases(text: str, max_n: int = 4) -> list[str]:
    tokens = [normalize_token(t) for t in re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\-']+", text)]
    tokens = [t for t in tokens if t]
    phrases=[]
    for n in range(1, max_n+1):
        for i in range(0, max(0, len(tokens)-n+1)):
            gram = tokens[i:i+n]
            if all(g in GENERIC_STOPWORDS for g in gram): continue
            if gram[0] in GENERIC_STOPWORDS or gram[-1] in GENERIC_STOPWORDS: continue
            term = " ".join(gram)
            if not is_bad_term(term):
                phrases.append(term)
    return phrases


def get_context(text: str, term: str, width: int = 220) -> str:
    m = re.search(re.escape(term), text, flags=re.I)
    if not m:
        return ""
    start=max(0, m.start()-width//2); end=min(len(text), m.end()+width//2)
    return clean_text_value(text[start:end])


def keyword_suggestions(chunks: list[dict], max_terms: int = 30) -> list[dict]:
    freq=Counter(); sections=defaultdict(set); contexts={}
    for ch in chunks:
        text=ch.get("text","")
        sid=ch.get("source_section_id") or ch.get("chunk_id")
        local=Counter(candidate_phrases(text))
        for term,count in local.items():
            freq[term]+=count
            sections[term].add(sid)
            contexts.setdefault(term, get_context(text, term))
    scored=[]
    section_total=max(1, len({ch.get("source_section_id") or ch.get("chunk_id") for ch in chunks}))
    for term,count in freq.items():
        if count < 2 and len(term.split()) == 1:
            continue
        spread=len(sections[term])
        score=count*(1+min(1.5, spread/section_total))
        # Prefer multiword/domain terms slightly
        score += 0.7*(len(term.split())-1)
        scored.append((score, term, count, spread, contexts.get(term,"")))
    scored.sort(reverse=True)
    out=[]
    for rank,(score,term,count,spread,ctx) in enumerate(scored[:max_terms], start=1):
        out.append({"term": term, "score": round(score,3), "frequency": count, "section_spread": spread, "section_spread_ratio": round(spread/section_total,3), "context_example": ctx, "note": "generic_keyword_suggestion_based_on_frequency_spread_and_context", "rank": rank})
    return out


def valid_entity_text(text: str) -> bool:
    text = clean_text_value(text)
    if len(text) < 3 or len(text) > 120: return False
    if not any(c.isalpha() for c in text): return False
    if len(text.split()) > 8: return False
    if re.match(r"^\d+(?:\.\d+)*\s+", text): return False
    if text.lower() in GENERIC_STOPWORDS: return False
    if any(m in text.lower() for m in BAD_MARKERS): return False
    return True


def fallback_entities(text: str) -> dict:
    # Generic fallback: acronyms and capitalized sequences only; suggestions, not truth.
    caps = re.findall(r"\b(?:[A-ZÀ-Ý][A-Za-zÀ-ÿ'.-]+(?:\s+|$)){2,6}", text)
    acronyms = re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", text)
    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:19|20)\d{2}\b", text)
    def pack(vals, confidence="low"):
        c=Counter(clean_text_value(v) for v in vals if valid_entity_text(v))
        return [{"text": k, "frequency": v, "confidence": confidence, "note":"generic_suggestion_only_not_verified_metadata"} for k,v in c.most_common(25)]
    return {
        "people": [],
        "organizations": pack(caps, "low"),
        "locations": [],
        "dates": [{"text": d, "frequency": c, "confidence":"medium", "note":"generic_suggestion_only_not_verified_metadata"} for d,c in Counter(dates).most_common(25)],
        "methods_tools_or_products": pack(acronyms, "medium"),
        "documents_or_standards": [],
        "events": [],
        "groups_or_nationalities": [],
    }


def spacy_entities(text: str, language: str) -> tuple[dict, bool, str | None]:
    nlp, model_name = load_spacy_model(language)
    if nlp is None:
        return fallback_entities(text), False, None
    doc = nlp(text[:900000])
    groups = {"people": [], "organizations": [], "locations": [], "dates": [], "methods_tools_or_products": [], "documents_or_standards": [], "events": [], "groups_or_nationalities": []}
    for ent in doc.ents:
        value=clean_text_value(ent.text)
        if not valid_entity_text(value): continue
        label=ent.label_.upper()
        if label in {"PER","PERSON"}: groups["people"].append(value)
        elif label in {"ORG","ORGANIZATION"}: groups["organizations"].append(value)
        elif label in {"LOC","GPE","LOCATION"}: groups["locations"].append(value)
        elif label in {"DATE","TIME"}: groups["dates"].append(value)
        elif label in {"PRODUCT","WORK_OF_ART","LAW"}: groups["methods_tools_or_products"].append(value)
        elif label in {"EVENT"}: groups["events"].append(value)
        elif label in {"NORP"}: groups["groups_or_nationalities"].append(value)
    # add acronym tools generically
    groups["methods_tools_or_products"].extend(re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b", text))
    packed={}
    for k, vals in groups.items():
        c=Counter(clean_text_value(v) for v in vals if valid_entity_text(v))
        conf="medium" if k in {"dates","methods_tools_or_products"} else "low"
        packed[k]=[{"text": t, "frequency": n, "confidence": conf, "note":"generic_suggestion_only_not_verified_metadata"} for t,n in c.most_common(25)]
    return packed, True, model_name


def process_document(document_id: str, data_dir: str | Path = "Data") -> dict:
    data_dir=Path(data_dir)
    silver_dir=data_dir/"silver"; out_dir=data_dir/"silver_nlp"; out_dir.mkdir(parents=True, exist_ok=True)
    silver_path=silver_dir/f"{document_id}_silver.json"
    chunks_path=silver_dir/f"{document_id}_chunks.jsonl"
    silver=read_json(silver_path)
    chunks=read_jsonl(chunks_path) or silver.get("chunks", [])
    main_text=silver.get("document_parts",{}).get("main_text","")
    language=silver.get("detected_language") or silver.get("language") or "unknown"

    kws=keyword_suggestions(chunks)
    ents, spacy_available, model_name = spacy_entities(main_text, language)
    chunk_nlp=[]
    for ch in chunks:
        ckws=keyword_suggestions([ch], max_terms=15)
        cents, _, _ = spacy_entities(ch.get("text",""), language)
        chunk_nlp.append({
            "chunk_id": ch.get("chunk_id"),
            "source_section_id": ch.get("source_section_id"),
            "source_section_heading": ch.get("source_section_heading"),
            "word_count": ch.get("word_count"),
            "keyword_suggestions": ckws,
            "entity_suggestions": cents,
        })

    output={
        "document_id": document_id,
        "original_file": silver.get("original_file"),
        "detected_language": language,
        "language": language,
        "processing_layer": "silver_nlp",
        "processing_version": "silver_nlp_local_generic_streamlit_v1_from_notebook",
        "created_at": datetime.now().isoformat(),
        "source_silver_version": silver.get("processing_version"),
        "local_execution_note": "This layer runs locally and does not call Ollama. It is compatible with any later Gold/Gold Meta Ollama model because the output schema is model-independent.",
        "genericity_note": "No document-specific terms, persons, organizations, locations, or project names are hardcoded. Suggestions are generated from the input document itself using generic NL/EN rules and optional local spaCy.",
        "important_note": "All entities and keywords in this layer are suggestions only. They are intended as hints for Gold and Gold Meta, not as final metadata.",
        "statistics": {
            "main_text_words": len(main_text.split()),
            "section_count": len(silver.get("sections", [])),
            "chunk_count": len(chunks),
            "keyword_suggestion_count": len(kws),
            "entity_suggestion_count_total": sum(len(v) for v in ents.values()),
            "spacy_available": spacy_available,
            "spacy_model": model_name,
        },
        "titlepage_candidates_from_silver": silver.get("titlepage_candidates", {}),
        "keyword_suggestions": kws,
        "entity_suggestions": ents,
        "chunk_nlp": chunk_nlp,
    }
    (out_dir/f"{document_id}_silver_nlp.json").write_text(json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8")
    (out_dir/f"{document_id}_keyword_suggestions.json").write_text(json.dumps({"document_id":document_id,"important_note":output["important_note"],"keyword_suggestions":kws}, indent=4, ensure_ascii=False), encoding="utf-8")
    (out_dir/f"{document_id}_entity_suggestions.json").write_text(json.dumps({"document_id":document_id,"important_note":output["important_note"],"entity_suggestions":ents}, indent=4, ensure_ascii=False), encoding="utf-8")
    with (out_dir/f"{document_id}_chunk_nlp.jsonl").open("w", encoding="utf-8") as f:
        for r in chunk_nlp: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    # compat
    (out_dir/f"{document_id}_nlp.json").write_text(json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8")
    return output


def run_silver_nlp_layer(document_ids: list[str] | None = None, data_dir: str | Path = "Data") -> list[str]:
    silver_dir=Path(data_dir)/"silver"
    if document_ids is None:
        document_ids=[p.name.replace("_silver.json","") for p in sorted(silver_dir.glob("*_silver.json"))]
    paths=[]
    for doc_id in document_ids:
        process_document(doc_id, data_dir=data_dir)
        paths.append(str(Path(data_dir)/"silver_nlp"/f"{doc_id}_silver_nlp.json"))
    return paths
