
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import re
import time
import requests

LOCAL_MODEL = "qwen2.5:3b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT_SECONDS = 900
TEMPERATURE = 0.0
NUM_CTX = 8192

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december"
)
ORG_MARKERS = {
    "university","universiteit","hogeschool","college","school","institute","instituut",
    "province","provincie","municipality","gemeente","ministry","ministerie","department",
    "bedrijf","company","bv","b.v.","nv","n.v.","ltd","inc","gmbh","foundation","stichting",
    "research","lectoraat","consortium","agency","centre","center","applied sciences"
}
SMALL_TITLE_WORDS = {"de","het","een","en","of","van","voor","met","in","op","aan","om","the","a","an","and","or","of","for","with","in","on","to","at","by"}
TITLE_STOP_SECTIONS = {"voorwoord","preface","abstract","samenvatting","inhoudsopgave","contents","table of contents"}
ACRONYMS_TO_KEEP = {"AI","ML","NLP","LLM","CPU","GPU","API","UI","UX","JSON","HTTP","HTTPS","FTP","PDF","PET","WMS","WFS","GIS","ISO","SQL","HTML","CSS","JS","EEG","CNN","RNN","PICU","AUC","ROC","MNE"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=4), encoding="utf-8")


def clean(v) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    try:
        r=requests.get(f"{base_url}/api/tags", timeout=5); r.raise_for_status(); return True
    except Exception:
        return False


def ollama_generate_json(prompt: str, model: str = LOCAL_MODEL) -> str:
    payload={"model": model, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX}}
    r=requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS); r.raise_for_status()
    return r.json().get("response","")


def parse_json_safely(text: str) -> dict:
    text=str(text or "").strip()
    try: return json.loads(text)
    except Exception:
        m=re.search(r"\{.*\}", text, flags=re.S)
        if m: return json.loads(m.group(0))
    raise ValueError("Could not parse JSON")


def get_titlepage_excerpt(silver: dict, max_chars: int = 4500) -> str:
    for key in ["titlepage_text", "front_matter", "first_page_text"]:
        val=silver.get(key)
        if isinstance(val,str) and val.strip(): return val[:max_chars]
    parts=silver.get("document_parts",{})
    if isinstance(parts, dict):
        for key in ["titlepage_text","front_matter","raw_start_text"]:
            val=parts.get(key)
            if isinstance(val,str) and val.strip(): return val[:max_chars]
        # include enough start of main if no front matter
        val=parts.get("main_text","")
        if val: return val[:max_chars]
    return ""


def extract_date(line: str) -> str | None:
    patterns=[
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        rf"\b\d{{1,2}}\s+(?:{MONTHS})(?:st|nd|rd|th)?[,]?\s+\d{{4}}\b",
        rf"\b(?:{MONTHS})\s+\d{{4}}\b",
    ]
    for p in patterns:
        m=re.search(p, line, flags=re.I)
        if m: return m.group(0)
    return None


def is_org_line(line: str) -> bool:
    low=clean(line).lower()
    return any(m in low for m in ORG_MARKERS)


def is_identifier_line(line: str) -> bool:
    line=clean(line)
    if not line: return True
    if re.fullmatch(r"[#A-Za-z]{0,4}\s*\(?\d{4,12}\)?", line): return True
    if re.fullmatch(r"\d{4,12}", line): return True
    return False


def smart_title_case(text: str) -> str | None:
    text=clean(text)
    if not text: return None
    words=[]
    for i,w in enumerate(text.split()):
        core=re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", w)
        if core.upper() in ACRONYMS_TO_KEEP:
            words.append(w.replace(core, core.upper()))
        elif core.isupper() and len(core)>1:
            lw=w.lower()
            words.append(lw if lw in SMALL_TITLE_WORDS and i>0 else lw.capitalize())
        elif i>0 and w.lower() in SMALL_TITLE_WORDS:
            words.append(w.lower())
        else:
            words.append(w)
    result=" ".join(words)
    if ":" not in result:
        # Add colon between short main title and long subtitle when input line evidence contained two title lines.
        result=result.replace("  ", " ")
    return result


def looks_like_person_name(line: str) -> bool:
    line=clean(line)
    if not line or is_identifier_line(line) or is_org_line(line) or extract_date(line): return False
    if re.search(r"\b(title|thesis|report|onderzoek|analysis|model|framework|detection|development|technologie|technology|effective|effectieve|reduction|reductie|climate|klimaat)\b", line, re.I):
        return False
    words=line.split()
    if not 2 <= len(words) <= 5: return False
    if any(w.lower() in SMALL_TITLE_WORDS for w in words): return False
    caps=0
    for w in words:
        core=re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ'.-]", "", w)
        if core and (core[0].isupper() or core.lower() in {"van","de","den","der","ten","ter","von","du","le","la"}):
            caps += 1
    return caps >= max(2, len(words)-1)


def extract_frontmatter_title_author_date(front_text: str) -> dict:
    raw=[clean(x) for x in str(front_text or "").splitlines()]
    lines=[x for x in raw if x]
    front=[]
    for line in lines[:80]:
        low=line.lower().strip(":- ")
        if low in TITLE_STOP_SECTIONS:
            break
        front.append(line)

    date=None; date_idx=None
    for i,line in enumerate(front):
        d=extract_date(line)
        if d:
            date=d; date_idx=i; break

    before_date = front[:date_idx] if date_idx is not None else front[:12]
    before_date = [x for x in before_date if not is_identifier_line(x) and not is_org_line(x) and not extract_date(x)]

    author=None; author_idx=None
    for i in range(len(before_date)-1, -1, -1):
        if looks_like_person_name(before_date[i]):
            author=smart_title_case(before_date[i]); author_idx=i; break

    title_lines = before_date[:author_idx] if author_idx is not None else before_date[:3]
    title_lines=[x for x in title_lines if len(x)>=4 and not looks_like_person_name(x)]
    title=None
    if title_lines:
        title = ": ".join(smart_title_case(x) for x in title_lines[:3])
        title = clean_title(title, authors=[author] if author else [], dates=[date] if date else [])
    return {"title": title, "author": author, "date": date, "front_lines": front[:30]}


def split_trailing_author_from_title(title: str) -> tuple[str | None, str | None]:
    title=clean(title)
    if not title: return None, None
    # Try shortest plausible trailing person first: last 2, then 3 words.
    words=title.split()
    for n in [2,3,4,5]:
        if len(words) <= n+3: continue
        candidate=" ".join(words[-n:])
        remaining=" ".join(words[:-n])
        if looks_like_person_name(candidate) and not re.search(r"\b(for|voor|with|met|of|van|de|the|and|en)$", remaining, re.I):
            return clean_title(remaining), smart_title_case(candidate)
    return title, None


def clean_title(title: str, authors: list[str] | None = None, dates: list[str] | None = None) -> str | None:
    title=clean(title)
    if not title: return None
    title=re.sub(r"\S+@\S+|https?://\S+|www\.\S+", "", title)
    for d in dates or []:
        if d: title=title.replace(d, " ")
    title=re.sub(r"^\s*(?:19|20)\d{2}\s+", "", title)
    title=re.sub(r"\s+\(?\d{4,12}\)?\s*$", "", title)
    for a in authors or []:
        if a: title=re.sub(r"\s+"+re.escape(a)+r"\s*$", "", title, flags=re.I)
    title=clean(title).strip(" -–—:;")
    title, trailing_author = split_trailing_author_from_title(title)
    return smart_title_case(title)


def extract_dates_from_text(text: str) -> list[str]:
    vals=[]
    for line in str(text or "").splitlines()[:120]:
        d=extract_date(line)
        if d and d not in vals: vals.append(d)
    return vals


def fallback_metadata(silver: dict, gold: dict) -> dict:
    lang=silver.get("detected_language") or silver.get("language") or "unknown"
    front=get_titlepage_excerpt(silver)
    parsed=extract_frontmatter_title_author_date(front)
    terms=[]
    for t in gold.get("top_terms", []):
        if isinstance(t, dict) and t.get("term"):
            terms.append(t["term"])
    return {
        "short_summary": gold.get("document_summary"),
        "research_or_project_topic": (gold.get("main_topics") or [None])[0],
        "research_question_or_goal": None,
        "keywords": terms[:15],
        "title": parsed.get("title"),
        "authors": [parsed.get("author")] if parsed.get("author") else [],
        "date": parsed.get("date") or (extract_dates_from_text(front) or [None])[0],
        "document_type": None,
        "language": lang,
        "tools_technologies_or_models": [],
        "main_outputs_or_results": gold.get("results_or_conclusions", []),
        "suitable_kmp_fields": {},
        "contact": {"name": None, "email": None, "phone": None},
        "confidence_notes": ["Fallback deterministic metadata used because Ollama was not reachable or failed."],
    }


def clean_keywords(values: list, fallback_terms: list, max_items: int = 15) -> list[str]:
    out=[]; seen=set()
    for v in list(values or []) + list(fallback_terms or []):
        v=clean(v)
        low=v.lower()
        if not v or low in seen or len(v.split())>7: continue
        if re.fullmatch(r"\d+(?:[.,]\d+)?", low): continue
        if low in SMALL_TITLE_WORDS: continue
        seen.add(low); out.append(v)
        if len(out)>=max_items: break
    return out


def process_document(document_id: str, data_dir: str | Path = "Data", model: str = LOCAL_MODEL, require_ollama: bool = True) -> dict:
    data_dir=Path(data_dir)
    silver=read_json(data_dir/"silver"/f"{document_id}_silver.json")
    gold=read_json(data_dir/"gold"/f"{document_id}_gold.json")
    nlp_path=data_dir/"silver_nlp"/f"{document_id}_silver_nlp.json"
    silver_nlp=read_json(nlp_path) if nlp_path.exists() else {}
    lang=silver.get("detected_language") or silver.get("language") or "unknown"
    front=get_titlepage_excerpt(silver)
    front_parse=extract_frontmatter_title_author_date(front)

    top_terms=[t.get("term") for t in gold.get("top_terms", []) if isinstance(t, dict) and t.get("term")]
    evidence={
        "detected_language": lang,
        "titlepage_excerpt": front[:4500],
        "frontmatter_deterministic_parse": front_parse,
        "silver_titlepage_candidates": silver.get("titlepage_candidates", {}),
        "gold_summary": gold.get("document_summary"),
        "gold_top_terms": gold.get("top_terms", [])[:15],
        "gold_main_topics": gold.get("main_topics", [])[:10],
        "silver_nlp_entity_suggestions": silver_nlp.get("entity_suggestions", {}),
    }
    schema={
        "short_summary": None, "research_or_project_topic": None, "research_question_or_goal": None,
        "keywords": [], "title": None, "authors": [], "date": None, "document_type": None, "language": None,
        "tools_technologies_or_models": [], "main_outputs_or_results": [],
        "suitable_kmp_fields": {"title": None, "description": None, "keywords": [], "contributors": [], "date": None, "language": None, "document_type": None},
        "contact": {"name": None, "email": None, "phone": None}, "confidence_notes": []
    }
    lang_rule = (
        "Write short_summary, research_or_project_topic, research_question_or_goal and confidence_notes in Dutch."
        if str(lang).lower().startswith("nl") else
        "Write short_summary, research_or_project_topic, research_question_or_goal and confidence_notes in English."
    )
    prompt=f"""
You are a strict metadata extraction system for a knowledge management platform.
Return valid JSON only.

Mandatory language rule:
- Detected language: {lang}
- {lang_rule}

Rules:
- Extract metadata only when there is evidence.
- Do not invent titles, names, dates, phone numbers, emails, or fields.
- Do not output subtitle.
- Organizations are not required for KMP metadata.
- The title usually comes from the title page/front matter. Do not include dates, student numbers, author names, page numbers, organizations, or locations in the title.
- If an author is attached to the end of the title, split it into authors.
- Return 10 to 15 meaningful keywords when possible.
- Return exactly one JSON object matching this schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}
"""
    start=time.time()
    if require_ollama and not check_ollama():
        raise RuntimeError("Ollama is not reachable. Start Ollama with 'ollama serve' and pull the selected model.")
    try:
        raw=ollama_generate_json(prompt, model=model)
        metadata=parse_json_safely(raw)
    except Exception:
        if require_ollama: raise
        metadata=fallback_metadata(silver, gold)

    # deterministic strict cleanup
    metadata.setdefault("authors", [])
    if not isinstance(metadata.get("authors"), list): metadata["authors"]=[clean(metadata["authors"])]
    title=metadata.get("title")
    if title:
        title, trailing_author = split_trailing_author_from_title(title)
        if trailing_author and trailing_author not in metadata["authors"]:
            metadata["authors"].append(trailing_author)
    # Prefer frontmatter title if current title is missing/noisy/ends with link word
    noisy = False
    if title:
        if re.search(r"\b\d{4,12}\b", title) or is_org_line(title) or re.search(r"\b(for|voor|with|met|of|van|de|the|and|en)$", title, re.I):
            noisy=True
    if front_parse.get("title") and (not title or noisy or len(title) > len(front_parse["title"])*1.35):
        title=front_parse["title"]
    title=clean_title(title, authors=metadata.get("authors", []), dates=[metadata.get("date"), front_parse.get("date")])
    metadata["title"]=title

    if not metadata.get("authors") and front_parse.get("author"):
        metadata["authors"]=[front_parse["author"]]
    # Remove bad author values that look like title phrases
    clean_authors=[]
    for a in metadata.get("authors", []):
        a=clean(a)
        if looks_like_person_name(a) and a.lower() not in [x.lower() for x in clean_authors]:
            clean_authors.append(smart_title_case(a))
    metadata["authors"]=clean_authors

    if not metadata.get("date"):
        metadata["date"]=front_parse.get("date") or (extract_dates_from_text(front) or [None])[0]
    metadata["language"]=metadata.get("language") or lang
    metadata["short_summary"]=metadata.get("short_summary") or gold.get("document_summary")
    metadata["keywords"]=clean_keywords(metadata.get("keywords", []), top_terms, 15)
    metadata.setdefault("contact", {"name": None, "email": None, "phone": None})
    # No hallucinated contact; keep only if exact evidence
    ev=json.dumps(evidence, ensure_ascii=False).lower()
    for k in ["email","phone","name"]:
        val=clean(metadata.get("contact",{}).get(k))
        if val and val.lower() not in ev:
            metadata["contact"][k]=None

    metadata["suitable_kmp_fields"]={
        "title": metadata.get("title"),
        "description": metadata.get("short_summary"),
        "keywords": metadata.get("keywords", []),
        "contributors": metadata.get("authors", []),
        "date": metadata.get("date"),
        "language": metadata.get("language"),
        "document_type": metadata.get("document_type"),
    }
    metadata["@pipeline"]={
        "document_id": document_id,
        "processing_layer": "gold_meta",
        "processing_version": "gold_meta_local_llm_streamlit_v1_from_notebook",
        "created_at": datetime.now().isoformat(),
        "runtime_seconds": round(time.time()-start,2),
        "local_execution_note": "This layer uses a local Ollama model. It can be upgraded by changing model.",
        "model": model,
        "source_gold_version": gold.get("@pipeline",{}).get("processing_version"),
        "source_silver_version": silver.get("processing_version"),
        "source_silver_nlp_version": silver_nlp.get("processing_version"),
    }
    out_dir=data_dir/"gold_meta"; out_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata, out_dir/f"{document_id}_gold_metadata.json")
    write_json(metadata, out_dir/f"{document_id}_meta.json")
    lines=[f"# Gold Metadata - {document_id}\n",
           f"**Title:** {metadata.get('title') or '-'}",
           f"**Authors:** {', '.join(metadata.get('authors', [])) or '-'}",
           f"**Date:** {metadata.get('date') or '-'}",
           f"**Document type:** {metadata.get('document_type') or '-'}",
           f"**Language:** {metadata.get('language') or '-'}",
           f"**Project/research topic:** {metadata.get('research_or_project_topic') or '-'}",
           f"**Research question/goal:** {metadata.get('research_question_or_goal') or '-'}",
           "\n## Short summary", metadata.get("short_summary") or "-",
           "\n## Keywords"]
    lines.extend([f"- {x}" for x in metadata.get("keywords", [])])
    (out_dir/f"{document_id}_gold_metadata.txt").write_text("\n".join(lines), encoding="utf-8")
    return metadata


def extract_metadata(document_id: str, metadata_text: str | None = None, data_dir: str | Path = "Data", model: str = LOCAL_MODEL, require_ollama: bool = True) -> dict:
    # Compatibility wrapper. metadata_text not used; evidence comes from Silver/Gold.
    return process_document(document_id, data_dir=data_dir, model=model, require_ollama=require_ollama)


def save_metadata(document_id: str, metadata: dict, data_dir: str | Path = "Data") -> str:
    out=Path(data_dir)/"gold_meta"/f"{document_id}_meta.json"; out.parent.mkdir(parents=True, exist_ok=True)
    write_json(metadata, out); return str(out)


def run_gold_meta_layer(document_ids: list[str] | None = None, data_dir: str | Path = "Data", model: str = LOCAL_MODEL, require_ollama: bool = True) -> list[str]:
    silver_dir=Path(data_dir)/"silver"
    if document_ids is None:
        document_ids=[p.name.replace("_silver.json","") for p in sorted(silver_dir.glob("*_silver.json"))]
    paths=[]
    for doc_id in document_ids:
        process_document(doc_id, data_dir=data_dir, model=model, require_ollama=require_ollama)
        paths.append(str(Path(data_dir)/"gold_meta"/f"{doc_id}_gold_metadata.json"))
    return paths
