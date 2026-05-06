from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import spacy
except Exception:
    spacy = None

SILVER_FOLDER = Path("Data/silver")
GOLD_META_FOLDER = Path("Data/gold_meta")
GOLD_META_FOLDER.mkdir(parents=True, exist_ok=True)

SPACY_MODELS = {"nl": "nl_core_news_sm", "en": "en_core_web_sm"}

MONTH_WORDS = [
    "januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december",
    "jan", "feb", "mrt", "apr", "jun", "jul", "aug", "sep", "sept", "okt", "nov", "dec",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
]
MONTHS = "|".join(MONTH_WORDS)

TITLE_LABELS = {"title", "titel", "document title", "rapporttitel", "report title", "naam", "name"}
DATE_LABELS = {"date", "datum", "publication date", "publicatiedatum", "published", "gepubliceerd", "version date", "versiedatum"}
CONTACT_LABELS = {"contact", "contactpersoon", "contact person", "informatie", "information"}

AUTHOR_LABELS = {
    "author", "authors", "auteur", "auteurs", "by", "door", "written by", "geschreven door",
    "prepared by", "opgesteld door", "compiled by", "samengesteld door", "samenstelling",
    "redactie", "editor", "editors", "eindredactie", "onderzoeker", "onderzoekers", "researcher", "researchers",
    "project team", "projectteam", "contributors", "contributor", "bijdrager", "bijdragers",
}
ORG_LABELS = {
    "publisher", "uitgever", "commissioned by", "in opdracht van", "client", "opdrachtgever",
    "organisation", "organization", "organisatie", "institution", "instituut", "department", "afdeling",
}

SECTION_WORDS = {
    "contents", "inhoud", "inhoudsopgave", "table of contents", "colofon", "colophon", "credits",
    "summary", "samenvatting", "introduction", "inleiding", "appendix", "bijlage", "references", "bronnen",
    "conclusies", "conclusions", "onderzoeksverantwoording", "methodology", "methode",
}
ORG_HINTS = r"\b(?:b\.v\.|bv|n\.v\.|nv|ltd|limited|inc|llc|gmbh|foundation|stichting|vereniging|gemeente|university|universiteit|hogeschool|ministerie|agency|instituut|institute|planbureau|bibliotheek|library|ggd|politie|bureau)\b"


def clean_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\u00a0", " ")).strip(" \t\n\r-–—:;,.|")


def unique_keep_order(values: Iterable[Any]) -> List[str]:
    seen, out = set(), []
    for value in values:
        value = clean_space(value)
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_document_language(document_id: str) -> str:
    path = SILVER_FOLDER / f"{document_id}_silver.json"
    if not path.exists():
        return "unknown"
    data = read_json(path)
    return data.get("language", "unknown") if isinstance(data, dict) else "unknown"


def read_metadata_text(document_id: str) -> str:
    path = SILVER_FOLDER / f"{document_id}_silver.json"
    if not path.exists():
        return ""
    data = read_json(path)
    if not isinstance(data, dict):
        return ""
    return data.get("metadata_text") or data.get("cleaned_text") or data.get("text") or ""


def front_matter(text: str, max_chars: int = 9000, max_lines: int = 160) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines()]
    return "\n".join(lines[:max_lines])[:max_chars]


def split_lines(text: str) -> List[str]:
    return [clean_space(line) for line in str(text or "").splitlines() if clean_space(line)]


def labeled_value_pairs(text: str) -> List[Tuple[str, str]]:
    pairs = []
    for raw in split_lines(text):
        if ":" in raw:
            label, value = raw.split(":", 1)
            label, value = clean_space(label).lower(), clean_space(value)
            if 1 <= len(label.split()) <= 6 and value:
                pairs.append((label, value))
    return pairs


def is_contact_or_address(line: str) -> bool:
    low = line.lower()
    return bool(
        re.search(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", line, re.I)
        or re.search(r"https?://|www\.", low)
        or re.search(r"\b(?:tel|phone|telefoon|fax|e-mail|email|postbus|postcode|zip|adres|address)\b", low)
        or re.search(r"\b\d{4}\s?[a-z]{2}\b", low)
    )


def is_page_or_table_noise(line: str) -> bool:
    low = line.lower().strip(" :-–—")
    if low in SECTION_WORDS:
        return True
    if re.fullmatch(r"\d+", low):
        return True
    if re.match(r"^(page|pagina|figure|figuur|table|tabel|grafiek)\s*\d*", low):
        return True
    if re.search(r"\b(?:aantal|totaal|bron|source)\b", low) and len(line.split()) < 8:
        return True
    if len(line) > 180:
        return True
    return False


def date_pattern() -> str:
    return rf"\b(?:\d{{1,2}}[\-/ ](?:\d{{1,2}}|{MONTHS})[\-/ ]\d{{2,4}}|(?:{MONTHS})\s+\d{{4}}|\d{{4}}[/-]\d{{2,4}}|\d{{4}})\b"


def looks_like_date_only(line: str) -> bool:
    return bool(re.fullmatch(date_pattern(), clean_space(line), re.I))


def title_score(line: str, idx: int) -> float:
    line = clean_space(line)
    if not line or is_contact_or_address(line) or is_page_or_table_noise(line) or looks_like_date_only(line):
        return -1000
    words = line.split()
    if not 2 <= len(words) <= 16:
        return -100
    score = 100 - idx * 2.5
    low = line.lower()
    if re.search(r"\b(?:rapport|report|onderzoek|study|analysis|analyse|monitor|dashboard|plan|strategy|strategie|policy|beleid|manual|handleiding|proposal|voorstel|onderzoek)\b", low):
        score += 25
    if re.search(r"\b(?:19|20)\d{2}(?:[/-](?:19|20)?\d{2})?\b", line):
        score += 10
    if line.endswith((".", ",", ";")):
        score -= 25
    if re.search(ORG_HINTS, line, re.I):
        score -= 20
    if line.isupper() and len(words) > 3:
        score -= 10
    return score


def extract_title_and_subtitle(text: str) -> Tuple[Optional[str], Optional[str]]:
    front = front_matter(text)
    for label, value in labeled_value_pairs(front):
        if label in TITLE_LABELS and 4 <= len(value) <= 180:
            return value, None

    lines = split_lines(front)
    if not lines:
        return None, None

    scored = [(title_score(line, i), i, line) for i, line in enumerate(lines[:50])]
    scored.sort(reverse=True, key=lambda x: x[0])
    if not scored or scored[0][0] < 0:
        return None, None

    _, title_idx, title = scored[0]
    subtitle = None
    for nxt in lines[title_idx + 1:title_idx + 5]:
        if nxt == title or is_contact_or_address(nxt) or is_page_or_table_noise(nxt) or looks_like_date_only(nxt):
            continue
        if 3 <= len(nxt.split()) <= 24 and not re.search(ORG_HINTS, nxt, re.I):
            subtitle = nxt
            break
    return title, subtitle


def extract_dates(text: str) -> Dict[str, Any]:
    front = front_matter(text, 12000, 200)
    pattern = date_pattern()
    dates = unique_keep_order(m.group(0) for m in re.finditer(pattern, front, flags=re.I))

    publication_date = None
    for label, value in labeled_value_pairs(front):
        if label in DATE_LABELS:
            match = re.search(pattern, value, flags=re.I)
            if match:
                publication_date = match.group(0)
                break

    if not publication_date:
        # Prefer month-year over bare year when available.
        publication_date = next((d for d in dates if re.search(rf"(?:{MONTHS})", d, re.I)), None) or (dates[0] if dates else None)

    years = []
    for y in re.findall(r"\b(?:19|20)\d{2}\b", front):
        try:
            years.append(int(y))
        except ValueError:
            pass

    # Keep start/end only when there is a clear period/range, otherwise they become misleading.
    range_match = re.search(r"\b((?:19|20)\d{2})\s*(?:/|-|–|to|t/m|tot)\s*((?:19|20)?\d{2})\b", front, re.I)
    start_date = end_date = None
    if range_match:
        start_date = range_match.group(1)
        end = range_match.group(2)
        end_date = end if len(end) == 4 else start_date[:2] + end

    return {"publication_date": publication_date, "start_date": start_date, "end_date": end_date, "dates_found": dates[:20]}


def split_people(value: str) -> List[str]:
    value = re.sub(r"\bet\s+al\.?\b", "", value, flags=re.I)
    # Do NOT split on hyphen/dash. Dutch surnames often contain “– van …”.
    parts = re.split(r"\s*(?:,|;|\band\b|\ben\b|&|/|\|)\s*", value)
    return [clean_space(p) for p in parts if clean_space(p)]


def is_valid_person(name: str) -> bool:
    name = clean_space(name)
    if not 5 <= len(name) <= 90:
        return False
    if any(ch.isdigit() for ch in name) or is_contact_or_address(name):
        return False
    if re.search(ORG_HINTS, name, re.I):
        return False
    words = name.split()
    if not 2 <= len(words) <= 8:
        return False
    low = name.lower()
    if re.search(r"\b(?:rapport|onderzoek|study|monitor|naleving|gemeente|pagina|tabel|grafiek|bron|zeeland)$", low):
        return False
    # Accept initials, particles, and capitalized words.
    valid_tokens = 0
    for w in words:
        if re.match(r"^[A-ZÀ-Ý][A-Za-zÀ-ÿ'.-]+$", w) or w.lower() in {"van", "de", "den", "der", "ten", "ter", "op", "von", "du", "le", "la"}:
            valid_tokens += 1
    return valid_tokens >= len(words) - 1


def is_valid_organization(name: str) -> bool:
    name = clean_space(name)
    if not 3 <= len(name) <= 130 or is_contact_or_address(name):
        return False
    return bool(re.search(ORG_HINTS, name, re.I))


def load_spacy_model(language: str):
    if spacy is None:
        return None
    model_name = SPACY_MODELS.get(language)
    if not model_name:
        return None
    try:
        return spacy.load(model_name)
    except Exception:
        return None


def extract_labeled_contributors(text: str) -> Dict[str, List[str]]:
    out = {"persons": [], "organizations": [], "authors": [], "editors": [], "compiled_by": [], "prepared_by": [], "commissioned_by": []}
    for label, value in labeled_value_pairs(front_matter(text, 14000, 220)):
        low = label.lower()
        is_author = low in AUTHOR_LABELS or any(x in low for x in AUTHOR_LABELS)
        is_org = low in ORG_LABELS or any(x in low for x in ORG_LABELS)
        if not (is_author or is_org):
            continue
        for part in split_people(value):
            if is_valid_person(part):
                out["persons"].append(part)
                if "editor" in low or "redactie" in low:
                    out["editors"].append(part)
                elif "compiled" in low or "samenstelling" in low:
                    out["compiled_by"].append(part)
                elif "prepared" in low or "opgesteld" in low:
                    out["prepared_by"].append(part)
                else:
                    out["authors"].append(part)
            elif is_valid_organization(part):
                out["organizations"].append(part)
                if is_org:
                    out["commissioned_by"].append(part)
    return {k: unique_keep_order(v) for k, v in out.items()}


def extract_cover_people(text: str) -> List[str]:
    lines = split_lines(front_matter(text, 4000, 60))
    people = []
    for i, line in enumerate(lines):
        if is_valid_person(line):
            # Cover authors often appear after a title/subtitle/date block and before TOC.
            if i < 45 and not is_page_or_table_noise(line):
                people.append(line)
    return unique_keep_order(people)


def extract_spacy_contributors(text: str, language: str) -> Dict[str, List[str]]:
    nlp = load_spacy_model(language)
    out = {"persons": [], "organizations": []}
    if nlp is None:
        return out
    doc = nlp(front_matter(text, 12000, 200))
    for ent in doc.ents:
        value = clean_space(ent.text)
        label = ent.label_.upper()
        if label in {"PER", "PERSON"} and is_valid_person(value):
            out["persons"].append(value)
        elif label in {"ORG", "ORGANIZATION"} and is_valid_organization(value):
            out["organizations"].append(value)
    return {k: unique_keep_order(v) for k, v in out.items()}


def extract_contributors(text: str, language: str = "unknown") -> Dict[str, Any]:
    labeled = extract_labeled_contributors(text)
    cover_people = extract_cover_people(text)
    ner = extract_spacy_contributors(text, language)
    persons = unique_keep_order(labeled["persons"] + cover_people + ner["persons"])
    orgs = unique_keep_order(labeled["organizations"] + ner["organizations"])
    authors = unique_keep_order(labeled["authors"] + cover_people)
    confidence = "high" if authors else "medium" if persons or orgs else "low"
    return {
        "contributors": authors or persons or orgs,
        "contributors_structured": {
            "persons": persons,
            "organizations": orgs,
            "authors": authors,
            "editors": unique_keep_order(labeled["editors"]),
            "compiled_by": unique_keep_order(labeled["compiled_by"]),
            "prepared_by": unique_keep_order(labeled["prepared_by"]),
            "commissioned_by": unique_keep_order(labeled["commissioned_by"]),
            "confidence": confidence,
        },
        "contributors_confidence": confidence,
    }


def extract_description(text: str, title: Optional[str], max_chars: int = 750) -> Optional[str]:
    paragraphs = [clean_space(p) for p in re.split(r"\n{2,}|(?<=[.!?])\s+", str(text or "")) if clean_space(p)]
    for p in paragraphs[:100]:
        if p == title or is_page_or_table_noise(p) or is_contact_or_address(p):
            continue
        if 18 <= len(p.split()) <= 100:
            return p[:max_chars]
    return None


def extract_email(text: str) -> Optional[str]:
    m = re.search(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", text, flags=re.I)
    return m.group(0) if m else None


def extract_phone(text: str) -> Optional[str]:
    # Avoid treating years/ranges like 2018-2019 as phone numbers.
    candidates = re.findall(r"(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?){2,5}\d{2,4}", text)
    for cand in candidates:
        digits = re.sub(r"\D", "", cand)
        if len(digits) >= 9 and not re.fullmatch(r"(?:19|20)\d{2}(?:19|20)?\d{2}", digits):
            return clean_space(cand)
    return None


def extract_contact_person(text: str) -> Dict[str, Any]:
    front = front_matter(text, 12000, 220)
    email = extract_email(front)
    phone = extract_phone(front)
    name = None
    for line in split_lines(front):
        low = line.lower()
        if any(label in low for label in CONTACT_LABELS):
            parts = re.split(r":|–|-", line, maxsplit=1)
            if len(parts) == 2 and is_valid_person(parts[1]):
                name = clean_space(parts[1])
                break
    return {"name": name, "email": email, "phone": phone, "confidence": "medium" if email or phone or name else "low"}


def extract_metadata(document_id: str, metadata_text: Optional[str] = None) -> Dict[str, Any]:
    text = metadata_text if metadata_text is not None else read_metadata_text(document_id)
    language = get_document_language(document_id)
    title, subtitle = extract_title_and_subtitle(text)
    contributors = extract_contributors(text, language)
    dates = extract_dates(text)
    return {
        "document_id": document_id,
        "language": language,
        "title": title,
        "subtitle": subtitle,
        "description": extract_description(text, title),
        **contributors,
        "contact_person": extract_contact_person(text),
        "dates": dates,
        "extracted_at": datetime.now().isoformat(),
    }


def save_metadata(document_id: str, metadata: Dict[str, Any]) -> str:
    GOLD_META_FOLDER.mkdir(parents=True, exist_ok=True)
    out = GOLD_META_FOLDER / f"{document_id}_meta.json"
    out.write_text(json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8")
    return str(out)


def run_gold_meta_layer(document_ids: Optional[List[str]] = None) -> List[str]:
    if document_ids is None:
        document_ids = [p.name.replace("_silver.json", "") for p in SILVER_FOLDER.glob("*_silver.json")]
    paths = []
    for document_id in document_ids:
        metadata = extract_metadata(document_id)
        paths.append(save_metadata(document_id, metadata))
    return paths
