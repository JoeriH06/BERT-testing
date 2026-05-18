
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import re
import unicodedata

try:
    from langdetect import detect as _langdetect
except Exception:
    _langdetect = None

TARGET_CHUNK_WORDS = 900
MAX_CHUNK_WORDS = 1200
MIN_CHUNK_WORDS = 120
CHUNK_OVERLAP_WORDS = 180

DUTCH_CUES = {"hoofdstuk", "inhoudsopgave", "samenvatting", "voorwoord", "bijlage", "bijlagen", "bibliografie", "onderzoek", "methode", "resultaten", "conclusie"}
ENGLISH_CUES = {"chapter", "contents", "table of contents", "abstract", "preface", "appendix", "appendices", "references", "introduction", "methodology", "results", "conclusion"}
MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december"
)

TOC_LABELS = {"contents", "table of contents", "inhoud", "inhoudsopgave"}
BODY_START_WORDS = {
    "introduction", "inleiding", "chapter", "hoofdstuk", "methodology", "methode",
    "theoretical framework", "theoretisch kader", "background", "achtergrond"
}
REFERENCE_HEADINGS = {"references", "bibliography", "bibliografie", "literature", "literatuurlijst", "bronnen"}
APPENDIX_HEADINGS = {"appendix", "appendices", "bijlage", "bijlagen"}


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", str(line or "").strip())


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    replacements = {
        "\u00ad": "", "\u2010": "-", "\u2011": "-", "\u2012": "-",
        "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\xa0": " ",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text


def repair_pdf_line_breaks(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove hyphenation at line break: "neural net-\nwork" -> "neural network"
    text = re.sub(r"([A-Za-zÀ-ÿ])-\n([A-Za-zÀ-ÿ])", r"\1\2", text)
    # Merge wrapped lines inside paragraphs, but keep likely headings/lists apart.
    lines = text.split("\n")
    out = []
    for line in lines:
        s = line.rstrip()
        if not out:
            out.append(s)
            continue
        prev = out[-1].rstrip()
        if not prev or not s:
            out.append(s)
            continue
        if is_likely_heading(s) or is_likely_heading(prev) or is_toc_row(s) or re.match(r"^\s*[•\-*]\s+", s):
            out.append(s)
        elif re.search(r"[a-zà-ÿ,;:]$", prev) and re.match(r"^[a-zà-ÿ(]", s):
            out[-1] = prev + " " + s.strip()
        else:
            out.append(s)
    return "\n".join(out)


def remove_pdf_artifacts(text: str) -> str:
    # Page markers such as "1 | Page", "Graduation Thesis Name" is handled softly as repeated headers.
    text = re.sub(r"^\s*\d+\s*(?:\|\s*)?(?:page|pagina|p\s*a\s*g\s*e)\s*$", "", text, flags=re.I | re.M)
    text = re.sub(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", "", text, flags=re.I | re.M)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def normalize_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    sample = (text or "")[:10000]
    low = sample.lower()
    nl_score = sum(low.count(x) for x in DUTCH_CUES)
    en_score = sum(low.count(x) for x in ENGLISH_CUES)
    if _langdetect:
        try:
            lang = _langdetect(sample)
            if lang in {"nl", "en"}:
                return lang
        except Exception:
            pass
    if nl_score > en_score:
        return "nl"
    if en_score > nl_score:
        return "en"
    return "unknown"


def is_toc_row(line: str) -> bool:
    s = clean_line(line)
    if not s:
        return False
    # "2.1 Heading .......... 4" or "2.1 Heading 4"
    if re.search(r"\.{3,}\s*\d{1,4}\s*$", s):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+.{3,90}\s+\d{1,4}$", s) and len(s.split()) <= 12:
        return True
    if re.match(r"^(appendix|bijlage)\s+\d{1,4}\b", s, flags=re.I) and len(s.split()) <= 12:
        return True
    return False


def is_likely_heading(line: str) -> bool:
    s = clean_line(line)
    if not s or len(s) > 160:
        return False
    if is_toc_row(s):
        return False
    low = s.lower().strip(":- ")
    if low in TOC_LABELS or low in REFERENCE_HEADINGS or low in APPENDIX_HEADINGS:
        return True
    if re.match(r"^(chapter|hoofdstuk)\s+\d+(?:\s*[-:]\s*.+)?$", s, flags=re.I):
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+[A-ZÀ-Ý][^\n]{2,120}$", s):
        # avoid sentences like "2 patients were..."
        if not re.search(r"[.!?]\s*$", s):
            return True
    return False


def heading_id(line: str) -> str:
    s = clean_line(line)
    m = re.match(r"^(?:chapter|hoofdstuk)?\s*(\d+(?:\.\d+)*)\b", s, flags=re.I)
    return m.group(1) if m else ""


def find_toc_block(lines: list[str]) -> tuple[int | None, int | None]:
    start = None
    for i, line in enumerate(lines[:250]):
        if clean_line(line).lower().strip(":") in TOC_LABELS:
            start = i
            break
    if start is None:
        return None, None
    end = start + 1
    toc_rows = 0
    for j in range(start + 1, min(len(lines), start + 250)):
        line = clean_line(lines[j])
        if not line:
            if toc_rows >= 3:
                # keep looking because TOCs often have empty lines
                end = j + 1
            continue
        if is_toc_row(line) or re.match(r"^\d+(?:\.\d+)*\s+[A-Za-zÀ-ÿ]", line):
            toc_rows += 1
            end = j + 1
            continue
        # Stop when actual prose begins after enough TOC rows.
        if toc_rows >= 3 and len(line.split()) > 12:
            break
        if toc_rows >= 3 and is_likely_heading(line) and not is_toc_row(line):
            break
    return start, end


def line_has_prose_after(lines: list[str], idx: int, window: int = 8) -> bool:
    text = " ".join(clean_line(x) for x in lines[idx+1:idx+1+window] if clean_line(x))
    # Real section start is followed by prose, not only more TOC rows.
    if len(text.split()) < 35:
        return False
    toc_count = sum(1 for x in lines[idx+1:idx+1+window] if is_toc_row(x))
    return toc_count <= 1


def find_body_start(lines: list[str], toc_end: int | None = None) -> int:
    search_from = toc_end or 0
    candidates = []
    for i in range(search_from, len(lines)):
        line = clean_line(lines[i])
        if not line or is_toc_row(line):
            continue
        low = line.lower()
        is_start_heading = (
            re.match(r"^(?:1|1\.0)\s+(introduction|inleiding)\b", low)
            or re.match(r"^(chapter|hoofdstuk)\s+1\b", low)
            or low in {"introduction", "inleiding"}
            or re.match(r"^1\s+[A-ZÀ-Ý][^\n]{2,80}$", line)
        )
        if is_start_heading and line_has_prose_after(lines, i):
            return i
        if is_likely_heading(line) and line_has_prose_after(lines, i):
            candidates.append(i)
    if candidates:
        return candidates[0]
    # fallback: after abstract/preface/front matter, first long paragraph
    for i in range(search_from, len(lines)):
        if len(clean_line(lines[i]).split()) > 45:
            return i
    return 0


def split_document_parts(cleaned: str) -> dict:
    lines = cleaned.splitlines()
    toc_start, toc_end = find_toc_block(lines)
    body_start = find_body_start(lines, toc_end)
    titlepage_text = "\n".join(lines[:body_start]).strip()
    toc_text = "\n".join(lines[toc_start:toc_end]).strip() if toc_start is not None and toc_end is not None else ""

    # Find real references/appendix after body start. Avoid compact appendix rows in TOC/frontmatter.
    ref_idx = app_idx = None
    for i in range(body_start, len(lines)):
        s = clean_line(lines[i])
        low = s.lower().strip(":- ")
        if ref_idx is None and low in REFERENCE_HEADINGS and line_has_prose_after(lines, i, window=12):
            ref_idx = i
        if app_idx is None and low in APPENDIX_HEADINGS and line_has_prose_after(lines, i, window=12):
            app_idx = i
        if ref_idx is not None and app_idx is not None:
            break

    cut_points = [x for x in [ref_idx, app_idx] if x is not None]
    main_end = min(cut_points) if cut_points else len(lines)
    main_text = "\n".join(lines[body_start:main_end]).strip()

    references_text = ""
    appendix_text = ""
    if ref_idx is not None:
        ref_end = app_idx if app_idx is not None and app_idx > ref_idx else len(lines)
        references_text = "\n".join(lines[ref_idx:ref_end]).strip()
    if app_idx is not None:
        appendix_text = "\n".join(lines[app_idx:]).strip()

    return {
        "titlepage_text": titlepage_text,
        "table_of_contents": toc_text,
        "main_text": main_text,
        "references_text": references_text,
        "appendix_text": appendix_text,
    }


def extract_titlepage_candidates(titlepage_text: str) -> dict:
    lines = [clean_line(x) for x in titlepage_text.splitlines() if clean_line(x)]
    # Use only early front matter, stop at preface/abstract/contents.
    early = []
    for line in lines[:80]:
        low = line.lower().strip(":- ")
        if low in {"preface","voorwoord","abstract","samenvatting","contents","inhoudsopgave","table of contents"}:
            break
        early.append(line)

    date_re = rf"\b(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{4}}[/-]\d{{1,2}}[/-]\d{{1,2}}|\d{{1,2}}\s+(?:{MONTHS})(?:st|nd|rd|th)?[,]?\s+\d{{4}}|(?:{MONTHS})\s+\d{{4}})\b"
    dates = []
    for line in lines[:120]:
        for m in re.finditer(date_re, line, flags=re.I):
            dates.append(m.group(0))

    org_markers = r"\b(university|universiteit|hogeschool|applied sciences|erasmus|department|faculty|school|college|institute|instituut|province|provincie|gemeente|municipality|ministry|ministerie|bv|b\.v\.|ltd|inc|foundation|stichting)\b"
    orgs = [x for x in early if re.search(org_markers, x, flags=re.I) and len(x) <= 160]

    def looks_person(line):
        if re.search(org_markers, line, re.I) or re.search(date_re, line, re.I) or any(ch.isdigit() for ch in line):
            return False
        words=line.split()
        if not 2 <= len(words) <= 5:
            return False
        caps=sum(1 for w in words if re.match(r"^[A-ZÀ-Ý][A-Za-zÀ-ÿ'.-]+$", w) or w.lower() in {"van","de","den","der","ten","ter","von"})
        return caps >= len(words)-1

    authors = [x for x in early if looks_person(x)]

    # Title is early non-date/non-org/non-author line(s), before author/date.
    title_lines = []
    for line in early[:12]:
        if re.search(date_re, line, re.I) or re.search(org_markers, line, re.I) or looks_person(line):
            continue
        if re.search(r"\b(submitted|supervised|prepared|opgesteld|begeleid)\b", line, re.I):
            continue
        if 4 <= len(line) <= 180:
            title_lines.append(line)
    return {
        "title_candidates": title_lines[:4],
        "author_candidates": authors[:8],
        "organization_candidates": orgs[:8],
        "date_candidates": list(dict.fromkeys(dates))[:8],
    }


def detect_sections(main_text: str) -> list[dict]:
    lines = main_text.splitlines()
    headings = []
    for i,line in enumerate(lines):
        if is_likely_heading(line):
            hid = heading_id(line) or f"section_{len(headings)+1}"
            headings.append((i,hid,clean_line(line)))
    if not headings:
        return [{"section_id":"main","heading":"Main text","start_char":0,"end_char":len(main_text),"word_count":len(main_text.split()),"text":main_text}]
    # char offsets by line starts
    offsets=[]; pos=0
    for l in lines:
        offsets.append(pos); pos += len(l)+1
    sections=[]
    for idx,(line_i,hid,head) in enumerate(headings):
        start = offsets[line_i]
        end_line = headings[idx+1][0] if idx+1 < len(headings) else len(lines)
        end = offsets[end_line] if end_line < len(offsets) else len(main_text)
        text = "\n".join(lines[line_i:end_line]).strip()
        if len(text.split()) >= 8:
            sections.append({"section_id":hid,"heading":head,"start_char":start,"end_char":end,"word_count":len(text.split()),"text":text})
    return sections or [{"section_id":"main","heading":"Main text","start_char":0,"end_char":len(main_text),"word_count":len(main_text.split()),"text":main_text}]


def split_words_with_overlap(words: list[str], target: int, max_words: int, overlap: int) -> list[list[str]]:
    chunks=[]; i=0
    while i < len(words):
        end=min(i+max_words, len(words))
        # Try to end near target on sentence boundary.
        window_end=min(i+target, len(words))
        if window_end < len(words):
            end=window_end
            for j in range(min(i+max_words,len(words))-1, max(i+target-200,i), -1):
                if re.search(r"[.!?]$", words[j]):
                    end=j+1; break
        chunks.append(words[i:end])
        if end >= len(words): break
        i=max(end-overlap, i+1)
    return chunks


def build_chunks(sections: list[dict], target_words: int = TARGET_CHUNK_WORDS, max_words: int = MAX_CHUNK_WORDS, min_words: int = MIN_CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> list[dict]:
    chunks=[]
    for sec in sections:
        words=sec.get("text","").split()
        if len(words) < min_words and chunks:
            # merge tiny section into previous when possible
            if chunks[-1]["word_count"] + len(words) <= max_words:
                chunks[-1]["text"] += "\n\n" + sec.get("text","")
                chunks[-1]["word_count"] = len(chunks[-1]["text"].split())
                continue
        parts = split_words_with_overlap(words, target_words, max_words, overlap_words) if len(words) > max_words else [words]
        for part_idx, part_words in enumerate(parts, start=1):
            if len(part_words) < 20:
                continue
            chunks.append({
                "chunk_id": f"chunk_{len(chunks)+1:03d}",
                "source_section_id": sec.get("section_id"),
                "source_section_heading": sec.get("heading"),
                "section_part": part_idx,
                "chunk_type": "main_text",
                "word_count": len(part_words),
                "text": " ".join(part_words),
            })
    return chunks


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_text(raw_text: str, document_id: str, original_file: str = "", data_dir: str | Path = "Data") -> dict:
    data_dir = Path(data_dir)
    silver_dir = data_dir / "silver"
    silver_dir.mkdir(parents=True, exist_ok=True)

    cleaned = normalize_spacing(remove_pdf_artifacts(repair_pdf_line_breaks(normalize_unicode(raw_text))))
    parts = split_document_parts(cleaned)
    # Protect against bad split: if main text is too small and appendix huge, recover by re-running body start later.
    if len(parts["main_text"].split()) < 400 and len(parts["appendix_text"].split()) > len(parts["main_text"].split()) * 3:
        # Some PDFs have appendix overview before real body. Treat appendix_text as body candidate until real reference/appendix later.
        candidate = parts["appendix_text"]
        lines = candidate.splitlines()
        body_start = find_body_start(lines, None)
        rebuilt = normalize_spacing("\n".join(lines[body_start:]))
        parts2 = split_document_parts(rebuilt)
        if len(parts2["main_text"].split()) > len(parts["main_text"].split()):
            parts["main_text"] = parts2["main_text"]
            parts["references_text"] = parts2["references_text"] or parts["references_text"]
            parts["appendix_text"] = parts2["appendix_text"]

    sections = detect_sections(parts["main_text"])
    chunks = build_chunks(sections)
    lang = detect_language(parts["main_text"] or cleaned)
    titlepage_candidates = extract_titlepage_candidates(parts["titlepage_text"])

    output = {
        "document_id": document_id,
        "original_file": original_file,
        "detected_language": lang,
        "language": lang,
        "processing_layer": "silver",
        "processing_version": "silver_local_generic_streamlit_v1_from_notebook",
        "created_at": datetime.now().isoformat(),
        "statistics": {
            "raw_characters": len(raw_text or ""),
            "cleaned_characters": len(cleaned),
            "main_text_characters": len(parts["main_text"]),
            "main_text_words": len(parts["main_text"].split()),
            "titlepage_words": len(parts["titlepage_text"].split()),
            "toc_words": len(parts["table_of_contents"].split()),
            "references_words": len(parts["references_text"].split()),
            "appendix_words": len(parts["appendix_text"].split()),
            "section_count": len(sections),
            "chunk_count": len(chunks),
            "target_chunk_words": target_words if (target_words:=TARGET_CHUNK_WORDS) else TARGET_CHUNK_WORDS,
            "max_chunk_words": MAX_CHUNK_WORDS,
            "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        },
        "titlepage_candidates": titlepage_candidates,
        "document_parts": parts,
        "sections": sections,
        "chunks": chunks,
        "quality": {
            "is_empty": len(parts["main_text"].split()) == 0,
            "word_count": len(parts["main_text"].split()),
            "character_count": len(parts["main_text"]),
            "chunk_count": len(chunks),
            "average_chunk_words": round(sum(c["word_count"] for c in chunks)/max(1,len(chunks)),2),
            "ready_for_modeling": len(parts["main_text"].split()) > 200 and len(chunks) > 0,
        }
    }

    (silver_dir / f"{document_id}_clean_main_text.txt").write_text(parts["main_text"], encoding="utf-8")
    (silver_dir / f"{document_id}_sections.json").write_text(json.dumps({"document_id": document_id, "sections": sections}, indent=4, ensure_ascii=False), encoding="utf-8")
    write_jsonl(chunks, silver_dir / f"{document_id}_chunks.jsonl")
    (silver_dir / f"{document_id}_silver.json").write_text(json.dumps(output, indent=4, ensure_ascii=False), encoding="utf-8")
    # Compatibility names for older app code.
    (silver_dir / f"{document_id}_clean.txt").write_text(parts["main_text"], encoding="utf-8")
    return output


def process_bronze_file(document_id: str, data_dir: str | Path = "Data") -> dict:
    bronze_path = Path(data_dir) / "bronze" / f"{document_id}.txt"
    if not bronze_path.exists():
        raise FileNotFoundError(f"Missing bronze text file: {bronze_path}")
    return process_text(bronze_path.read_text(encoding="utf-8"), document_id, original_file=bronze_path.name, data_dir=data_dir)


def run_silver_layer(document_ids: list[str] | None = None, data_dir: str | Path = "Data") -> list[dict]:
    bronze_dir = Path(data_dir) / "bronze"
    files = sorted(bronze_dir.glob("*.txt"))
    if document_ids:
        wanted=set(document_ids)
        files=[p for p in files if p.stem in wanted]
    return [process_text(p.read_text(encoding="utf-8"), p.stem, original_file=p.name, data_dir=data_dir) for p in files]
