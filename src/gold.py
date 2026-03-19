import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


GOLD_FOLDER = Path("Data/gold/MBART")
GOLD_FOLDER.mkdir(parents=True, exist_ok=True)

TEXT_COLUMN = "cleaned_text"
RAW_TEXT_COLUMN = "raw_text"
DOC_ID_COLUMN = "document_id"
SENTENCES_COLUMN = "sentences"

MBART_MODEL = "facebook/mbart-large-50"
SRC_LANG = "en_XX"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 170
MIN_OUTPUT_LEN = 110

MAX_SIMILARITY = 0.68
MAX_SELECTED_SENTENCES = 12
MAX_KEY_ENTITIES = 15
MAX_TOP_TERMS = 10

RESEARCH_GROUPS = [
    "Aquaculture in Delta Areas",
    "Assetmanagement",
    "Biobased Bouwen",
    "Building with Nature",
    "Data Science",
    "Delta Power",
    "Excellence and Innovation in Education",
    "HZ Kenniscentrum Kusttoerisme",
    "HZ Kenniscentrum Ondernemen en Innoiveren",
    "HZ Kenniscentrum Zeeuwse Samenleving",
    "HZ Nexus",
    "Healthy Region",
    "Kunst Cultuur Transitie",
    "Marine Biobased Chemie",
    "Ouderenzorg",
    "Resilient Deltas",
    "Supply Chain Innovation",
    "Water Technology",
]

SECTION_LABELS = {
    "abstract", "introduction", "background", "discussion", "conclusion",
    "conclusions", "results", "result", "method", "methods", "methodology",
    "references", "bibliography", "appendix", "summary", "contents",
    "table of contents"
}

INSTITUTION_PATTERNS = [
    r"\buniversity\b",
    r"\bcollege\b",
    r"\bfaculty\b",
    r"\bschool\b",
    r"\binstitute\b",
    r"\bdepartment\b",
]

MONTH_WORDS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)

DATE_PATTERN = (
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b\d{4}-\d{2}-\d{2}\b|"
    rf"\b\d{{1,2}}\s+(?:{MONTH_WORDS})\s+\d{{2,4}}\b|"
    rf"\b(?:{MONTH_WORDS})\s+\d{{1,2}},?\s+\d{{2,4}}\b|"
    rf"\b(?:{MONTH_WORDS})\s+\d{{4}}\b|"
    rf"\b\d{{4}}\s+(?:{MONTH_WORDS})\b|"
    rf"\b\d{{1,2}}\s+(?:{MONTH_WORDS})\b|"
    rf"\b(?:{MONTH_WORDS})\s+\d{{1,2}}\b"
)

AUTHOR_PREFIX_PATTERNS = [
    r"^(author|authors|written by|student|students|name|contributors?)\s*[:\-]?\s+",
    r"^by\s+"
]


# -----------------------
# MODEL LOADING
# -----------------------

def load_gold_models() -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        MBART_MODEL,
        src_lang=SRC_LANG,
        use_fast=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(MBART_MODEL).to(device)
    model.eval()

    nlp = spacy.load("en_core_web_sm")

    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model=ZERO_SHOT_MODEL,
        device=0 if torch.cuda.is_available() else -1
    )

    return {
        "device": device,
        "tokenizer": tokenizer,
        "model": model,
        "nlp": nlp,
        "zero_shot_classifier": zero_shot_classifier,
    }


# -----------------------
# BASIC HELPERS
# -----------------------

def safe_string(value: Any) -> str:
    return "" if value is None else str(value)


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", safe_string(text)).strip()


def simple_tokenize(text: Any) -> List[str]:
    return re.findall(r"\b\w+\b", safe_string(text).lower())


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        clean = normalize_whitespace(item)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def clean_text(text: str) -> str:
    text = safe_string(text)
    text = text.replace("\x00", " ").replace("\ufeff", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []
    for line in text.splitlines():
        line_strip = normalize_whitespace(line)

        if re.fullmatch(r"\d{1,4}", line_strip):
            continue
        if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", line_strip, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d+\s*\|\s*p\s*a\s*g\s*e", line_strip, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d+\s*\|\s*page", line_strip, flags=re.IGNORECASE):
            continue

        cleaned_lines.append(line.strip())

    return "\n".join([x for x in cleaned_lines if x]).strip()


def split_sentences(text: str, nlp) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    doc = nlp(text)
    sentences = [normalize_whitespace(sent.text) for sent in doc.sents]
    return [s for s in sentences if s]


def line_looks_like_author_affiliation(line: str) -> bool:
    s = normalize_whitespace(line)
    lower = s.lower()

    if "@" in s:
        return True
    if re.search(r"\b(university|faculty|department|institute|school|college)\b", lower):
        return True
    if re.search(r"\b[a-z]+\s*,\s*[A-Z][a-z]+", s):
        return True
    return False


def is_title_label_line(s: str) -> bool:
    s = normalize_whitespace(s).lower()
    return s in {
        "research paper",
        "article",
        "original article",
        "review article",
        "case study",
        "conference paper"
    }


def split_title_author_affiliation_line(line: str) -> List[str]:
    s = normalize_whitespace(line)
    if not s:
        return []

    if "@" in s:
        s = s.split("@")[0].strip()

    s = re.sub(r"_+", " ", s)
    s = normalize_whitespace(s)

    m = re.match(
        r"^(?P<title>.+?)\s+"
        r"(?P<author>[A-Z][a-z]+(?:\s+(?:de|den|der|van|von|ten|ter|op|la|le|du|di))?(?:\s+[A-Z][a-z]+)+)\s+"
        r"(?P<affil>(?:University|Faculty|Department|Institute|School|College)\b.+)$",
        s
    )

    if m:
        parts = []
        title = normalize_whitespace(m.group("title"))
        author = normalize_whitespace(m.group("author"))
        affil = normalize_whitespace(m.group("affil"))

        if title:
            parts.append(title)
        if author:
            parts.append(author)
        if affil:
            parts.append(affil)
        return parts

    return [s]


def clean_person_entity(candidate: str) -> str:
    candidate = normalize_whitespace(candidate)

    candidate = re.sub(
        r"\b(university|faculty|department|institute|school|college)\b.*$",
        "",
        candidate,
        flags=re.IGNORECASE
    )
    candidate = re.sub(
        r"\b(poland|netherlands|germany|france|spain|italy)\b.*$",
        "",
        candidate,
        flags=re.IGNORECASE
    )
    candidate = normalize_whitespace(candidate.strip(",;:- "))

    parts = candidate.split()
    if len(parts) > 3:
        candidate = " ".join(parts[:3])

    return candidate


def is_valid_person_name(name: str) -> bool:
    name = normalize_whitespace(name)
    if not name:
        return False

    parts = name.split()
    if len(parts) < 2 or len(parts) > 5:
        return False

    if any(ch.isdigit() for ch in name):
        return False
    if re.search(r"[•()\[\]{}:;\\/|_+=<>@]", name):
        return False

    particles = {"de", "den", "der", "van", "von", "ten", "ter", "op", "la", "le", "du", "di"}

    def is_cap_name(tok: str) -> bool:
        return bool(re.fullmatch(r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?", tok))

    def is_upper_name(tok: str) -> bool:
        return bool(re.fullmatch(r"[A-Z]{2,}", tok))

    def is_name_token(tok: str, i: int) -> bool:
        if i > 0 and tok.lower() in particles:
            return True
        return is_cap_name(tok) or is_upper_name(tok)

    if not all(is_name_token(tok, i) for i, tok in enumerate(parts)):
        return False

    core_tokens = [tok for i, tok in enumerate(parts) if not (i > 0 and tok.lower() in particles)]
    if len(core_tokens) < 2:
        return False

    bad_phrases = {
        "research paper",
        "eurocall review",
        "advanced learners",
        "mobile devices",
        "english language",
        "learning english",
        "data driven",
        "applied sciences"
    }

    if name.lower() in bad_phrases:
        return False

    alpha = [ch for ch in name if ch.isalpha()]
    if alpha:
        uppercase_ratio = sum(ch.isupper() for ch in alpha) / len(alpha)
        if uppercase_ratio > 0.85 and len(parts) >= 2:
            return False

    return True


def looks_like_author_list(line: str) -> bool:
    s = normalize_whitespace(line)
    if not s or "," not in s:
        return False

    parts = [normalize_whitespace(p) for p in s.split(",")]
    valid_parts = [p for p in parts if is_valid_person_name(clean_person_entity(p))]
    return len(valid_parts) >= 2


def split_collapsed_front_blob(line: str) -> List[str]:
    s = normalize_whitespace(line)
    if not s:
        return []

    if "@" in s:
        s = s.split("@")[0].strip()

    parts: List[str] = []

    code_match = re.search(r"\([A-Z]{2,6}\d{3,}[A-Z0-9]*\)", s)
    if code_match:
        left = normalize_whitespace(s[:code_match.end()])
        right = normalize_whitespace(s[code_match.end():])
        if left:
            parts.append(left)
        if right:
            parts.append(right)
    else:
        parts.append(s)

    expanded: List[str] = []

    for part in parts:
        part = normalize_whitespace(part)
        if not part:
            continue

        caps_matches = list(
            re.finditer(r"\b(?:[A-Z][A-Z&/\-]*)(?:\s+[A-Z][A-Z&/\-]*){1,8}\b", part)
        )

        if caps_matches:
            cursor = 0
            for m in caps_matches:
                before = normalize_whitespace(part[cursor:m.start()])
                caps = normalize_whitespace(m.group(0))
                if before:
                    expanded.append(before)
                if caps:
                    expanded.append(caps)
                cursor = m.end()

            after = normalize_whitespace(part[cursor:])
            if after:
                expanded.append(after)
        else:
            expanded.append(part)

    final_parts: List[str] = []

    for part in expanded:
        part = normalize_whitespace(part)
        if not part:
            continue

        name_tail = re.search(
            r"([A-Z][a-z]+(?:\s+(?:de|den|der|van|von|ten|ter|op|la|le|du|di))?(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+(?:de|den|der|van|von|ten|ter|op|la|le|du|di))?(?:\s+[A-Z][a-z]+)+)*)$",
            part
        )
        if name_tail:
            left = normalize_whitespace(part[:name_tail.start()])
            right = normalize_whitespace(name_tail.group(1))
            if left:
                final_parts.append(left)
            if right:
                final_parts.append(right)
        else:
            final_parts.append(part)

    return [normalize_whitespace(x) for x in final_parts if normalize_whitespace(x)]


def extract_top_terms(
    text: str,
    nlp,
    top_n: int = 10,
    allowed_pos: set = {"NOUN", "PROPN", "ADJ"}
) -> List[str]:
    if not text:
        return []

    doc = nlp(text)
    tokens = []

    for token in doc:
        if token.is_stop:
            continue
        if token.is_punct:
            continue
        if token.is_space:
            continue
        if token.like_num:
            continue
        if len(token.text) < 3:
            continue
        if token.pos_ not in allowed_pos:
            continue

        lemma = token.lemma_.lower().strip()
        if not lemma.isalpha():
            continue

        tokens.append(lemma)

    if not tokens:
        return []

    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_n)]


def looks_like_non_person_phrase(s: str) -> bool:
    s = normalize_whitespace(s)
    if not s:
        return True

    if len(s.split()) > 4:
        return True
    if re.match(r"^\d+(\.\d+)*\b", s):
        return True
    if re.match(r"^[•\-–]", s):
        return True
    if "." * 5 in s:
        return True
    if ":" in s:
        return True

    lower = s.lower()
    if any(term in lower for term in ["report", "assignment", "thesis", "paper"]):
        return True

    return False


def looks_like_section_or_topic(s: str) -> bool:
    s = normalize_whitespace(s)
    if not s:
        return True

    if len(s.split()) > 4:
        return True
    if re.match(r"^\d+(\.\d+)*\b", s):
        return True
    if re.match(r"^[•\-–]", s):
        return True
    if "." * 5 in s:
        return True
    if ":" in s:
        return True

    return False


def extract_name_like_spans(text: str) -> List[str]:
    s = normalize_whitespace(text)
    if not s:
        return []

    pattern = re.compile(
        r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:de|den|der|van|von|ten|ter|op|la|le|du|di|[A-Z][a-z]+|[A-Z]{2,})){1,3}\b"
    )

    found = []

    for m in pattern.finditer(s):
        span = normalize_whitespace(m.group(0))
        tokens = span.split()

        found.append(span)

        for size in (2, 3):
            if len(tokens) >= size:
                for i in range(len(tokens) - size + 1):
                    sub = " ".join(tokens[i:i + size])
                    found.append(sub)

    return dedupe_keep_order(found)


# -----------------------
# SENTENCE SELECTION
# -----------------------

def remove_section_prefix(sentence: str) -> str:
    s = normalize_whitespace(sentence)
    s = re.sub(
        r"^(abstract|introduction|background|discussion|conclusion|conclusions|results|result|method|methods|methodology|summary)\b[:\-\s]*",
        "",
        s,
        flags=re.IGNORECASE
    )
    return normalize_whitespace(s)


def is_table_of_contents_line(line: str) -> bool:
    s = normalize_whitespace(line)

    if "table of contents" in s.lower():
        return True
    if re.search(r"\.{4,}\s*\d+$", s):
        return True
    if re.search(r"\.{4,}", s):
        return True
    if re.match(r"^\d+(\.\d+)*\s+.+\s+\d+$", s.lower()):
        return True
    if re.search(r"\b\d+\.\d+\b", s) and len(s) < 120:
        return True

    return False


def looks_like_reference(sentence: str) -> bool:
    s = normalize_whitespace(sentence)

    if re.search(r"\bet al\.", s, flags=re.IGNORECASE):
        return True
    if re.search(r"\bdoi\b", s, flags=re.IGNORECASE):
        return True
    if re.search(r"https?://|www\.", s):
        return True
    if re.search(r"\breferences\b|\bbibliography\b", s, flags=re.IGNORECASE):
        return True
    if re.search(r"\(\s*[A-Z][A-Za-z].*\d{4}.*\)", s):
        return True

    return False


def is_heading_like(sentence: str) -> bool:
    s = normalize_whitespace(sentence)

    if not s:
        return True
    if s.lower() in SECTION_LABELS:
        return True
    if len(s.split()) <= 5 and s.istitle():
        return True
    if re.fullmatch(r"[A-Z][A-Za-z\s/&\-]{1,60}", s) and len(s.split()) <= 8:
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Za-z]", s):
        return True

    return False


def is_front_matter_blob(sentence: str) -> bool:
    s = normalize_whitespace(sentence)
    lower = s.lower()

    if len(s) < 40:
        return False

    score = 0
    if "contents" in lower:
        score += 2
    if "university" in lower or "faculty" in lower or "institute" in lower or "school" in lower:
        score += 1
    if "assignment" in lower or "research report" in lower or "thesis" in lower:
        score += 1
    if re.search(r"\b\d{1,2}\s+(?:%s)\b" % MONTH_WORDS, lower, flags=re.IGNORECASE):
        score += 1

    return score >= 3


def clean_sentence_for_selection(sentence: str) -> str:
    s = remove_section_prefix(sentence)
    s = re.sub(r"\s+", " ", s).strip()

    if is_front_matter_blob(s):
        return ""

    return s


def is_good_sentence(sentence: str) -> bool:
    s = clean_sentence_for_selection(sentence)

    if len(s) < 45:
        return False
    if len(s.split()) < 8:
        return False
    if is_table_of_contents_line(s):
        return False
    if looks_like_reference(s):
        return False
    if is_heading_like(s):
        return False
    if re.match(r"^\d+(\.\d+)*\s*$", s):
        return False
    if "................................................................" in s:
        return False
    if "•" in s and len(s.split()) < 15:
        return False

    return True


def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(simple_tokenize(a))
    set_b = set(simple_tokenize(b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def sentence_position_score(index: int, total: int) -> float:
    if total <= 0:
        return 0.0

    relative = index / max(total, 1)

    if relative < 0.10:
        return 0.9
    if relative < 0.25:
        return 0.5
    if relative > 0.90:
        return 0.4

    return 0.0


def sentence_information_score(sentence: str) -> float:
    s = sentence.lower()
    word_count = len(sentence.split())
    score = 0.0

    if 12 <= word_count <= 35:
        score += 1.0
    elif 36 <= word_count <= 50:
        score += 0.6
    elif word_count > 55:
        score -= 0.5

    discourse_terms = [
        "this study", "this report", "this paper", "the aim", "the purpose",
        "the objective", "the analysis", "the findings", "the results",
        "shows that", "demonstrates that", "indicates that", "suggests that",
        "because", "therefore", "however", "overall", "in conclusion"
    ]
    for term in discourse_terms:
        if term in s:
            score += 0.5

    tokens = simple_tokenize(sentence)
    if tokens:
        unique_ratio = len(set(tokens)) / len(tokens)
        score += 0.8 * unique_ratio

    vague_phrases = [
        "this is needed", "it also makes", "it means that", "there are a lot of"
    ]
    for phrase in vague_phrases:
        if phrase in s:
            score -= 0.5

    return score


def classify_sentence_role(sentence: str) -> str:
    s = sentence.lower()

    if any(x in s for x in [
        "this study", "this report", "this paper", "the aim", "the purpose",
        "we examine", "we explore", "we investigate", "focus on", "focuses on"
    ]):
        return "objective"

    if any(x in s for x in [
        "method", "approach", "analysis", "using", "based on",
        "evaluate", "evaluates", "assessment", "assesses", "analyzes", "examines"
    ]):
        return "method"

    if any(x in s for x in [
        "results", "findings", "shows that", "demonstrates that",
        "indicates that", "reveals that", "suggests that", "is reduced"
    ]):
        return "findings"

    if any(x in s for x in [
        "recommend", "should", "could", "future work", "implementation",
        "strategy", "plan", "timeline", "metrics for success"
    ]):
        return "recommendation"

    if any(x in s for x in [
        "in conclusion", "overall", "to conclude", "in summary"
    ]):
        return "conclusion"

    return "background"


def rank_sentences(sentences: List[str]) -> List[Tuple[str, float]]:
    cleaned_candidates = []
    for i, s in enumerate(sentences):
        cleaned = clean_sentence_for_selection(s)
        if not cleaned:
            continue
        if is_good_sentence(cleaned):
            cleaned_candidates.append((i, cleaned))

    total = len(cleaned_candidates)
    scored = []

    for i, sentence in cleaned_candidates:
        score = sentence_information_score(sentence) + sentence_position_score(i, total)
        scored.append((sentence, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def select_balanced_sentences(
    sentences: List[str],
    tokenizer,
    max_sentences: int = MAX_SELECTED_SENTENCES,
    max_similarity: float = MAX_SIMILARITY,
    max_input_tokens: int = MAX_INPUT_LEN
) -> List[str]:
    ranked = rank_sentences(sentences)

    buckets = {
        "objective": [],
        "method": [],
        "findings": [],
        "recommendation": [],
        "conclusion": [],
        "background": []
    }

    for sent, _ in ranked:
        buckets[classify_sentence_role(sent)].append(sent)

    selected = []
    target_plan = [
        ("objective", 2),
        ("method", 2),
        ("findings", 2),
        ("recommendation", 2),
        ("conclusion", 1),
        ("background", 3),
    ]

    for role, limit in target_plan:
        count = 0
        for sent in buckets[role]:
            if any(jaccard_similarity(sent, prev) > max_similarity for prev in selected):
                continue

            candidate_text = " ".join(selected + [sent])
            token_count = len(tokenizer(candidate_text, truncation=False)["input_ids"])
            if token_count > max_input_tokens - 30:
                continue

            selected.append(sent)
            count += 1
            if count >= limit or len(selected) >= max_sentences:
                break

    if len(selected) < max_sentences:
        for sent, _ in ranked:
            if sent in selected:
                continue
            if any(jaccard_similarity(sent, prev) > max_similarity for prev in selected):
                continue

            candidate_text = " ".join(selected + [sent])
            token_count = len(tokenizer(candidate_text, truncation=False)["input_ids"])
            if token_count > max_input_tokens - 30:
                continue

            selected.append(sent)
            if len(selected) >= max_sentences:
                break

    ordered = []
    cleaned_source = [clean_sentence_for_selection(s) for s in sentences]
    selected_set = set(selected)

    for s in cleaned_source:
        if s in selected_set and s not in ordered:
            ordered.append(s)

    return ordered[:max_sentences]


def build_summary_input(selected_sentences: List[str]) -> str:
    cleaned = []

    for s in selected_sentences:
        s = clean_sentence_for_selection(s)
        if is_heading_like(s):
            continue
        cleaned.append(s)

    text = " ".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------
# DATE HELPERS
# -----------------------

def parse_two_digit_year(year_2d: int) -> int:
    if 0 <= year_2d <= 30:
        return 2000 + year_2d
    return 1900 + year_2d


def is_date_like_line(line: str) -> bool:
    s = normalize_whitespace(line)

    if not s:
        return False

    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s):
        return True
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*[-–]\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return True
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}\s*[-–]\s*\d{4}-\d{2}-\d{2}", s):
        return True
    if re.fullmatch(rf"\d{{1,2}}\s+(?:{MONTH_WORDS})(?:\s+\d{{2,4}})?", s, flags=re.IGNORECASE):
        return True
    if re.fullmatch(rf"(?:{MONTH_WORDS})\s+\d{{1,2}}(?:,?\s+\d{{2,4}})?", s, flags=re.IGNORECASE):
        return True
    if re.fullmatch(rf"(?:{MONTH_WORDS})\s+\d{{4}}", s, flags=re.IGNORECASE):
        return True
    if re.fullmatch(rf"\d{{4}}\s+(?:{MONTH_WORDS})", s, flags=re.IGNORECASE):
        return True

    return False


def extract_dates(text: str) -> List[str]:
    text = safe_string(text)
    found = []

    for match in re.findall(DATE_PATTERN, text, flags=re.IGNORECASE):
        clean = normalize_whitespace(match)
        if clean:
            found.append(clean)

    return dedupe_keep_order(found)


def extract_dates_in_order(text: str) -> List[str]:
    return extract_dates(text)


def extract_start_end_dates(text: str) -> Tuple[str, str, List[str]]:
    front = extract_front_matter(text)
    front_lines = get_front_lines(text, max_chars=5000, max_lines=20)

    found = extract_dates_in_order(front)

    joined_candidates = []
    for i in range(len(front_lines) - 1):
        a = normalize_whitespace(front_lines[i])
        b = normalize_whitespace(front_lines[i + 1])
        joined = normalize_whitespace(f"{a} {b}")
        joined_candidates.append(joined)

    for candidate in joined_candidates:
        for match in re.findall(DATE_PATTERN, candidate, flags=re.IGNORECASE):
            clean = normalize_whitespace(match)
            if clean:
                found.append(clean)

    found = dedupe_keep_order(found)

    if not found:
        found = extract_dates_in_order(text[:3000])

    if not found:
        return "", "", []

    if len(found) == 1:
        return found[0], "", found

    return found[0], found[-1], found


# -----------------------
# FRONT MATTER / METADATA
# -----------------------

def extract_front_matter(text: str, max_chars: int = 5000) -> str:
    text = safe_string(text)
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    stop_patterns = [
        r"\babstract\b",
        r"\btable\s+of\s+contents\b",
        r"^\s*contents\s*$",
        r"^\s*contents\b",
        r"^\s*\d+\.\d+(\.\d+)*\s+[A-Za-z].*$",
        r"^\s*\d+\.\s+[A-Za-z].*$",
    ]

    earliest = None
    for pattern in stop_patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            if earliest is None or m.start() < earliest:
                earliest = m.start()

    if earliest is not None:
        return text[:earliest].strip()

    return text[:max_chars].strip()


def get_front_lines_raw(text: str, max_chars: int = 5000, max_lines: int = 40) -> List[str]:
    front = extract_front_matter(text, max_chars=max_chars)
    lines = []
    for line in front.splitlines():
        raw = safe_string(line).strip()
        if raw:
            lines.append(raw)
    return lines[:max_lines]


def split_joined_front_line(line: str) -> List[str]:
    s = normalize_whitespace(line)
    if not s:
        return []

    parts: List[str] = []

    date_range_match = re.search(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*[-–]\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        s
    )
    if date_range_match:
        before = normalize_whitespace(s[:date_range_match.start()])
        date_part = normalize_whitespace(date_range_match.group(0))
        after = normalize_whitespace(s[date_range_match.end():])

        out = []
        if before:
            out.append(before)
        if date_part:
            out.append(date_part)
        if after:
            out.append(after)
        return out

    code_match = re.search(r"\([A-Z]{2,6}\d{3,}[A-Z0-9]*\)", s)
    if code_match:
        left = normalize_whitespace(s[:code_match.start()])
        code = normalize_whitespace(code_match.group(0))
        right = normalize_whitespace(s[code_match.end():])

        if left:
            parts.append(left)
        if code:
            parts.append(code)
        if right:
            parts.append(right)
    else:
        parts.append(s)

    final_parts: List[str] = []

    for part in parts:
        part = normalize_whitespace(part)
        if not part:
            continue

        caps_matches = list(re.finditer(r"\b(?:[A-Z][A-Z&/\-]*)(?:\s+[A-Z][A-Z&/\-]*){1,7}\b", part))
        if caps_matches:
            cursor = 0
            for m in caps_matches:
                before = normalize_whitespace(part[cursor:m.start()])
                caps = normalize_whitespace(m.group(0))
                if before:
                    final_parts.append(before)
                if caps:
                    final_parts.append(caps)
                cursor = m.end()
            after = normalize_whitespace(part[cursor:])
            if after:
                final_parts.append(after)
        else:
            final_parts.append(part)

    out: List[str] = []
    for p in final_parts:
        p = normalize_whitespace(p)
        if p:
            out.append(p)

    return out


def get_front_lines(text: str, max_chars: int = 5000, max_lines: int = 40) -> List[str]:
    raw_lines = get_front_lines_raw(text, max_chars=max_chars, max_lines=max_lines)

    processed: List[str] = []
    for line in raw_lines:
        split_parts = split_joined_front_line(line)
        interim_parts = split_parts if split_parts else [normalize_whitespace(line)]

        for part in interim_parts:
            blob_parts = split_collapsed_front_blob(part)
            blob_parts = blob_parts if blob_parts else [normalize_whitespace(part)]

            for blob_part in blob_parts:
                mixed_parts = split_title_author_affiliation_line(blob_part)
                if mixed_parts:
                    processed.extend(mixed_parts)
                else:
                    processed.append(normalize_whitespace(blob_part))

    return [normalize_whitespace(x) for x in processed if normalize_whitespace(x)][:max_lines]


def is_likely_org_line(line: str) -> bool:
    s = normalize_whitespace(line)
    return any(re.search(p, s, flags=re.IGNORECASE) for p in INSTITUTION_PATTERNS)


def is_title_noise(line: str) -> bool:
    s = normalize_whitespace(line)

    if not s:
        return True
    if s.lower() in SECTION_LABELS:
        return True
    if is_title_label_line(s):
        return True
    if looks_like_author_list(s):
        return True
    if is_table_of_contents_line(s):
        return True
    if looks_like_reference(s):
        return True
    if is_date_like_line(s):
        return True
    if re.fullmatch(r"\([A-Z]{2,6}\d{3,}[A-Z0-9]*\)", s):
        return True
    if re.fullmatch(r"\d+\s*\|\s*p\s*a\s*g\s*e", s, flags=re.IGNORECASE):
        return True
    if "@" in s:
        return True
    return False


def is_strong_title_candidate(line: str) -> bool:
    s = normalize_whitespace(line)
    if not s:
        return False
    if is_title_noise(s):
        return False
    if looks_like_author_list(s):
        return False
    if is_date_like_line(s):
        return False
    if line_looks_like_author_affiliation(s):
        return False
    if re.search(r"\([A-Z]{2,6}\d{3,}[A-Z0-9]*\)", s):
        return False

    words = s.split()
    if not (2 <= len(words) <= 8):
        return False

    alpha = [ch for ch in s if ch.isalpha()]
    if not alpha:
        return False

    uppercase_ratio = sum(ch.isupper() for ch in alpha) / len(alpha)
    return uppercase_ratio >= 0.8


def looks_like_title_line(s: str) -> bool:
    s = normalize_whitespace(s)
    if not s:
        return False

    if is_title_noise(s):
        return False
    if is_likely_org_line(s):
        return False
    if is_date_like_line(s):
        return False

    word_count = len(s.split())
    if word_count < 1 or word_count > 8:
        return False

    alpha = [ch for ch in s if ch.isalpha()]
    uppercase_ratio = sum(ch.isupper() for ch in alpha) / len(alpha) if alpha else 0.0

    if uppercase_ratio > 0.75:
        return True
    if s.istitle() and word_count <= 6:
        return True

    return False


def extract_contributors(text: str, nlp, max_authors: int = 8) -> List[str]:
    front_lines = get_front_lines(text, max_chars=2500, max_lines=12)
    if not front_lines:
        return []

    ents: List[str] = []

    for line in front_lines[:10]:
        s = normalize_whitespace(line)
        if not s:
            continue

        if is_title_label_line(s):
            continue
        if is_date_like_line(s):
            continue
        if is_likely_org_line(s):
            continue
        if is_strong_title_candidate(s):
            continue

        for pattern in AUTHOR_PREFIX_PATTERNS:
            s = re.sub(pattern, "", s, flags=re.IGNORECASE)

        if "@" in s:
            s = s.split("@")[0].strip()

        if looks_like_author_list(s):
            parts = [normalize_whitespace(p) for p in s.split(",")]
            for part in parts:
                cleaned = clean_person_entity(part)
                if is_valid_person_name(cleaned):
                    ents.append(cleaned)
            continue

        cleaned = clean_person_entity(s)
        if is_valid_person_name(cleaned):
            ents.append(cleaned)
            continue

        doc = nlp(s)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                cleaned = clean_person_entity(ent.text)
                if is_valid_person_name(cleaned):
                    ents.append(cleaned)

    return dedupe_keep_order(ents)[:max_authors]


def extract_title(text: str, contributors: Optional[List[str]] = None) -> str:
    contributors = contributors or []
    front_lines = get_front_lines(text, max_chars=5000, max_lines=20)

    clean_lines = []
    for line in front_lines:
        s = normalize_whitespace(line)
        if not s:
            continue
        if is_title_noise(s):
            continue
        clean_lines.append(s)

    strong_titles = []
    for i, s in enumerate(clean_lines):
        if any(s.lower() == c.lower() for c in contributors):
            continue
        if is_strong_title_candidate(s):
            score = 10.0
            if i <= 4:
                score += 1.0
            strong_titles.append((s, score))

    if strong_titles:
        strong_titles.sort(key=lambda x: x[1], reverse=True)
        return strong_titles[0][0]

    candidates = []

    for i, s in enumerate(clean_lines):
        if any(s.lower() == c.lower() for c in contributors):
            continue

        if 3 <= len(s.split()) <= 18 and not line_looks_like_author_affiliation(s):
            score = 0.0
            if i <= 3:
                score += 3.0
            if ":" in s:
                score += 1.2
            if not s.endswith("."):
                score += 0.4
            if any(ch.isdigit() for ch in s):
                score -= 2.0
            if is_likely_org_line(s):
                score -= 3.0
            candidates.append((s, score))

        if i + 1 < len(clean_lines):
            s2 = clean_lines[i + 1]

            if line_looks_like_author_affiliation(s) or line_looks_like_author_affiliation(s2):
                continue

            merged = normalize_whitespace(f"{s} {s2}")
            if 5 <= len(merged.split()) <= 24:
                score = 0.0
                if i <= 3:
                    score += 3.5
                if ":" in merged:
                    score += 1.5
                if not s.endswith((".", "!", "?", ":")):
                    score += 0.8
                if any(ch.isdigit() for ch in merged):
                    score -= 2.0
                if is_likely_org_line(merged):
                    score -= 3.0
                candidates.append((merged, score))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def extract_abstract_text(text: str, max_chars: int = 1600) -> str:
    text = safe_string(text)

    match = re.search(
        r"\babstract\b\s*(.*?)(?=\n\s*(table\s+of\s+contents|\d+(\.\d+)*\s+[A-Za-z]|introduction)\b|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if not match:
        return ""

    abstract = match.group(1)
    abstract = clean_text(abstract)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    return abstract[:max_chars]


# -----------------------
# RESEARCH GROUP
# -----------------------

def build_research_group_text(
    title: str,
    summary: str,
    summary_input: str,
    raw_text: str,
    max_chars: int = 2500
) -> str:
    pieces = []

    if title:
        pieces.append(f"Title: {title}")
    if summary:
        pieces.append(f"Summary: {summary}")
    if summary_input:
        pieces.append(f"Key content: {summary_input[:1200]}")
    elif raw_text:
        pieces.append(f"Key content: {raw_text[:1200]}")

    text = " ".join(pieces)
    return normalize_whitespace(text)[:max_chars]


def classify_research_group(
    title: str,
    summary: str,
    summary_input: str,
    raw_text: str,
    candidate_labels: List[str],
    zero_shot_classifier,
    top_k: int = 3
) -> Dict[str, Any]:
    classification_text = build_research_group_text(
        title=title,
        summary=summary,
        summary_input=summary_input,
        raw_text=raw_text
    )

    if not classification_text.strip():
        return {
            "research_group": "",
            "research_group_confidence": None,
            "research_group_top3": []
        }

    result = zero_shot_classifier(
        classification_text,
        candidate_labels=candidate_labels,
        multi_label=False,
        hypothesis_template="This document best fits the research group {}."
    )

    labels = result["labels"]
    scores = result["scores"]

    top_items = []
    for label, score in list(zip(labels, scores))[:top_k]:
        top_items.append({
            "label": label,
            "score": round(float(score), 6)
        })

    best_label = labels[0] if labels else ""
    best_score = float(scores[0]) if scores else None

    return {
        "research_group": best_label,
        "research_group_confidence": round(best_score, 6) if best_score is not None else None,
        "research_group_top3": top_items
    }


# -----------------------
# EVALUATION
# -----------------------

def get_key_entities(text: str, nlp, max_entities: int = MAX_KEY_ENTITIES) -> List[str]:
    doc = nlp(safe_string(text)[:12000])

    entities = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "PRODUCT", "EVENT"}:
            ent_text = normalize_whitespace(ent.text)
            if len(ent_text) > 2:
                entities.append(ent_text)

    return dedupe_keep_order(entities)[:max_entities]


def evaluate_summary(
    original_text: str,
    summary_text: str,
    source_entities: Optional[List[str]] = None
) -> Dict[str, Any]:
    original_tokens = simple_tokenize(original_text)
    summary_tokens = simple_tokenize(summary_text)

    original_len = len(original_tokens)
    summary_len = len(summary_tokens)
    compression_ratio = summary_len / original_len if original_len > 0 else 0.0

    source_entities = source_entities or []
    summary_lower = summary_text.lower()

    preserved_entities = []
    missed_entities = []

    for ent in source_entities:
        ent_clean = normalize_whitespace(ent)
        if ent_clean.lower() in summary_lower:
            preserved_entities.append(ent_clean)
        else:
            missed_entities.append(ent_clean)

    total_entities = len(source_entities)
    entity_preservation_score = len(preserved_entities) / total_entities if total_entities > 0 else None

    return {
        "original_token_count": original_len,
        "summary_token_count": summary_len,
        "compression_ratio": compression_ratio,
        "preserved_entities": preserved_entities,
        "missed_entities": missed_entities,
        "entity_preservation_score": entity_preservation_score
    }


def sentence_coverage_score(source_sentences: List[str], summary_text: str, threshold: float = 0.2) -> float:
    summary_words = set(simple_tokenize(summary_text))
    covered = 0

    for sent in source_sentences:
        sent_words = set(simple_tokenize(sent))
        if not sent_words:
            continue
        overlap = len(summary_words & sent_words) / len(sent_words)
        if overlap >= threshold:
            covered += 1

    return covered / len(source_sentences) if source_sentences else 0.0


def summary_redundancy_score(summary_text: str, nlp) -> float:
    summary_sentences = split_sentences(summary_text, nlp)

    if len(summary_sentences) < 2:
        return 0.0

    similarities = []
    for i in range(len(summary_sentences)):
        for j in range(i + 1, len(summary_sentences)):
            similarities.append(jaccard_similarity(summary_sentences[i], summary_sentences[j]))

    return sum(similarities) / len(similarities) if similarities else 0.0


# -----------------------
# METADATA EXTRACTION
# -----------------------

def extract_generic_metadata(
    text: str,
    summary: str,
    summary_input: str,
    document_id: str,
    nlp,
    zero_shot_classifier
) -> Dict[str, Any]:
    cleaned_text = clean_text(text)

    contributors = extract_contributors(text, nlp)
    title = extract_title(text, contributors=contributors)

    if title:
        contributors = [c for c in contributors if c.lower() != title.lower()]

    start_date, end_date, all_dates = extract_start_end_dates(text)
    abstract_text = extract_abstract_text(text)
    description = abstract_text or summary or summary_input[:500]

    research_group_info = classify_research_group(
        title=title,
        summary=summary,
        summary_input=summary_input,
        raw_text=cleaned_text,
        candidate_labels=RESEARCH_GROUPS,
        zero_shot_classifier=zero_shot_classifier
    )

    if (
        research_group_info["research_group_confidence"] is not None
        and research_group_info["research_group_confidence"] < 0.30
    ):
        research_group_info["research_group"] = ""

    top_terms = extract_top_terms(
        text=cleaned_text[:12000],
        nlp=nlp,
        top_n=MAX_TOP_TERMS
    )

    return {
        "id": document_id,
        "title": title,
        "title_found": bool(title),
        "contributors": contributors,
        "contributors_found": bool(contributors),
        "start_date": start_date,
        "end_date": end_date,
        "dates_found": all_dates,
        "description": description,
        "research_group": research_group_info["research_group"],
        "research_group_confidence": research_group_info["research_group_confidence"],
        "research_group_top3": research_group_info["research_group_top3"],
        "top_terms": top_terms,
        "metadata_debug": {
            "front_lines_sample": get_front_lines(text)[:12]
        }
    }


# -----------------------
# SUMMARIZATION
# -----------------------

def summarize_mbart(
    text: str,
    tokenizer,
    model,
    device,
    max_input_len: int = MAX_INPUT_LEN,
    max_output_len: int = MAX_OUTPUT_LEN,
    min_output_len: int = MIN_OUTPUT_LEN
) -> Tuple[str, int, int]:
    prompt_text = f"Summarize the following document: {safe_string(text)}"

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    )
    input_token_count = int(inputs["input_ids"].shape[1])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_len,
            min_length=min_output_len,
            num_beams=4,
            length_penalty=1.45,
            no_repeat_ngram_size=5,
            repetition_penalty=1.25,
            early_stopping=True
        )

    output_token_count = int(summary_ids.shape[1])
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary, input_token_count, output_token_count


def clean_generated_summary(summary: str) -> str:
    summary = safe_string(summary).strip()
    summary = re.sub(
        r"^summarize the following document:\s*",
        "",
        summary,
        flags=re.IGNORECASE
    )
    summary = remove_section_prefix(summary)
    summary = re.sub(r"\s+", " ", summary).strip()
    return summary


# -----------------------
# MAIN GOLD PIPELINE
# -----------------------

def summarize_document_row(row: pd.Series, resources: Dict[str, Any]) -> Dict[str, Any]:
    document_id = safe_string(row.get(DOC_ID_COLUMN, ""))

    text_source = row.get(TEXT_COLUMN)
    if not safe_string(text_source).strip():
        text_source = row.get(RAW_TEXT_COLUMN, "")

    text = clean_text(safe_string(text_source))

    if not text:
        raise ValueError("Input document has no usable text.")

    tokenizer = resources["tokenizer"]
    model = resources["model"]
    device = resources["device"]
    nlp = resources["nlp"]
    zero_shot_classifier = resources["zero_shot_classifier"]

    if SENTENCES_COLUMN in row and isinstance(row[SENTENCES_COLUMN], list) and row[SENTENCES_COLUMN]:
        sentences = [normalize_whitespace(s) for s in row[SENTENCES_COLUMN] if normalize_whitespace(s)]
    else:
        sentences = split_sentences(text, nlp)

    abstract_text = extract_abstract_text(text)

    if abstract_text and len(abstract_text.split()) >= 80:
        selected_sentences = []
        summary_input = abstract_text
    else:
        selected_sentences = select_balanced_sentences(sentences, tokenizer=tokenizer)
        summary_input = build_summary_input(selected_sentences)

        if not summary_input:
            summary_input = normalize_whitespace(text[:4000])

    key_entities = get_key_entities(text[:12000], nlp)

    start = time.time()
    summary, input_token_count, output_token_count = summarize_mbart(
        summary_input,
        tokenizer=tokenizer,
        model=model,
        device=device
    )
    runtime_seconds = time.time() - start

    summary = clean_generated_summary(summary)

    eval_result = evaluate_summary(
        original_text=text,
        summary_text=summary,
        source_entities=key_entities
    )
    eval_result["runtime_seconds"] = runtime_seconds
    eval_result["sentence_coverage_score"] = sentence_coverage_score(selected_sentences, summary)
    eval_result["summary_redundancy_score"] = summary_redundancy_score(summary, nlp)
    eval_result["input_token_count_to_model"] = input_token_count
    eval_result["output_token_count_from_model"] = output_token_count

    metadata = extract_generic_metadata(
        text=text,
        summary=summary,
        summary_input=summary_input,
        document_id=document_id,
        nlp=nlp,
        zero_shot_classifier=zero_shot_classifier
    )

    return {
        "document_id": document_id,
        "model": "mBART_single_input_general_reproducible_v8",
        "selected_sentences": selected_sentences,
        "summary_input": summary_input,
        "summary": summary,
        "metadata": metadata,
        "evaluation": {
            "runtime_seconds": eval_result["runtime_seconds"],
            "original_token_count": eval_result["original_token_count"],
            "summary_token_count": eval_result["summary_token_count"],
            "compression_ratio": eval_result["compression_ratio"],
            "entity_preservation_score": eval_result["entity_preservation_score"],
            "sentence_coverage_score": eval_result["sentence_coverage_score"],
            "summary_redundancy_score": eval_result["summary_redundancy_score"],
            "input_token_count_to_model": eval_result["input_token_count_to_model"],
            "output_token_count_from_model": eval_result["output_token_count_from_model"],
            "preserved_entities": eval_result["preserved_entities"],
            "missed_entities": eval_result["missed_entities"]
        }
    }


def run_gold(
    silver_nlp_json_path: str | Path,
    output_dir: str | Path = GOLD_FOLDER,
    resources: Optional[Dict[str, Any]] = None
) -> dict:
    silver_nlp_json_path = Path(silver_nlp_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(silver_nlp_json_path, orient="records")

    if df.empty:
        raise ValueError("Input JSON is empty.")

    if resources is None:
        resources = load_gold_models()

    row = df.loc[0]
    result = summarize_document_row(row, resources)

    document_id = result["document_id"]
    output_path = output_dir / f"{document_id}_mbart_output.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return {
        "document_id": document_id,
        "gold_json_path": str(output_path)
    }