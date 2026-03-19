import json
import re
from pathlib import Path


SILVER_FOLDER = Path("Data/silver")
SILVER_FOLDER.mkdir(parents=True, exist_ok=True)


def normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def replace_special_spaces(text: str) -> str:
    text = text.replace("\t", " ")
    text = text.replace("\xa0", " ")
    return text


def remove_zero_width_characters(text: str) -> str:
    zero_width_chars = ['\u200B', '\u200C', '\u200D', '\uFEFF']
    for char in zero_width_chars:
        text = text.replace(char, '')
    return text


def fix_hyphenated_linebreaks(text: str) -> str:
    # example: "infor-\nmation" -> "information"
    return re.sub(r'(\w)-\n(\w)', r'\1\2', text)


def remove_pdf_page_markers(text: str) -> str:
    patterns = [
        r'Page\s+\d+\s+of\s+\d+',
        r'^\s*Page\s+\d+\s*$',
        r'^\s*\d+\s*\|\s*P\s*a\s*g\s*e\s*$',
        r'^\s*P\s*a\s*g\s*e\s*\d+\s*$',
        r'^\s*\d+\s*$'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def remove_urls_and_emails(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)
    return text


def remove_trailing_spaces(text: str) -> str:
    return re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)


def remove_repeated_spaces(text: str) -> str:
    return re.sub(r' {2,}', ' ', text)


def normalize_linebreak_spacing(text: str) -> str:
    return re.sub(r' *\n *', '\n', text)


def merge_broken_lines(text: str) -> str:
    lines = text.split('\n')
    merged = []

    for line in lines:
        line = line.strip()

        if not line:
            merged.append("")
            continue

        if not merged:
            merged.append(line)
            continue

        prev = merged[-1]

        # merge if previous line does not look like paragraph end
        if prev and not prev.endswith(('.', '!', '?', ':')) and line and not line[0].isupper():
            merged[-1] = prev + ' ' + line
        elif prev and not prev.endswith(('.', '!', '?', ':')) and len(line.split()) < 6:
            merged[-1] = prev + ' ' + line
        else:
            merged.append(line)

    return '\n'.join(merged)


def remove_reference_like_lines(text: str) -> str:
    cleaned_lines = []

    for line in text.split('\n'):
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        if re.match(r'^[A-Z][a-zA-Z\-]+,\s+[A-Z]\.', stripped):
            continue
        if re.search(r'\(\d{4}\)', stripped) and len(stripped.split()) < 12:
            continue
        if stripped.lower().startswith("references"):
            continue
        if stripped.lower().startswith("bibliography"):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_duplicate_lines(text: str) -> str:
    seen = set()
    result = []

    for line in text.split('\n'):
        key = re.sub(r'\s+', ' ', line.strip().lower())
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(line)

    return '\n'.join(result)


def remove_many_blank_lines(text: str) -> str:
    return re.sub(r'\n{3,}', '\n\n', text)


def clean_text(text: str) -> str:
    text = normalize_line_endings(text)
    text = replace_special_spaces(text)
    text = remove_zero_width_characters(text)
    text = fix_hyphenated_linebreaks(text)
    text = remove_pdf_page_markers(text)
    text = remove_urls_and_emails(text)
    text = remove_trailing_spaces(text)
    text = remove_repeated_spaces(text)
    text = normalize_linebreak_spacing(text)
    text = merge_broken_lines(text)
    text = remove_reference_like_lines(text)
    text = remove_duplicate_lines(text)
    text = remove_many_blank_lines(text)
    return text.strip()


def sanity_check(text: str) -> str:
    print("Original Text:\n")
    print(text[:500])
    print("\n---\n")
    print("Cleaned Text:\n")
    cleaned = clean_text(text)
    print(cleaned[:500])
    return cleaned


def run_silver(bronze_json_path: str | Path, output_dir: str | Path = SILVER_FOLDER) -> dict:
    """
    Load bronze JSON, clean raw text, save silver JSON.
    Returns:
        {
            "document_id": ...,
            "silver_json_path": ...
        }
    """
    bronze_json_path = Path(bronze_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(bronze_json_path, "r", encoding="utf-8") as f:
        bronze_data = json.load(f)

    if not bronze_data:
        raise ValueError(f"No records found in {bronze_json_path}")

    record = bronze_data[0]
    document_id = record["document_id"]
    raw_text = record["raw_text"]

    cleaned_text = clean_text(raw_text)

    silver_record = [{
        "document_id": document_id,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text
    }]

    silver_json_path = output_dir / f"{document_id}_silver.json"

    with open(silver_json_path, "w", encoding="utf-8") as f:
        json.dump(silver_record, f, ensure_ascii=False, indent=4)

    return {
        "document_id": document_id,
        "silver_json_path": str(silver_json_path)
    }