from pathlib import Path
import json
import re
from datetime import datetime

# pip install langdetect

from langdetect import detect

def detect_language(text):
    try:
        return detect(text[:5000])
    except:
        return "unknown"


BRONZE_FOLDER = Path("Data/bronze")
SILVER_FOLDER = Path("Data/silver")

SILVER_FOLDER.mkdir(parents=True, exist_ok=True)

BAD_CHUNK_PATTERNS = [
    "summarize the text factually",
    "use only information explicitly present",
    "don't invent sources",
    "do not write in news-article style",
]


def normalize_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def remove_table_of_contents(text):
    start = text.lower().find("inhoudsopgave")
    end = text.lower().find("managementsamenvatting")

    if start != -1 and end != -1 and end > start:
        return text[:start] + text[end:]

    return text



def remove_pdf_artifacts(text):
    # Remove dotted table-of-content lines
    text = re.sub(r"\.{5,}", " ", text)

    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove figure/table lines
    text = re.sub(r"^\s*(Afbeelding|Figuur|Tabel)\s+\d+.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove very long numeric garbage strings
    text = re.sub(r"\b[\d,%.\-]{15,}\b", " ", text)

    # Remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    return text

def fix_spacing_for_metadata(text):
    # Keep line structure
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def fix_spacing_for_modeling(text):
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Only merge obvious broken sentence lines
    text = re.sub(
        r"(?<=[a-zà-ÿ,;])\n(?=[a-zà-ÿ])",
        " ",
        text
    )

    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def clean_text(text, mode="modeling"):
    text = normalize_text(text)
    text = remove_table_of_contents(text)
    text = remove_pdf_artifacts(text)

    if mode == "metadata":
        return fix_spacing_for_metadata(text)

    return fix_spacing_for_modeling(text)

def clean_text_for_chunking(text):
    text = re.sub(r"\n{3,}", "\n\n", text)

    cleaned_lines = []

    for line in text.splitlines():
        line_stripped = line.strip()

        if not line_stripped:
            cleaned_lines.append("")
            continue

        lower = line_stripped.lower()

        if any(pattern in lower for pattern in BAD_CHUNK_PATTERNS):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

def split_long_text_by_sentences(text, max_words=900):
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue

        word_count = len(sentence.split())

        if current_words + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_words = word_count
        else:
            current_chunk.append(sentence)
            current_words += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def split_into_semantic_chunks(text, max_words=900, min_words=250):
    text = clean_text_for_chunking(text)

    # Split on section headings like P.1, P.2, P.3.1, etc.
    section_parts = re.split(r"(?=\bP\.\d+(?:\.\d+)?\s+)", text)

    chunks = []

    for section in section_parts:
        section = section.strip()

        if len(section.split()) < 40:
            continue

        if len(section.split()) <= max_words:
            chunks.append(section)
        else:
            smaller_chunks = split_long_text_by_sentences(
                section,
                max_words=max_words
            )
            chunks.extend(smaller_chunks)

    # Merge tiny chunks with the previous chunk
    merged_chunks = []

    for chunk in chunks:
        word_count = len(chunk.split())

        if (
            merged_chunks
            and word_count < min_words
            and len(merged_chunks[-1].split()) + word_count <= max_words
        ):
            merged_chunks[-1] = merged_chunks[-1] + "\n\n" + chunk
        else:
            merged_chunks.append(chunk)

    return merged_chunks

def build_chunk_records(chunks):
    records = []

    for index, chunk in enumerate(chunks, start=1):
        records.append({
            "chunk_id": f"chunk_{index:03d}",
            "text": chunk,
            "word_count": len(chunk.split()),
            "character_count": len(chunk)
        })

    return records

def build_quality_report(cleaned_text, chunks):
    word_count = len(cleaned_text.split())
    character_count = len(cleaned_text)

    if chunks:
        average_chunk_words = sum(len(chunk.split()) for chunk in chunks) / len(chunks)
    else:
        average_chunk_words = 0

    return {
        "is_empty": word_count == 0,
        "word_count": word_count,
        "character_count": character_count,
        "chunk_count": len(chunks),
        "average_chunk_words": round(average_chunk_words, 2),
        "ready_for_modeling": word_count > 50 and len(chunks) > 0
    }

def save_silver_output(
    document_id,
    cleaned_text,
    metadata_text,
    chunk_records,
    quality_report,
    language
):
    txt_output = SILVER_FOLDER / f"{document_id}_clean.txt"
    meta_txt_output = SILVER_FOLDER / f"{document_id}_metadata_clean.txt"
    json_output = SILVER_FOLDER / f"{document_id}_silver.json"

    txt_output.write_text(cleaned_text, encoding="utf-8")
    meta_txt_output.write_text(metadata_text, encoding="utf-8")

    silver_data = {
        "document_id": document_id,
        "clean_text_file": str(txt_output),
        "metadata_text_file": str(meta_txt_output),
        "quality": quality_report,
        "chunks": chunk_records,
        "language": language,
        "processed_at": datetime.now().isoformat()
    }

    json_output.write_text(
        json.dumps(silver_data, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

def run_silver_layer():
    bronze_files = sorted(BRONZE_FOLDER.glob("doc_*.txt"))

    if not bronze_files:
        print("No bronze files found.")
        return

    for file_path in bronze_files:
        document_id = file_path.stem

        print(f"Cleaning {file_path.name}...")

        raw_text = file_path.read_text(encoding="utf-8")

        cleaned_text = clean_text(raw_text, mode="modeling")
        metadata_text = clean_text(raw_text, mode="metadata")

        chunks = split_into_semantic_chunks(
            cleaned_text,
            max_words=900,
            min_words=250
        )

        chunk_records = build_chunk_records(chunks)
        quality_report = build_quality_report(cleaned_text, chunks)
        language = detect_language(cleaned_text)

        save_silver_output(
            document_id=document_id,
            cleaned_text=cleaned_text,
            metadata_text=metadata_text,
            chunk_records=chunk_records,
            quality_report=quality_report,
            language=language
        )

        print(f"Saved {document_id} to silver layer.")
        print(f"Ready for modeling: {quality_report['ready_for_modeling']}")
