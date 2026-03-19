import json
from pathlib import Path
import spacy


SILVER_FOLDER = Path("Data/silver")
SILVER_FOLDER.mkdir(parents=True, exist_ok=True)


def load_spacy_model():
    """
    Load spaCy model once.
    Make sure this is installed:
    python -m spacy download en_core_web_sm
    """
    return spacy.load("en_core_web_sm")


def preprocess_tokens(doc) -> list[str]:
    tokens = []
    for token in doc:
        if (
            not token.is_stop and
            not token.is_punct and
            not token.is_space and
            token.is_alpha and
            len(token.text) > 2
        ):
            tokens.append(token.lemma_.lower())
    return tokens


def valid_sentence(sent: str) -> bool:
    sent = sent.strip()
    if len(sent) < 40:
        return False
    if len(sent.split()) < 8:
        return False
    return True


def extract_sentences(doc) -> list[str]:
    return [sent.text.strip() for sent in doc.sents if valid_sentence(sent.text)]


def extract_entities(doc) -> list[list[str]]:
    """
    JSON-safe entity output.
    tuples become lists in JSON anyway, so we save as lists directly.
    """
    return [[ent.text, ent.label_] for ent in doc.ents]


def run_silver_nlp(
    silver_json_path: str | Path,
    output_dir: str | Path = SILVER_FOLDER,
    nlp_model=None
) -> dict:
    """
    Load cleaned silver JSON, run NLP, save silver_nlp JSON.
    Returns:
        {
            "document_id": ...,
            "silver_nlp_json_path": ...
        }
    """
    silver_json_path = Path(silver_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(silver_json_path, "r", encoding="utf-8") as f:
        silver_data = json.load(f)

    if not silver_data:
        raise ValueError(f"No records found in {silver_json_path}")

    record = silver_data[0]
    document_id = record["document_id"]
    cleaned_text = record["cleaned_text"]

    if nlp_model is None:
        nlp_model = load_spacy_model()

    doc = nlp_model(cleaned_text)

    tokens = preprocess_tokens(doc)
    sentences = extract_sentences(doc)
    entities = extract_entities(doc)

    silver_nlp_record = [{
        "document_id": document_id,
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "sentences": sentences,
        "entities": entities
    }]

    silver_nlp_json_path = output_dir / f"{document_id}_silver_nlp.json"

    with open(silver_nlp_json_path, "w", encoding="utf-8") as f:
        json.dump(silver_nlp_record, f, ensure_ascii=False, indent=4)

    return {
        "document_id": document_id,
        "silver_nlp_json_path": str(silver_nlp_json_path)
    }