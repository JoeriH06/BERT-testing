from pathlib import Path
import json
import re
from datetime import datetime
from collections import Counter

import spacy

SILVER_FOLDER = Path("Data/silver")
SILVER_NLP_FOLDER = Path("Data/silver_nlp")

SILVER_NLP_FOLDER.mkdir(parents=True, exist_ok=True)

SPACY_MODELS = {
    "nl": "nl_core_news_sm",
    "en": "en_core_web_sm"
}

def load_spacy_model(language):
    model_name = SPACY_MODELS.get(language)

    if model_name is None:
        return None

    return spacy.load(model_name)

def is_valid_entity(text, label):
    text = text.strip()

    if len(text) < 3:
        return False

    if not any(char.isalpha() for char in text):
        return False

    if len(text.split()) > 8:
        return False

    digit_count = sum(c.isdigit() for c in text)
    letter_count = sum(c.isalpha() for c in text)

    if digit_count > letter_count:
        return False

    if re.fullmatch(r"[A-Z]\.", text):
        return False

    if re.search(r"\d{4,}\d{4,}", text):
        return False

    if "%" in text and label != "DATE":
        return False

    if label == "DATE":
        has_year = re.search(r"\b(18|19|20)\d{2}\b", text)
        has_number = any(c.isdigit() for c in text)
        has_time_word = re.search(
            r"\b(jaar|maand|week|dag|januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\b",
            text.lower()
        )

        if not (has_year or has_number or has_time_word):
            return False

    return True

def extract_entities(doc):
    raw_entities = {
        "persons": [],
        "organizations": [],
        "locations": [],
        "dates": []
    }

    for ent in doc.ents:
        entity_text = ent.text.strip()

        if not is_valid_entity(entity_text, ent.label_):
            continue

        if ent.label_ == "PER":
            raw_entities["persons"].append(entity_text)

        elif ent.label_ == "ORG":
            raw_entities["organizations"].append(entity_text)

        elif ent.label_ in ["LOC", "GPE"]:
            raw_entities["locations"].append(entity_text)

        elif ent.label_ == "DATE":
            raw_entities["dates"].append(entity_text)

    filtered_entities = {}

    for key, values in raw_entities.items():
        counter = Counter(values)

        filtered_entities[key] = [
            {
                "text": text,
                "count": count
            }
            for text, count in counter.most_common(25)
        ]

    return filtered_entities

def extract_keywords(doc, max_keywords=30):
    words = []

    for token in doc:
        if (
            token.is_alpha
            and not token.is_stop
            and len(token.text) > 3
            and token.pos_ in ["NOUN", "PROPN", "ADJ"]
        ):
            words.append(token.lemma_.lower())

    counter = Counter(words)

    return [
        {
            "keyword": word,
            "count": count
        }
        for word, count in counter.most_common(max_keywords)
    ]

def extract_statistics(text, doc):
    return {
        "characters": len(text),
        "words": len(text.split()),
        "sentences": len(list(doc.sents)),
        "entities_found_raw": len(doc.ents)
    }

def build_modeling_hints(entities, keywords):
    important_entities = []

    for group in ["persons", "organizations", "locations"]:
        for item in entities.get(group, [])[:10]:
            important_entities.append(item["text"])

    main_topics = [item["keyword"] for item in keywords[:10]]

    return {
        "main_topics": main_topics,
        "important_entities": important_entities
    }


def save_nlp_output(document_id, nlp_data):
    output_path = SILVER_NLP_FOLDER / f"{document_id}_nlp.json"

    output_path.write_text(
        json.dumps(nlp_data, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Saved NLP output: {output_path}")


def run_silver_nlp_layer():
    clean_files = sorted(SILVER_FOLDER.glob("doc_*_clean.txt"))

    if not clean_files:
        print("No clean silver files found.")

    for file_path in clean_files:
        document_id = file_path.stem.replace("_clean", "")

        print(f"Processing NLP for {document_id}...")

        text = file_path.read_text(encoding="utf-8")

        doc = nlp(text)

        statistics = extract_statistics(text, doc)
        entities = extract_entities(doc)
        keywords = extract_keywords(doc)
        modeling_hints = build_modeling_hints(entities, keywords)

        nlp_data = {
            "document_id": document_id,
            "source_clean_file": str(file_path),
            "statistics": statistics,
            "entities": entities,
            "keywords": keywords,
            "modeling_hints": modeling_hints,
            "processed_at": datetime.now().isoformat()
        }

        save_nlp_output(document_id, nlp_data)

    print("Silver NLP layer completed.")



_MODEL_CACHE = {}

def get_language_for_document(document_id):
    silver_json = SILVER_FOLDER / f"{document_id}_silver.json"
    if silver_json.exists():
        try:
            return json.loads(silver_json.read_text(encoding="utf-8")).get("language", "en")
        except Exception:
            return "en"
    return "en"

def get_nlp(language):
    language = language if language in SPACY_MODELS else "en"
    if language not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[language] = load_spacy_model(language)
        except Exception:
            _MODEL_CACHE[language] = None
    return _MODEL_CACHE[language]

def run_silver_nlp_layer(document_ids=None):
    clean_files = sorted(SILVER_FOLDER.glob("doc_*_clean.txt"))
    if document_ids:
        document_ids = set(document_ids)
        clean_files = [p for p in clean_files if p.stem.replace("_clean", "") in document_ids]

    if not clean_files:
        print("No clean silver files found.")
        return []

    outputs = []
    for file_path in clean_files:
        document_id = file_path.stem.replace("_clean", "")
        print(f"Processing NLP for {document_id}...")

        text = file_path.read_text(encoding="utf-8")
        language = get_language_for_document(document_id)
        nlp = get_nlp(language)

        if nlp is None:
            statistics = {
                "characters": len(text),
                "words": len(text.split()),
                "sentences": len(re.split(r"(?<=[.!?])\s+", text)),
                "entities_found_raw": 0
            }
            entities = {"persons": [], "organizations": [], "locations": [], "dates": []}
            keywords = [{"keyword": word, "count": count} for word, count in Counter(
                w.lower() for w in re.findall(r"\b[A-Za-zÀ-ÿ]{4,}\b", text)
            ).most_common(30)]
        else:
            doc = nlp(text[:1000000])
            statistics = extract_statistics(text, doc)
            entities = extract_entities(doc)
            keywords = extract_keywords(doc)

        modeling_hints = build_modeling_hints(entities, keywords)

        nlp_data = {
            "document_id": document_id,
            "source_clean_file": str(file_path),
            "language": language,
            "statistics": statistics,
            "entities": entities,
            "keywords": keywords,
            "modeling_hints": modeling_hints,
            "processed_at": datetime.now().isoformat()
        }

        save_nlp_output(document_id, nlp_data)
        outputs.append(str(SILVER_NLP_FOLDER / f"{document_id}_nlp.json"))

    print("Silver NLP layer completed.")
    return outputs
