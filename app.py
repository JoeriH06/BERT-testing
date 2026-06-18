
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
import time

import streamlit as st

try:
    import fitz  # PyMuPDF
    from PIL import Image
    import io
    PDF_PREVIEW_AVAILABLE = True
except Exception:
    PDF_PREVIEW_AVAILABLE = False

from medallion_pipeline import run_pipeline, load_result, ensure_data_dirs, clear_data_layers, check_ollama

st.set_page_config(
    page_title="KMP PDF Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("Data")
RAW_DIR = DATA_DIR / "raw"
ensure_data_dirs(DATA_DIR)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main-header {
        padding: 1.35rem 1.5rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, rgba(80,120,255,.16), rgba(0,190,150,.13));
        border: 1px solid rgba(130,130,130,.2);
        margin-bottom: 1rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
    }
    .main-header p {
        margin: .35rem 0 0 0;
        color: rgba(120,120,120,.95);
        font-size: 1rem;
    }
    .soft-card {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        border: 1px solid rgba(130,130,130,.18);
        background: rgba(250,250,250,.035);
        margin-bottom: .8rem;
    }
    .term-chip {
        display: inline-block;
        padding: .33rem .6rem;
        margin: .18rem .15rem;
        border-radius: 999px;
        border: 1px solid rgba(130,130,130,.22);
        background: rgba(120,120,255,.08);
        font-size: .88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def init_state():
    defaults = {
        "pipeline_info": None,
        "result": None,
        "uploaded_pdf_path": None,
        "last_error": None,
        "review_saved": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def save_uploaded_file(uploaded_file) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def render_pdf_preview(pdf_path: Path):
    st.subheader("PDF preview")
    if not pdf_path or not Path(pdf_path).exists():
        st.info("Upload a PDF first.")
        return
    if not PDF_PREVIEW_AVAILABLE:
        st.warning("PDF preview requires PyMuPDF and Pillow. Install: pip install pymupdf pillow")
        return

    with st.expander("Preview pages", expanded=True):
        zoom = st.slider("Zoom", 1.0, 2.5, 1.35, 0.05)
        max_pages = st.number_input("Pages", min_value=1, max_value=15, value=3)
        doc = fitz.open(str(pdf_path))
        for page_number in range(min(len(doc), int(max_pages))):
            page = doc[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            st.caption(f"Page {page_number + 1}")
            st.image(img, use_container_width=True)
        doc.close()


def terms_as_chips(terms):
    if not terms:
        st.write("No terms found.")
        return
    html = "".join(f"<span class='term-chip'>{str(t)}</span>" for t in terms)
    st.markdown(html, unsafe_allow_html=True)


def normalize_meta(result: dict) -> dict:
    meta = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    if not meta and any(k in result for k in ["title", "authors", "short_summary", "keywords"]):
        meta = {
            "title": result.get("title"),
            "subtitle": result.get("subtitle"),
            "contributors": result.get("authors", []),
            "supervisors": result.get("supervisors", []),
            "publication_date": result.get("date"),
            "language": result.get("language"),
            "document_type": result.get("document_type"),
            "research_or_project_topic": result.get("research_or_project_topic"),
            "research_question_or_goal": result.get("research_question_or_goal"),
            "description": result.get("short_summary"),
            "keywords": result.get("keywords", []),
            "technologies_tools_models": result.get("tools_technologies_or_models", []),
            "main_outputs_or_results": result.get("main_outputs_or_results", []),
            "keyword_suggestions": result.get("keyword_suggestions", []),
            "keyword_target_count": result.get("keyword_target_count"),
            "keyword_word_count": result.get("keyword_word_count"),
            "field_suggestions": result.get("field_suggestions", {}),
            "field_confidence": result.get("field_confidence", {}),
            "review_status": result.get("review_status"),
        }
    return meta


def build_api_output(result: dict) -> dict:
    meta = normalize_meta(result)
    return {
        "document_id": result.get("document_id") or meta.get("id"),
        "summary": result.get("document_summary") or result.get("summary"),
        "metadata": {
            "title": meta.get("title"),
            "subtitle": meta.get("subtitle"),
            "contributors": meta.get("contributors", []),
            "supervisors": meta.get("supervisors", []),
            "publication_date": meta.get("publication_date"),
            "language": meta.get("language"),
            "document_type": meta.get("document_type"),
            "description": meta.get("description"),
            "keywords": meta.get("keywords", []),
            "technologies_tools_models": meta.get("technologies_tools_models", []),
            "research_or_project_topic": meta.get("research_or_project_topic"),
            "research_question_or_goal": meta.get("research_question_or_goal"),
            "review_status": meta.get("review_status"),
            "field_confidence": meta.get("field_confidence", {}),
        },
        "search": {
            "keywords": meta.get("keywords", []),
            "keyword_suggestions": meta.get("keyword_suggestions", []),
            "target_keyword_count": meta.get("keyword_target_count"),
        },
        "topics": result.get("main_topics", []),
        "entities": result.get("suggested_entities", {}),
        "quality": result.get("quality", {}),
        "pipeline": result.get("@pipeline", {}),
    }


def metadata_status(value) -> str:
    if isinstance(value, list):
        return "Found" if value else "Needs review"
    return "Found" if str(value or "").strip() else "Needs review"


def render_field_status(label: str, value) -> None:
    status = metadata_status(value)
    if status == "Found":
        st.success(f"{label}: found")
    else:
        st.warning(f"{label}: not confident")


def field_suggestions(meta: dict, field: str) -> list[dict]:
    suggestions = meta.get("field_suggestions") or {}
    values = suggestions.get(field) or []
    if isinstance(values, dict):
        values = [values]
    return [item for item in values if isinstance(item, dict) and item.get("value")]


def display_suggestion_value(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value if str(x).strip())
    return str(value or "").strip()


def apply_field_suggestion(result: dict, field: str, value) -> None:
    meta = normalize_meta(result)
    if field == "contributors":
        if isinstance(value, list):
            meta[field] = [str(x).strip() for x in value if str(x).strip()]
        else:
            meta[field] = [x.strip() for x in str(value).split(",") if x.strip()]
    else:
        meta[field] = display_suggestion_value(value)
    result["metadata"] = meta
    st.session_state.result = result


def render_field_suggestions(result: dict, meta: dict) -> None:
    field_labels = {
        "title": "Title",
        "contributors": "Contributors",
        "publication_date": "Date",
        "document_type": "Document type",
    }
    shown = False
    for field, label in field_labels.items():
        current = meta.get(field)
        if metadata_status(current) == "Found":
            continue
        suggestions = field_suggestions(meta, field)
        if not suggestions:
            continue
        if not shown:
            st.markdown("#### Suggestions to review")
            st.caption("The pipeline is not confident enough to fill these fields automatically. Accept a suggestion with one click or leave it empty.")
            shown = True
        best = sorted(suggestions, key=lambda item: float(item.get("confidence") or 0), reverse=True)[0]
        value_text = display_suggestion_value(best.get("value"))
        cols = st.columns([1.3, 3.5, 0.9, 0.8, 0.8])
        cols[0].markdown(f"**{label}**")
        cols[1].write(value_text)
        cols[2].caption(f"{float(best.get('confidence') or 0):.2f} · {best.get('source') or 'suggestion'}")
        if cols[3].button("Accept", key=f"accept_{field}_{value_text}", use_container_width=True):
            apply_field_suggestion(result, field, best.get("value"))
            st.rerun()
        if cols[4].button("Skip", key=f"skip_{field}_{value_text}", use_container_width=True):
            st.toast(f"Skipped suggested {label.lower()}.")


def keyword_suggestion_rows(meta: dict) -> list[dict]:
    suggestions = meta.get("keyword_suggestions") or []
    suggestion_by_term = {}
    for item in suggestions:
        if isinstance(item, dict):
            term = item.get("term") or item.get("keyword") or item.get("text")
            if term:
                suggestion_by_term[str(term).lower()] = item

    current_keywords = [str(term).strip() for term in meta.get("keywords", []) if str(term).strip()]
    if current_keywords:
        rows = []
        for term in current_keywords:
            suggestion = suggestion_by_term.get(term.lower(), {})
            confidence = suggestion.get("confidence")
            rows.append({
                "Use": True,
                "Keyword": term,
                "Confidence": None if confidence is None else round(float(confidence), 2),
                "Source": suggestion.get("source") or "reviewed",
                "Why": suggestion.get("reason") or "Saved in the reviewed keyword list.",
            })
        return rows[:30]

    if not suggestions:
        suggestions = [{"term": term, "confidence": None, "source": "metadata"} for term in meta.get("keywords", [])]
    rows = []
    for item in suggestions:
        if isinstance(item, dict):
            term = item.get("term") or item.get("keyword") or item.get("text")
            confidence = item.get("confidence")
            source = item.get("source")
            reason = item.get("reason")
        else:
            term, confidence, source, reason = str(item), None, None, None
        if term:
            rows.append({
                "Use": True,
                "Keyword": term,
                "Confidence": None if confidence is None else round(float(confidence), 2),
                "Source": source or "",
                "Why": reason or "",
            })
    return rows[:30]


def parse_keyword_text(value: str) -> list[str]:
    keywords = []
    seen = set()
    for item in re.split(r"[,;\n]+", str(value or "")):
        keyword = item.strip()
        low = keyword.lower()
        if keyword and low not in seen:
            keywords.append(keyword)
            seen.add(low)
    return keywords


def top_term_strings(result: dict) -> list[str]:
    terms = []
    for item in result.get("top_terms", []):
        if isinstance(item, dict) and item.get("term"):
            terms.append(item["term"])
        elif isinstance(item, str):
            terms.append(item)
    if not terms:
        terms = result.get("metadata", {}).get("keywords", [])
    return terms



def load_best_pipeline_result(info: dict, data_dir: Path) -> dict:
    """Load the richest final result. Prefer Gold Meta output because Streamlit displays the final KMP metadata result."""
    candidate_paths = []

    for key in [
        "gold_meta_json_path",
        "gold_metadata_json_path",
        "metadata_json_path",
        "meta_json_path",
        "final_json_path",
        "gold_json_path",
    ]:
        value = info.get(key) if isinstance(info, dict) else None
        if value:
            candidate_paths.append(Path(value))

    # First load any provided path to discover document_id.
    discovered_doc_id = None
    for p in candidate_paths:
        try:
            if p.exists():
                tmp = json.loads(p.read_text(encoding="utf-8"))
                discovered_doc_id = tmp.get("document_id") or tmp.get("@pipeline", {}).get("document_id")
                break
        except Exception:
            pass

    # Prefer the explicit Gold Meta files written by the Medaillon Gold Meta notebook.
    if discovered_doc_id:
        candidate_paths.insert(0, data_dir / "gold_meta" / f"{discovered_doc_id}_gold_metadata.json")
        candidate_paths.insert(1, data_dir / "gold_meta" / f"{discovered_doc_id}_meta.json")

    # Fallback: newest Gold Meta result.
    gold_meta_dir = data_dir / "gold_meta"
    if gold_meta_dir.exists():
        newest_meta = sorted(gold_meta_dir.glob("*_gold_metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidate_paths.extend(newest_meta)

    loaded = []
    for p in candidate_paths:
        try:
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                loaded.append((p, obj))
        except Exception:
            pass

    if not loaded:
        raise FileNotFoundError("No pipeline result JSON could be loaded.")

    def score_result(item):
        p, obj = item
        score = 0
        if isinstance(obj.get("evaluation"), dict):
            score += 100
        if isinstance(obj.get("metadata"), dict):
            score += 50
        if "gold_meta" in str(p).lower():
            score += 30
        if obj.get("quality", {}).get("runtime_seconds") is not None:
            score += 10
        return score

    best_path, best_result = sorted(loaded, key=score_result, reverse=True)[0]
    best_result["_loaded_result_path"] = str(best_path)
    return best_result



def render_metadata_editor(result: dict):
    meta = normalize_meta(result)
    st.markdown("### Review suggested metadata")
    st.caption("Empty fields mean the pipeline did not find enough evidence. That is safer than filling in a wrong title or contributor.")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        render_field_status("Title", meta.get("title"))
    with s2:
        render_field_status("Contributors", meta.get("contributors"))
    with s3:
        render_field_status("Date", meta.get("publication_date"))
    with s4:
        render_field_status("Document type", meta.get("document_type"))

    render_field_suggestions(result, meta)

    with st.form("metadata_review_form"):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Title", value=meta.get("title") or "")
            contributors = st.text_input("Contributors", value=", ".join(meta.get("contributors") or []))
            publication_date = st.text_input("Date", value=meta.get("publication_date") or "")
            document_type = st.text_input("Document type", value=meta.get("document_type") or "")
        with col2:
            language = st.text_input("Language", value=meta.get("language") or "")
            topic = st.text_area("Research/project topic", value=meta.get("research_or_project_topic") or "", height=90)
            question = st.text_area("Research question / goal", value=meta.get("research_question_or_goal") or "", height=90)

        description = st.text_area("Description / summary", value=meta.get("description") or result.get("document_summary") or "", height=160)

        saved = st.form_submit_button("Save reviewed metadata", type="primary", use_container_width=True)

    if saved:
        meta["title"] = title.strip()
        meta["contributors"] = [x.strip() for x in contributors.split(",") if x.strip()]
        meta["publication_date"] = publication_date.strip()
        meta["document_type"] = document_type.strip()
        meta["language"] = language.strip()
        meta["research_or_project_topic"] = topic.strip()
        meta["research_question_or_goal"] = question.strip()
        meta["description"] = description.strip()
        meta["keywords"] = meta.get("keywords", [])
        meta.setdefault("field_suggestions", result.get("field_suggestions", {}))
        meta["suitable_kmp_fields"] = {
            "title": meta["title"],
            "description": meta["description"],
            "keywords": meta["keywords"],
            "contributors": meta["contributors"],
            "date": meta["publication_date"],
            "language": meta["language"],
            "document_type": meta["document_type"],
        }
        result["metadata"] = meta
        result["document_summary"] = description.strip()
        result["summary"] = description.strip()
        result["api_output"] = build_api_output(result)
        st.session_state.result = result

        doc_id = result.get("document_id", "reviewed")
        out = DATA_DIR / "gold" / f"{doc_id}_reviewed_result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
        st.session_state.review_saved = True
        st.success(f"Reviewed result saved to {out}")


def render_keyword_selector(result: dict):
    meta = normalize_meta(result)
    rows = keyword_suggestion_rows(meta)
    st.markdown("### Search keywords")
    target = meta.get("keyword_target_count")
    word_count = meta.get("keyword_word_count") or result.get("quality", {}).get("word_count") or result.get("statistics", {}).get("main_text_words")
    if target:
        st.caption(f"Review the suggested search terms, add missing terms, then save one final keyword list. Recommended for this document: about {target}.")
    else:
        st.caption("Review the suggested search terms, add missing terms, then save one final keyword list.")
    if not rows:
        st.info("No keyword suggestions were generated.")
        rows = [{"Use": True, "Keyword": term, "Confidence": None, "Source": "manual", "Why": ""} for term in meta.get("keywords", [])]

    k1, k2, k3 = st.columns(3)
    k1.metric("Current suggestions", len(rows))
    k2.metric("Recommended", target or "-")
    k3.metric("Document words", word_count or "-")

    edited = []
    if rows:
        edited = st.data_editor(
            rows,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Use": st.column_config.CheckboxColumn("Keep"),
                "Keyword": st.column_config.TextColumn("Keyword", width="medium"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                "Why": st.column_config.TextColumn("Why", width="large"),
            },
            disabled=["Keyword", "Confidence", "Source", "Why"],
            key=f"keyword_selector_{result.get('document_id', 'result')}",
        )

    extra_keywords = st.text_input(
        "Add missing keywords",
        placeholder="Example: policy evaluation, stakeholder interviews",
        key=f"extra_keywords_{result.get('document_id', 'result')}",
    )

    edited_rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
    selected = [row["Keyword"] for row in edited_rows if row.get("Use")]
    selected.extend(parse_keyword_text(extra_keywords))
    deduped = []
    seen = set()
    for keyword in selected:
        keyword = str(keyword).strip()
        low = keyword.lower()
        if keyword and low not in seen:
            deduped.append(keyword)
            seen.add(low)

    st.caption(f"{len(deduped)} keywords will be saved.")
    if st.button("Save selected keywords", use_container_width=True):
        meta["keywords"] = deduped
        meta["keyword_suggestions"] = [
            {
                "term": keyword,
                "confidence": None,
                "source": "reviewed",
                "reason": "Saved by reviewer.",
            }
            for keyword in deduped
        ]
        result["metadata"] = meta
        result["api_output"] = build_api_output(result)
        st.session_state.result = result
        st.success(f"Saved {len(deduped)} keywords.")
        st.rerun()


def render_api_output(result: dict):
    api_output = result.get("api_output") or build_api_output(result)
    result["api_output"] = api_output
    st.markdown("### API-ready output")
    st.caption("This compact JSON is meant for a future server/API. Other systems can request only the parts they need, such as metadata, search keywords, entities, or summary.")

    sections = {
        "Metadata": api_output.get("metadata", {}),
        "Search": api_output.get("search", {}),
        "Summary": {"summary": api_output.get("summary")},
        "Entities": api_output.get("entities", {}),
        "Quality": api_output.get("quality", {}),
    }
    selected_sections = st.multiselect(
        "Preview output sections",
        options=list(sections.keys()),
        default=["Metadata", "Search", "Summary"],
        key=f"api_sections_{result.get('document_id', 'result')}",
    )
    preview = {"document_id": api_output.get("document_id")}
    for section in selected_sections:
        preview[section.lower()] = sections[section]

    st.json(preview)
    st.download_button(
        "Download API-ready JSON",
        data=json.dumps(api_output, indent=2, ensure_ascii=False),
        file_name=f"{api_output.get('document_id') or 'document'}_api_output.json",
        mime="application/json",
        use_container_width=True,
    )


def render_result(result: dict):
    meta = normalize_meta(result)
    quality = result.get("quality", {})
    stats = result.get("statistics", {})

    st.subheader(meta.get("title") or "Document ready for review")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Language", meta.get("language") or result.get("language") or "unknown")
    m2.metric("Words", quality.get("word_count") or stats.get("main_text_words") or "-")
    m3.metric("Chunks", quality.get("chunk_count") or stats.get("chunk_count") or "-")
    m4.metric("Model", result.get("model") or "-")

    st.markdown("### Summary")
    st.write(result.get("document_summary") or result.get("summary") or "No summary generated.")
    value = result.get("possible_value_for_knowledge_platform")
    if value:
        with st.expander("Possible value for the document system", expanded=False):
            st.write(value)

    st.divider()
    render_metadata_editor(result)

    st.divider()
    render_keyword_selector(result)

    st.divider()
    with st.expander("API-ready output", expanded=False):
        render_api_output(result)

    with st.expander("Optional details", expanded=False):
        col_entities, col_eval = st.columns(2)
        with col_entities:
            st.markdown("#### Suggested entities")
            entities = result.get("suggested_entities") or {}
            if not entities:
                st.write("No entity suggestions.")
            for group, values in entities.items():
                if values:
                    st.markdown(f"**{group}**")
                    if isinstance(values, list):
                        shown = [v.get("text") or v.get("term") or str(v) if isinstance(v, dict) else str(v) for v in values]
                        terms_as_chips(shown[:15])
                    else:
                        st.write(values)
        with col_eval:
            st.markdown("#### Evaluation")
            evaluation = result.get("evaluation", {}) if isinstance(result.get("evaluation"), dict) else {}
            if not evaluation:
                st.write("No evaluation metrics found.")
            else:
                st.table([
                    {"Metric": key.replace("_", " ").title(), "Value": value}
                    for key, value in evaluation.items()
                    if key not in {"criteria"}
                ][:10])

    with st.expander("Developer output JSON", expanded=False):
        st.download_button(
            "Download result JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name=f"{result.get('document_id', 'result')}.json",
            mime="application/json",
            use_container_width=True,
        )
        st.json(result)


# -----------------------------
# App
# -----------------------------
init_state()

st.markdown(
    """
    <div class="main-header">
        <h1>📄 KMP PDF Intelligence</h1>
        <p>Local Bronze → Silver → Silver NLP → Gold → Gold Meta pipeline for Dutch and English PDFs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Pipeline settings")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    model = st.text_input("Ollama model", value="qwen2.5:3b-instruct")
    require_ollama = st.toggle("Require Ollama", value=True, help="Turn this off only for testing deterministic fallback output.")
    clear_previous = st.toggle("Clear previous Data outputs", value=True)

    st.divider()
    if st.button("Check Ollama", use_container_width=True):
        if check_ollama():
            st.success("Ollama is reachable.")
        else:
            st.error("Ollama is not reachable. Run: ollama serve")

    run_clicked = st.button("Run complete pipeline", type="primary", disabled=uploaded_file is None, use_container_width=True)

    st.caption("For laptop testing use qwen2.5:3b-instruct. On a school server you can switch to 7B/14B.")

if uploaded_file is not None:
    pdf_path = save_uploaded_file(uploaded_file)
    st.session_state.uploaded_pdf_path = str(pdf_path)

    preview_tab, workspace_tab = st.tabs(["PDF preview", "Extraction workspace"])

    with preview_tab:
        render_pdf_preview(pdf_path)

    with workspace_tab:
        st.info(f"Ready to process: **{uploaded_file.name}**")

        if run_clicked:
            progress_bar = st.progress(0)
            status = st.empty()
            log_box = st.empty()

            logs = []

            def progress_callback(step: str, value: float):
                progress_bar.progress(min(max(value, 0.0), 1.0))
                status.write(f"**{step}**")
                logs.append(f"{time.strftime('%H:%M:%S')} — {step}")
                log_box.code("\n".join(logs[-8:]))

            try:
                if clear_previous:
                    clear_data_layers(DATA_DIR, keep_raw_file=uploaded_file.name)

                with st.spinner("Running local pipeline..."):
                    info = run_pipeline(
                        pdf_path,
                        model=model.strip() or "qwen2.5:3b-instruct",
                        data_dir=DATA_DIR,
                        require_ollama=require_ollama,
                        clear_previous=False,
                        progress_callback=progress_callback,
                    )

                result = load_best_pipeline_result(info, DATA_DIR)

                st.session_state.pipeline_info = info
                st.session_state.result = result
                st.session_state.last_error = None

                st.success("Pipeline completed successfully.")
                st.rerun()

            except Exception as e:
                st.session_state.last_error = str(e)
                st.error(f"Pipeline failed: {e}")
                st.info("If it failed at Gold/Gold Meta, make sure Ollama is running and the model is pulled.")

elif st.session_state.uploaded_pdf_path:
    render_pdf_preview(Path(st.session_state.uploaded_pdf_path))
else:
    st.info("Upload a PDF in the sidebar to start.")

if st.session_state.last_error:
    with st.expander("Last error", expanded=False):
        st.code(st.session_state.last_error)

if st.session_state.result:
    st.divider()
    render_result(st.session_state.result)
