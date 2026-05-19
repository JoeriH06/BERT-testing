
from __future__ import annotations

import json
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

from src.pipeline import run_pipeline, load_result, ensure_data_dirs, clear_data_layers
from src.gold import check_ollama

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


def render_metadata_editor(result: dict):
    meta = result.get("metadata", {})
    with st.form("metadata_review_form"):
        st.markdown("### Review metadata")

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
        keywords = st.text_area("Keywords, comma-separated", value=", ".join(meta.get("keywords") or []), height=90)

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
        meta["keywords"] = [x.strip() for x in keywords.split(",") if x.strip()]
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
        st.session_state.result = result

        doc_id = result.get("document_id", "reviewed")
        out = DATA_DIR / "gold" / f"{doc_id}_reviewed_result.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=4, ensure_ascii=False), encoding="utf-8")
        st.session_state.review_saved = True
        st.success(f"Reviewed result saved to {out}")


def render_result(result: dict):
    meta = result.get("metadata", {})
    quality = result.get("quality", {})
    stats = result.get("statistics", {})

    st.subheader(meta.get("title") or "Untitled document")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Language", meta.get("language") or result.get("language") or "unknown")
    m2.metric("Words", quality.get("word_count") or stats.get("main_text_words") or "—")
    m3.metric("Chunks", quality.get("chunk_count") or stats.get("chunk_count") or "—")
    m4.metric("Model", result.get("model") or "—")

    tab_summary, tab_meta, tab_terms, tab_entities, tab_json = st.tabs(
        ["Summary", "Metadata", "Top terms", "Suggested entities", "JSON"]
    )

    with tab_summary:
        st.markdown("### Summary")
        st.write(result.get("document_summary") or result.get("summary") or "No summary generated.")
        value = result.get("possible_value_for_knowledge_platform")
        if value:
            st.markdown("### Possible value for KMP")
            st.write(value)

    with tab_meta:
        render_metadata_editor(result)

    with tab_terms:
        st.markdown("### Top terms")
        terms_as_chips(top_term_strings(result))

        if result.get("top_terms"):
            for item in result.get("top_terms", []):
                if isinstance(item, dict):
                    with st.container(border=True):
                        st.markdown(f"**{item.get('rank', '')}. {item.get('term', '')}**")
                        if item.get("context"):
                            st.write(item.get("context"))
                        if item.get("evidence"):
                            st.caption(str(item.get("evidence")))

    with tab_entities:
        st.markdown("### Suggested possible entities")
        st.caption("These are suggestions only, not final metadata.")
        entities = result.get("suggested_entities") or {}
        if not entities:
            st.write("No entity suggestions.")
        for group, values in entities.items():
            st.markdown(f"**{group}**")
            if not values:
                st.write("—")
            else:
                if isinstance(values, list):
                    shown = []
                    for v in values:
                        if isinstance(v, dict):
                            shown.append(v.get("text") or v.get("term") or str(v))
                        else:
                            shown.append(str(v))
                    terms_as_chips(shown)
                else:
                    st.write(values)

    with tab_json:
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

                result = load_result(info["gold_json_path"])
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
