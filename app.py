import base64
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import streamlit.components.v1 as components

from src.pipeline import run_pipeline
from src.gold import load_gold_models

import fitz  # PyMuPDF
from PIL import Image
import io

st.set_page_config(
    page_title="PDF Intelligence Studio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

RAW_DIR = Path("Data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

        
def clear_data_layer(keep_raw_filename: str | None = None):
    """Remove previous pipeline inputs/outputs so the Data folder contains only the active document."""
    data_root = Path("Data")
    folders = ["raw", "bronze", "silver", "silver_nlp", "gold_meta", "gold"]

    for folder_name in folders:
        folder = data_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        for item in folder.iterdir():
            if folder_name == "raw" and keep_raw_filename and item.name == keep_raw_filename:
                continue

            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item)


def file_signature(file_name: str, file_bytes: bytes) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    return f"{file_name}:{len(file_bytes)}:{digest}"


st.markdown(
    """
    <style>
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 1.2rem;
        background: linear-gradient(135deg, rgba(90, 120, 255, .14), rgba(0, 200, 150, .12));
        border: 1px solid rgba(120, 120, 120, .18);
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin-bottom: .2rem; }
    .metric-card {
        padding: 1rem;
        border-radius: 1rem;
        border: 1px solid rgba(120, 120, 120, .18);
        background: rgba(250, 250, 250, .035);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_gold_resources():
    return load_gold_models()


def load_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def render_pdf_preview(pdf_path: Path):
    st.subheader("Uploaded document preview")

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        st.warning("No uploaded PDF found.")
        return

    with st.expander("PDF preview", expanded=True):
        zoom = st.slider("Zoom", 1.0, 3.0, 1.4, 0.1)
        max_pages = st.number_input("Pages to preview", 1, 20, 5)

        doc = fitz.open(pdf_path)

        for page_number in range(min(len(doc), int(max_pages))):
            page = doc[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            st.caption(f"Page {page_number + 1}")
            st.image(img, use_container_width=True)

        doc.close()

    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download uploaded PDF",
            data=f,
            file_name=pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )

def init_session_state():
    defaults = {
        "result_data": None,
        "pipeline_result": None,
        "review_banner_visible": False,
        "review_saved": False,
        "edited_metadata": None,
        "edited_summary": "",
        "uploaded_pdf_path": None,
        "uploaded_pdf_name": "",
        "uploaded_pdf_signature": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_review_state():
    st.session_state.review_banner_visible = False
    st.session_state.review_saved = False
    st.session_state.edited_metadata = None
    st.session_state.edited_summary = ""


def normalize_terms(value):
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def get_display_metadata(data: dict) -> Dict[str, Any]:
    metadata = data.get("metadata", {})

    contributors = metadata.get("contributors", [])
    contributors_text = ", ".join(contributors) if isinstance(contributors, list) else str(contributors or "")

    research_group = metadata.get("research_group")
    if not research_group:
        top3 = metadata.get("research_group_top3", [])
        research_group = top3[0]["label"] if top3 else ""

    return {
        "document_id": metadata.get("id") or data.get("document_id") or "",
        "title": metadata.get("title") or "",
        "subtitle": metadata.get("subtitle") or "",
        "contributors": contributors_text,
        "research_group": research_group or "",
        "publication_date": metadata.get("publication_date") or "",
        "start_date": metadata.get("start_date") or "",
        "end_date": metadata.get("end_date") or "",
        "model": data.get("model", "Unknown model"),
        "top_terms": normalize_terms(metadata.get("top_terms", [])),
    }


def get_current_metadata_for_display() -> Dict[str, Any]:
    if st.session_state.edited_metadata:
        return st.session_state.edited_metadata
    if st.session_state.result_data:
        return get_display_metadata(st.session_state.result_data)
    return {
        "document_id": "",
        "title": "",
        "subtitle": "",
        "contributors": "",
        "research_group": "",
        "publication_date": "",
        "start_date": "",
        "end_date": "",
        "model": "",
        "top_terms": [],
    }


def get_current_summary_for_display() -> str:
    if st.session_state.edited_summary:
        return st.session_state.edited_summary
    if st.session_state.result_data:
        return st.session_state.result_data.get("summary", "")
    return ""


def save_reviewed_metadata():
    if not st.session_state.result_data:
        return

    contributors = [c.strip() for c in st.session_state.get("edit_contributors", "").split(",") if c.strip()]
    top_terms = [t.strip() for t in st.session_state.get("edit_top_terms", "").split(",") if t.strip()]

    edited_metadata = {
        "document_id": st.session_state.get("edit_document_id", "").strip(),
        "title": st.session_state.get("edit_title", "").strip(),
        "subtitle": st.session_state.get("edit_subtitle", "").strip(),
        "contributors": ", ".join(contributors),
        "research_group": st.session_state.get("edit_research_group", "").strip(),
        "publication_date": st.session_state.get("edit_publication_date", "").strip(),
        "start_date": st.session_state.get("edit_start_date", "").strip(),
        "end_date": st.session_state.get("edit_end_date", "").strip(),
        "model": st.session_state.get("edit_model", "").strip(),
        "top_terms": top_terms,
    }

    result_data = st.session_state.result_data
    result_data["document_id"] = edited_metadata["document_id"]
    result_data["summary"] = st.session_state.get("edit_summary", "").strip()
    result_data.setdefault("metadata", {})
    result_data["metadata"].update({
        "id": edited_metadata["document_id"],
        "title": edited_metadata["title"],
        "subtitle": edited_metadata["subtitle"],
        "contributors": contributors,
        "research_group": edited_metadata["research_group"],
        "publication_date": edited_metadata["publication_date"],
        "start_date": edited_metadata["start_date"],
        "end_date": edited_metadata["end_date"],
        "top_terms": top_terms,
    })

    st.session_state.result_data = result_data
    st.session_state.edited_metadata = edited_metadata
    st.session_state.edited_summary = result_data["summary"]
    st.session_state.review_saved = True
    st.session_state.review_banner_visible = False


def render_review_panel():
    if not st.session_state.review_banner_visible or not st.session_state.result_data:
        return

    display = get_display_metadata(st.session_state.result_data)
    summary = st.session_state.result_data.get("summary", "")

    with st.container(border=True):
        left, right = st.columns([11, 1])
        with left:
            st.warning("Review the extracted metadata. Edit only the fields you want to change, then save.")
        with right:
            if st.button("✕", key="dismiss_review_banner"):
                st.session_state.review_banner_visible = False
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Document ID", value=display["document_id"], key="edit_document_id")
            st.text_input("Title", value=display["title"], key="edit_title")
            st.text_input("Subtitle", value=display["subtitle"], key="edit_subtitle")
            st.text_input("Research Group", value=display["research_group"], key="edit_research_group")
        with col2:
            st.text_input("Publication Date", value=display["publication_date"], key="edit_publication_date")
            st.text_input("Start Date", value=display["start_date"], key="edit_start_date")
            st.text_input("End Date", value=display["end_date"], key="edit_end_date")
            st.text_input("Model", value=str(display["model"]), key="edit_model", disabled=True)

        st.text_area("Contributors (comma-separated)", value=display["contributors"], height=80, key="edit_contributors")
        st.text_area("Top Terms (comma-separated)", value=", ".join(display["top_terms"]), height=80, key="edit_top_terms")
        st.text_area("Summary", value=summary, height=180, key="edit_summary")

        save_col, dismiss_col = st.columns([1, 1])
        with save_col:
            if st.button("Save reviewed metadata", type="primary", use_container_width=True):
                save_reviewed_metadata()
                st.rerun()
        with dismiss_col:
            if st.button("Dismiss without saving", use_container_width=True):
                st.session_state.review_banner_visible = False
                st.rerun()


def render_dashboard():
    data = st.session_state.result_data
    if not data:
        return

    display = get_current_metadata_for_display()
    top_terms = normalize_terms(display.get("top_terms", []))
    summary = get_current_summary_for_display()
    pipeline = st.session_state.pipeline_result or {}
    quality = pipeline.get("quality_report", {})

    st.subheader("Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Language", pipeline.get("language", data.get("language", "unknown")))
    m2.metric("Words", quality.get("word_count", "—"))
    m3.metric("Chunks", quality.get("chunk_count", "—"))
    m4.metric("Ready", "Yes" if quality.get("ready_for_modeling") else "Check")

    tab_summary, tab_metadata, tab_terms, tab_json = st.tabs(["Executive summary", "Metadata", "Top terms", "JSON"])

    with tab_summary:
        st.markdown("### Summary")
        st.write(summary or "No summary generated.")
        st.download_button(
            "Download reviewed result JSON",
            data=json.dumps(st.session_state.result_data, indent=2, ensure_ascii=False),
            file_name=f"{display.get('document_id', 'result')}_reviewed.json",
            mime="application/json",
        )

    with tab_metadata:
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Document ID", value=display.get("document_id") or "Not found", disabled=True)
            st.text_input("Title", value=display.get("title") or "Not found", disabled=True)
            st.text_input("Subtitle", value=display.get("subtitle") or "Not found", disabled=True)
            st.text_input("Research Group", value=display.get("research_group") or "Not found", disabled=True)
        with col2:
            st.text_input("Publication Date", value=display.get("publication_date") or "Not found", disabled=True)
            st.text_input("Start Date", value=display.get("start_date") or "Not found", disabled=True)
            st.text_input("End Date", value=display.get("end_date") or "Not found", disabled=True)
            st.text_input("Model", value=str(display.get("model") or "Unknown model"), disabled=True)
        st.text_area("Contributors", value=display.get("contributors") or "Not found", height=90, disabled=True)

    with tab_terms:
        if top_terms:
            st.markdown(" ".join([f"`{term}`" for term in top_terms]))
        else:
            st.write("No terms found.")

    with tab_json:
        st.json(data)


init_session_state()

st.markdown(
    """
    <div class="hero">
      <h1>📄 PDF Intelligence Studio</h1>
      <p>Upload a PDF, run the bronze → silver → NLP → gold pipeline, then review the extracted metadata and summary.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Pipeline")
    st.caption("The uploaded PDF is saved to Data/raw, then processed into the Data layer folders.")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    run_clicked = st.button("Run complete pipeline", type="primary", disabled=uploaded_file is None, use_container_width=True)

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    signature = file_signature(uploaded_file.name, uploaded_bytes)
    pdf_path = RAW_DIR / uploaded_file.name

    if signature != st.session_state.uploaded_pdf_signature:
        clear_data_layer(keep_raw_filename=uploaded_file.name)
        reset_review_state()
        st.session_state.result_data = None
        st.session_state.pipeline_result = None
        st.session_state.uploaded_pdf_signature = signature

    with open(pdf_path, "wb") as f:
        f.write(uploaded_bytes)

    st.session_state.uploaded_pdf_path = str(pdf_path)
    st.session_state.uploaded_pdf_name = uploaded_file.name

    st.info(f"Ready to process: **{uploaded_file.name}**")

    preview_tab, workspace_tab = st.tabs(["PDF preview", "Extraction workspace"])

    with preview_tab:
        st.subheader("Uploaded document preview")
        render_pdf_preview(Path(st.session_state.uploaded_pdf_path))

    with workspace_tab:
        st.caption("Run the pipeline here, then review the generated metadata and summary below.")

        if run_clicked:
            try:
                clear_data_layer(keep_raw_filename=uploaded_file.name)
                reset_review_state()

                with st.spinner("Preparing resources..."):
                    gold_resources = get_gold_resources()

                with st.spinner("Running bronze, silver, silver NLP, gold metadata, and gold summary layers..."):
                    pipeline_result = run_pipeline(str(pdf_path), gold_resources=gold_resources)

                result_data = load_json(pipeline_result["gold_json_path"])

                st.session_state.pipeline_result = pipeline_result
                st.session_state.result_data = result_data
                st.session_state.review_banner_visible = True

                st.success("Pipeline completed successfully.")
                st.rerun()

            except Exception as e:
                st.error(f"Pipeline failed: {e}")

elif st.session_state.uploaded_pdf_path:
    st.subheader("Uploaded document preview")
    render_pdf_preview(st.session_state.uploaded_pdf_path)

render_review_panel()

if st.session_state.review_saved:
    st.success("Reviewed metadata saved.")

render_dashboard()
