import json
from pathlib import Path
from typing import Dict

import streamlit as st

from src.pipeline import run_pipeline
from src.gold import load_gold_models


st.set_page_config(page_title="PDF Metadata Dashboard", layout="wide")

RAW_DIR = Path("Data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_gold_resources():
    return load_gold_models()


def load_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_session_state():
    defaults = {
        "result_data": None,
        "pipeline_result": None,
        "review_banner_visible": False,
        "review_saved": False,
        "edited_metadata": None,
        "edited_summary": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_review_state():
    st.session_state.review_banner_visible = False
    st.session_state.review_saved = False
    st.session_state.edited_metadata = None
    st.session_state.edited_summary = ""


def get_display_metadata(data: dict) -> Dict[str, str]:
    metadata = data.get("metadata", {})

    title = metadata.get("title") or ""
    doc_id = metadata.get("id") or data.get("document_id") or ""

    contributors = metadata.get("contributors", [])
    contributors_text = ", ".join(contributors) if contributors else ""

    research_group = metadata.get("research_group")
    if not research_group:
        top3 = metadata.get("research_group_top3", [])
        research_group = top3[0]["label"] if top3 else ""

    start_date = metadata.get("start_date") or ""
    end_date = metadata.get("end_date") or ""
    model_name = data.get("model", "Unknown model")

    top_terms = metadata.get("top_terms", [])

    return {
        "document_id": doc_id,
        "title": title,
        "contributors": contributors_text,
        "research_group": research_group,
        "start_date": start_date,
        "end_date": end_date,
        "model": model_name,
        "top_terms": top_terms,
    }


def get_current_metadata_for_display() -> Dict[str, str]:
    if st.session_state.edited_metadata:
        return st.session_state.edited_metadata
    if st.session_state.result_data:
        return get_display_metadata(st.session_state.result_data)
    return {
        "document_id": "",
        "title": "",
        "contributors": "",
        "research_group": "",
        "start_date": "",
        "end_date": "",
        "model": "",
        "top_terms": "",
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

    contributors_raw = st.session_state.get("edit_contributors", "").strip()
    contributors = [c.strip() for c in contributors_raw.split(",") if c.strip()]

    top_terms_raw = st.session_state.get("edit_top_terms", "").strip()
    top_terms = [t.strip() for t in top_terms_raw.split(",") if t.strip()]

    edited_metadata = {
        "document_id": st.session_state.get("edit_document_id", "").strip(),
        "title": st.session_state.get("edit_title", "").strip(),
        "contributors": ", ".join(contributors),
        "research_group": st.session_state.get("edit_research_group", "").strip(),
        "start_date": st.session_state.get("edit_start_date", "").strip(),
        "end_date": st.session_state.get("edit_end_date", "").strip(),
        "model": st.session_state.get("edit_model", "").strip(),
        "top_terms": ", ".join(top_terms),
    }

    st.session_state.edited_metadata = edited_metadata
    st.session_state.edited_summary = st.session_state.get("edit_summary", "").strip()
    st.session_state.review_saved = True
    st.session_state.review_banner_visible = False

    result_data = st.session_state.result_data
    result_data["document_id"] = edited_metadata["document_id"]
    result_data["summary"] = st.session_state.edited_summary

    if "metadata" not in result_data:
        result_data["metadata"] = {}

    result_data["metadata"]["id"] = edited_metadata["document_id"]
    result_data["metadata"]["title"] = edited_metadata["title"]
    result_data["metadata"]["title_found"] = bool(edited_metadata["title"])
    result_data["metadata"]["contributors"] = contributors
    result_data["metadata"]["contributors_found"] = bool(contributors)
    result_data["metadata"]["research_group"] = edited_metadata["research_group"]
    result_data["metadata"]["start_date"] = edited_metadata["start_date"]
    result_data["metadata"]["end_date"] = edited_metadata["end_date"]
    result_data["metadata"]["top_terms"] = top_terms

    st.session_state.result_data = result_data


def dismiss_review_banner():
    st.session_state.review_banner_visible = False


def render_review_banner():
    if not st.session_state.review_banner_visible or not st.session_state.result_data:
        return

    display = get_display_metadata(st.session_state.result_data)
    summary = st.session_state.result_data.get("summary", "")

    with st.container(border=True):
        top_left, top_right = st.columns([10, 1])

        with top_left:
            st.warning("Review the suggested metadata below. Edit only the fields you want to change, then save.")
        with top_right:
            if st.button("✕", key="dismiss_review_banner", help="Dismiss review banner"):
                dismiss_review_banner()
                st.rerun()

        st.markdown("### Review and edit metadata")

        col1, col2 = st.columns(2)

        with col1:
            st.text_input(
                "Document ID",
                value=display["document_id"],
                key="edit_document_id",
            )
            st.text_input(
                "Title",
                value=display["title"],
                key="edit_title",
            )
            st.text_input(
                "Research Group",
                value=display["research_group"],
                key="edit_research_group",
            )
            st.text_area(
                "Top Terms (comma-separated)",
                value=display["top_terms"],
                height=100,
                key="edit_top_terms",
            )

        with col2:
            st.text_input(
                "Start Date",
                value=display["start_date"],
                key="edit_start_date",
            )
            st.text_input(
                "End Date",
                value=display["end_date"],
                key="edit_end_date",
            )
            st.text_input(
                "Model",
                value=display["model"],
                key="edit_model",
                disabled=True,
            )

        st.text_area(
            "Contributors (comma-separated)",
            value=display["contributors"],
            height=80,
            key="edit_contributors",
        )

        st.text_area(
            "Summary",
            value=summary,
            height=180,
            key="edit_summary",
        )

        btn1, btn2 = st.columns([1, 1])

        with btn1:
            if st.button("Save reviewed metadata", type="primary", key="save_reviewed_metadata"):
                save_reviewed_metadata()
                st.rerun()

        with btn2:
            if st.button("Dismiss without saving", key="dismiss_without_saving"):
                dismiss_review_banner()
                st.rerun()


def display_metadata(data: dict):
    display = get_current_metadata_for_display()

    title = display["title"] or "Not found"
    doc_id = display["document_id"] or "Not found"
    contributors_text = display["contributors"] or "Not found"
    research_group = display["research_group"] or "Not found"
    start_date = display["start_date"] or "Not found"
    
    end_date = display["end_date"] or "Not found"
    model_name = display["model"] or "Unknown model"
    top_terms = display["top_terms"]

    st.markdown("### Top Terms")

    if top_terms:
        st.write(", ".join(top_terms))
    else:
        st.write("Not found")

    st.subheader("Extracted Metadata")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Document ID", value=doc_id, disabled=True)
        st.text_input("Title", value=title, disabled=True)
        st.text_input("Research Group", value=research_group, disabled=True)

    with col2:
        st.text_input("Start Date", value=start_date, disabled=True)
        st.text_input("End Date", value=end_date, disabled=True)
        st.text_input("Model", value=model_name, disabled=True)

    st.text_area("Contributors", value=contributors_text, height=80, disabled=True)
    st.text_area("Top Terms", value=top_terms_text, height=100, disabled=True)

    top3 = data.get("metadata", {}).get("research_group_top3", [])
    if top3:
        st.markdown("### Research Group Candidates")
        for item in top3:
            label = item.get("label", "Unknown")
            score = item.get("score", 0)
            st.write(f"- {label}: {score:.3f}")


def display_summary(data: dict):
    summary = get_current_summary_for_display()
    if summary:
        st.markdown("### Summary")
        st.write(summary)


def display_json(data: dict):
    with st.expander("Show full model output JSON"):
        st.json(data)


init_session_state()

st.title("PDF Upload and Metadata Extraction Dashboard")
st.write("Upload a PDF, run the pipeline, and review extracted metadata from the mBART gold layer.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_path = RAW_DIR / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Run Pipeline"):
        try:
            reset_review_state()

            with st.spinner("Loading models..."):
                gold_resources = get_gold_resources()

            with st.spinner("Running bronze, silver, silver_nlp, and gold pipeline..."):
                pipeline_result = run_pipeline(str(pdf_path), gold_resources=gold_resources)

            result_data = load_json(pipeline_result["gold_json_path"])

            st.session_state.pipeline_result = pipeline_result
            st.session_state.result_data = result_data
            st.session_state.review_banner_visible = True

            st.success("Pipeline completed successfully.")
            st.rerun()

        except Exception as e:
            st.error(f"Pipeline failed: {e}")

render_review_banner()

if st.session_state.review_saved:
    st.success("Reviewed metadata saved.")

if st.session_state.result_data:
    display_metadata(st.session_state.result_data)
    display_summary(st.session_state.result_data)
    display_json(st.session_state.result_data)