# KMP Streamlit Pipeline

This package contains Streamlit-compatible Python layers converted from the notebooks:

- Bronze: PDF text extraction
- Silver: generic document cleaning, body detection, sectioning, chunking
- Silver NLP: local keyword/entity suggestions, optional spaCy
- Gold: local Ollama document analysis
- Gold Meta: local Ollama metadata extraction
- Pipeline: orchestrates all layers
- app.py: Streamlit UI

## Run

```bash
pip install -r requirements.txt
ollama serve
ollama pull qwen2.5:3b-instruct
streamlit run app.py
```

For a bigger school server model, change the sidebar model to:

```text
qwen2.5:7b-instruct
```

or:

```text
qwen2.5:14b-instruct
```

Silver and Silver NLP do not use Ollama. Gold and Gold Meta do.
