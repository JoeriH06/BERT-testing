# Automating Knowledge Extraction for the Kennis Management Platform (KMP)

## Overview

This repository contains the Proof of Concept (PoC) developed during an internship project at HZ University of Applied Sciences for the Kennis Management Platform (KMP).

The goal of this project is to automate parts of the document upload workflow by extracting meaningful information from PDF documents using Natural Language Processing (NLP) and Large Language Models (LLMs).

The system processes Dutch and English PDF documents and generates:

- Document summaries
- Metadata
- Keywords
- Top terms
- Suggested entities
- Research topics
- Knowledge platform upload suggestions

The project was designed as an experimental and extensible NLP pipeline that can support future integration within the KMP environment.

---

# Project Objectives

The main objective of this project is to reduce manual work during the upload process by automatically generating structured information from unstructured documents.

Key research areas included:

- PDF preprocessing
- Document cleaning and normalization
- Metadata extraction
- Automatic summarization
- Local LLM deployment
- Evaluation of NLP models
- Knowledge platform integration

---

# Architecture

The solution follows a Medallion-inspired architecture consisting of multiple processing layers:

```text
PDF Upload
    │
    ▼
Bronze Layer
    │
    ▼
Silver Layer
    │
    ▼
Silver NLP Layer
    │
    ▼
Gold Layer
    │
    ▼
Gold Metadata Layer
    │
    ▼
Streamlit Application
```

---

# Processing Layers

## Bronze

Responsible for:

- PDF ingestion
- Raw text extraction
- Document storage

Output:
- Raw document text

---

## Silver

Responsible for:

- Text cleaning
- Header removal
- Footer removal
- Page number removal
- Table of contents detection
- Reference detection
- Text normalization
- Chunk preparation

Output:
- Cleaned document content

---

## Silver NLP

Responsible for:

- Keyword extraction
- Top term extraction
- Named Entity Recognition (NER)
- Language analysis
- Research topic detection
- Contributor candidates
- Organization detection

Output:
- NLP enrichment data

---

## Gold

Responsible for:

- Document summarization
- Topic identification
- Knowledge extraction
- Evaluation metrics

Generated outputs:

- Summary
- Main topics
- Results and conclusions
- Suggested entities
- Knowledge platform value

Evaluation metrics:

- Runtime
- Compression ratio
- Entity preservation
- Coverage

---

## Gold Metadata

Responsible for:

- Metadata extraction
- KMP field mapping
- Contributor extraction
- Keyword generation
- Confidence scoring
- Human review recommendations

Generated metadata includes:

- Title
- Contributors
- Language
- Keywords
- Description
- Research topic
- Research question
- Document type
- Contact information

---

# Technologies Used

## NLP

- spaCy
- HuggingFace Transformers
- KeyBERT
- SentenceTransformers

## LLMs

- Ollama
- Qwen 2.5
- Gemini (evaluation phase)

## Application

- Streamlit

## PDF Processing

- PyMuPDF
- pdfplumber

## Data Processing

- Python
- Pandas
- NumPy

---

# Model Experiments

During the project multiple models were evaluated:

- BERT
- DistilBART
- BART
- mBART
- Gemini Flash 2.0
- Qwen 2.5

Evaluation criteria included:

- Runtime performance
- Compression ratio
- Entity preservation
- Sentence coverage
- Metadata quality
- User feedback

Qwen 2.5 was selected as the primary local model due to its balance between performance, output quality, privacy, and deployment feasibility.

---

# Streamlit Application

The Streamlit interface allows users to:

- Upload PDF documents
- View extracted summaries
- View metadata
- Inspect top terms
- Review suggested entities
- Explore generated JSON outputs

The application serves as a demonstration of the Proof of Concept.

---

# Current Limitations

The following challenges remain:

- Heterogeneous document structures
- Contributor extraction reliability
- Title extraction reliability
- Metadata confidence validation
- Limited evaluation dataset
- Human review workflow

Document layouts vary significantly between uploads, making fully automated metadata extraction difficult.

---

# Future Improvements

Potential future developments include:

- Human-in-the-loop validation
- Metadata confidence scoring
- Ontology-based topic classification
- Improved contributor extraction
- Server deployment within HZ infrastructure
- Larger evaluation datasets
- GPU-supported models
- Multimodal document analysis

---

# Research Context

This repository was developed as part of the HBO-ICT Data Science internship at:

**HZ University of Applied Sciences**

Project:

**Automating Knowledge Extraction for the Kennis Management Platform Using Natural Language Processing**

---

# Disclaimer

This repository contains a Proof of Concept developed for research and experimentation purposes.

The system demonstrates the feasibility of applying NLP and LLM techniques to support document ingestion and metadata generation within a knowledge management environment.

The current implementation should not be considered production-ready without additional validation, testing, and integration work.

---

# Author

Joeri Hage

HZ University of Applied Sciences

Data Science Track