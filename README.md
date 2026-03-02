# BERT-testing Manuel

## In this project we will recreate datasets ourself, consisting of JSON text files with random generated content and structure.
### we will make use of the spaCY library to do the pre-processing steps and we will use BERT models for further modification.


## Why spaCy?
spaCy provides reliable tokenization, sentence segmentation, and linguistic annotations that are well-suited for preparing text data for downstream NLP models such as transformer-based architectures.

---

## Processing Structure

The processing in this layer follows the steps below:

### 1. Parse and Select Fields (Structure Cleaning)
JSON blob structures are parsed and flattened into a consistent schema.  
Relevant text and metadata fields are selected, such as:
- document identifiers
- document type (e.g. project, media)
- titles and headings
- main body text
- contributor and date metadata

This step ensures structural consistency and traceability of the original source data.

---

### 2. Normalize and De-Noise Text (Light Text Cleaning)
Light text normalization is applied to improve downstream NLP performance while preserving semantic meaning:
- removal of duplicated content blocks
- removal of known boilerplate or UI-generated text
- stripping of HTML or markup when present
- normalization of whitespace and unicode characters

Aggressive text cleaning (e.g. global lowercasing, stop-word removal, or punctuation removal) is intentionally avoided, as it can negatively affect entity recognition and heading detection.

---

### 3. spaCy Annotation (NLP Preparation)
The cleaned text is processed using spaCy to generate linguistic annotations:
- tokenization
- sentence segmentation
- optional lemmatization
- named entity recognition (NER) for supporting downstream extraction

These annotations are used to prepare the data for transformer-based extraction in the Gold layer and are not considered final analytical outputs.

---

### 4. Gold Layer Extraction (Transformer-Based)
In the Gold layer, transformer models (e.g. BERT) are used to extract key content and structured information such as:
- important text segments
- contributors
- release or publication dates
- semantic representations for downstream analysis

spaCy annotations from the Silver layer are used as supporting signals during this process.

---

### 5. Summarization
In the final step, summarization is applied to the cleaned and content-selected text.  
Summaries are generated only after irrelevant and noisy content has been removed, ensuring that the output reflects the most relevant information in each document.

---

## Key Considerations
- Casing and punctuation are preserved to support heading detection and entity recognition.
- Numerical values are retained to allow extraction of dates and version information.
- Metadata fields from the original JSON are preferred when available, with NLP methods used as fallback.
- Both raw and cleaned text are retained to ensure reproducibility and traceability.