# Multilingual Retrieval System

Semantic Search, BM25 Keyword Search, and Hybrid Search
(Ollama + Chroma + Streamlit)

This project is a multilingual information retrieval system supporting:

* Semantic search using multilingual embeddings from Ollama
* BM25 keyword-based retrieval
* Hybrid retrieval that combines semantic and lexical scoring
* A Streamlit interface for interactive exploration
* A complete evaluation notebook comparing models, search modes, and performance

The system allows users to upload a multilingual dataset and retrieve documents using queries in any language.
If the dataset includes `en_translation`, both the original text and English translation are displayed.

A full evaluation notebook (`system_evaluation.ipynb`) is included.

---

## Project Structure

```
.
├── retrival_system.py            # Core search engine: Semantic, BM25, Hybrid
├── streamlit_app.py              # Streamlit application
├── system_evaluation.ipynb       # Evaluation of all models and retrieval modes
├── multilingual_dataset.csv      # Sample multilingual dataset
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Features

### 1. Semantic Search (Embedding-based)

Uses Ollama embedding models (default: `snowflake-arctic-embed2`) to compute multilingual vector embeddings and retrieve semantically relevant results.

### 2. BM25 Keyword Search

Implements classical term-frequency-based retrieval using BM25.
Useful for exact-match queries, rare terms, and lexical search baselines.

### 3. Hybrid Search (Semantic + BM25)

A weighted combination of semantic similarity and BM25 lexical scores:

```
hybrid_score = α * semantic_score  +  (1 - α) * bm25_score
```

α (alpha) controls the weighting and can be adjusted in the Streamlit UI.

### 4. Multilingual Query Support

Queries can be written in Urdu, English, Arabic, French, or any other language supported by the embedding model.

### 5. Optional English Translation Display

If the dataset includes an `en_translation` column, translations are shown alongside original text.

### 6. Full Model Evaluation Suite

The evaluation notebook compares:

* Multiple embedding models served by Ollama
* Semantic, BM25, and Hybrid retrieval
* Performance metrics: Recall@5, MRR@5, Latency

---

## Evaluation Results (from system_evaluation.ipynb)

Below are the measured metrics for all evaluated models and retrieval modes:
| Model / Mode                                  | Recall@5 | MRR@5 | Latency (s) |
| --------------------------------------------- | -------- | ----- | ----------- |
| BM25 (keyword)                                | 0.150    | 1.000 | 0.001       |
| snowflake-arctic-embed2:latest (Semantic)     | 0.500    | 1.000 | 0.168       |
| snowflake-arctic-embed2:latest (Hybrid α=0.5) | 0.412    | 1.000 | 0.170       |
| nomic-embed-text:latest (Semantic)            | 0.212    | 1.000 | 0.042       |
| nomic-embed-text:latest (Hybrid α=0.5)        | 0.325    | 1.000 | 0.048       |
| mxbai-embed-large:latest (Semantic)           | 0.212    | 0.854 | 0.063       |
| mxbai-embed-large:latest (Hybrid α=0.5)       | 0.325    | 1.000 | 0.073       |
| snowflake-arctic-embed:latest (Semantic)      | 0.200    | 0.875 | 0.060       |
| snowflake-arctic-embed:latest (Hybrid α=0.5)  | 0.263    | 0.875 | 0.067       |
| embeddinggemma:latest (Semantic)              | 0.425    | 0.938 | 0.121       |
| embeddinggemma:latest (Hybrid α=0.5)          | 0.350    | 1.000 | 0.130       |


These results demonstrate:

* BM25 is extremely fast but weaker on semantic understanding.
* Semantic models vary significantly in quality and latency.
* Hybrid retrieval frequently improves Recall@5 over pure semantic search.
* snowflake-arctic-embed2 provides the strongest overall consistency.

---

## Requirements

* Python 3.9–3.11
* Ollama installed and running locally
* macOS, Linux, or Windows via WSL

---

## Installing Ollama

Download from:

[https://ollama.com](https://ollama.com)

Pull the embedding model:

```bash
ollama pull snowflake-arctic-embed2
```

Verify availability:

```bash
curl http://localhost:11434/api/tags
```

If needed, start the server:

```bash
ollama serve
```

---

## Installation

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit App

From the project root:

```bash
streamlit run streamlit_app.py
```

The UI will be available at:

```
http://localhost:8501
```

---

## Using the App

### 1. Upload Dataset

Your CSV must contain:

```
doc_id, lang, text
```

Optional:

```
en_translation
```

### 2. Build Indexes

Click "Build / Rebuild Indexes" to generate:

* Chroma vectorstore
* BM25 keyword index
* Hybrid retrieval components

### 3. Choose Search Mode

Options:

* Semantic
* BM25
* Hybrid (with adjustable alpha)

### 4. Perform Search

Enter any query in any language.

Outputs include:

* Rank and score
* Original text preview
* English translation preview
* Full content of the top-ranked document

---

## Evaluation Notebook: system_evaluation.ipynb

The notebook benchmarks all retrieval modes and embedding models.
It includes:

* Query set definition
* Relevance checks
* Recall@5 computation
* MRR@5 computation
* Latency measurement
* Consolidated comparison table

Modify `EVAL_QUERIES` to add your own test queries and ground-truth relevant document IDs.