# 🌍 Multilingual Retrieval System (Ollama + Chroma + Streamlit)

This project is a **multilingual semantic search system** built using:

* **Ollama** for local embedding generation
* **Chroma** for vector storage
* **Streamlit** for an interactive UI

You can upload a multilingual dataset, build a vector store, and query in **any language**.
The system will show both:

* the **original text**, and
* the **English translation** (from the dataset column `en_translation`)

The repo also includes a sample dataset: **`multilingual_dataset.csv`**.

---

## 📁 Project Structure

```
.
├── retrival_system.py        # Core retrieval and embedding logic
├── streamlit_app.py          # Streamlit frontend
├── multilingual_dataset.csv  # Sample dataset for testing
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🧠 Requirements

* Python 3.9–3.11
* Ollama installed locally
* Works on macOS, Linux, and Windows (via WSL)

---

## ⚙️ Installing Ollama

1. Download & install from:
   **[https://ollama.com](https://ollama.com)**

2. Pull the embedding model used in this project:

```bash
ollama pull snowflake-arctic-embed2
```

3. Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response listing available models.

If Ollama isn’t running:

* **macOS:** Open the Ollama app
* **Linux/WSL:**

```bash
ollama serve
```

---

## 📦 Install Python Dependencies

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

### Install required packages

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit App

From the project root:

```bash
streamlit run streamlit_app.py
```

This will open the interface at:

```
http://localhost:8501
```

---

## 🧪 Using the App

### 1. Upload Dataset

Use **multilingual_dataset.csv** or your own dataset containing:

* `doc_id`
* `lang`
* `text`
* `en_translation`

### 2. Build Vector Store

Click **"🔄 Build / Rebuild Vector Store"**.
The system will embed all documents using `snowflake-arctic-embed2` and store them in Chroma.

### 3. Search

Enter any query in any language.
The app will show:

* Rank
* Similarity score
* Original text preview
* English translation preview
* Full text + translation for the top result

---

## ❗ Troubleshooting

### ❌ Error: “Model not found”

Run:

```bash
ollama pull snowflake-arctic-embed2
```

### ❌ Error: Cannot connect to Ollama

Start the Ollama server:

```bash
ollama serve
```

### ❌ Missing CSV columns

Ensure the dataset includes:

```
doc_id, lang, text, en_translation
```