import pandas as pd
import streamlit as st

import retrival_system as rs

st.set_page_config(page_title="Multilingual Retrieval System", layout="wide")
st.title("🌍 Multilingual Retrieval System")

st.markdown(
    """
Upload a CSV with columns **`doc_id`**, **`lang`**, **`text`**, and optionally **`en_translation`**.

Then choose a search mode:

- **Semantic** (multilingual embeddings + Chroma)  
- **BM25** (keyword-based)  
- **Hybrid** (combines both)
"""
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs_df" not in st.session_state:
    st.session_state.docs_df = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "bm25_meta" not in st.session_state:
    st.session_state.bm25_meta = None

with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain: doc_id, lang, text (and optionally en_translation)",
    )

    top_k = st.slider("Top K results", min_value=1, max_value=20, value=5, step=1)

    search_mode = st.selectbox(
        "Search mode",
        ["Semantic", "BM25", "Hybrid"],
    )

    alpha = 0.5
    if search_mode == "Hybrid":
        alpha = st.slider("Hybrid alpha (semantic vs BM25)", 0.0, 1.0, 0.5, 0.05)

    build_button = st.button("🔄 Build / Rebuild Indexes")


def build_indexes(df_uploaded):
    required_cols = {"doc_id", "lang", "text"}
    missing = required_cols - set(df_uploaded.columns)
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return None, None, None

    df_model = df_uploaded.copy()
    df_model["text"] = df_model["text"].astype(str).map(rs.preprocess_text)

    with st.spinner("Building vector and BM25 indexes..."):
        vectorstore = rs.insert_docs(df_model)
        bm25, bm25_meta = rs.build_bm25_index(df_model)

    st.success(f"Indexes built with {len(df_model)} documents.")
    return vectorstore, bm25, bm25_meta


if build_button:
    if uploaded_file is None:
        st.warning("Please upload a CSV file first.")
    else:
        try:
            df_uploaded = pd.read_csv(uploaded_file)

            if "en_translation" not in df_uploaded.columns:
                st.warning(
                    "Column `en_translation` not found. "
                    "Translations will not be shown unless this column exists."
                )

            vs, bm25, bm25_meta = build_indexes(df_uploaded)
            if vs is not None:
                st.session_state.vectorstore = vs
                st.session_state.docs_df = df_uploaded
                st.session_state.bm25 = bm25
                st.session_state.bm25_meta = bm25_meta
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

st.markdown("---")

if st.session_state.vectorstore is None or st.session_state.bm25 is None:
    st.info("👈 Upload a CSV and click **Build / Rebuild Indexes** to get started.")
else:
    st.subheader("🔍 Search")

    query = st.text_input(
        "Enter your query (any language)",
        placeholder="مثال: مشین لرننگ سفارشاتی نظام میں کیسے استعمال ہوتی ہے؟",
    )

    if query:
        if st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                mode = search_mode

                if mode == "Semantic":
                    df_res = rs.search_multilingual(
                        st.session_state.vectorstore,
                        query,
                        top_k=top_k,
                    )
                    df_res = df_res.rename(columns={"score": "score"})

                elif mode == "BM25":
                    df_res = rs.search_bm25(
                        st.session_state.bm25,
                        st.session_state.bm25_meta,
                        query,
                        top_k=top_k,
                    )
                    df_res = df_res.rename(columns={"bm25_score": "score"})

                else:
                    df_res = rs.search_hybrid(
                        st.session_state.vectorstore,
                        st.session_state.bm25,
                        st.session_state.bm25_meta,
                        query,
                        top_k=top_k,
                        alpha=alpha,
                    )
                    df_res = df_res.rename(columns={"hybrid_score": "score"})

            if df_res.empty:
                st.warning("No results found.")
            else:
                docs_df = st.session_state.docs_df

                if docs_df is not None and "en_translation" in docs_df.columns:
                    try:
                        docs_merge = docs_df[["doc_id", "en_translation"]].copy()
                        if docs_merge["doc_id"].dtype != df_res["doc_id"].dtype:
                            docs_merge["doc_id"] = docs_merge["doc_id"].astype(
                                df_res["doc_id"].dtype
                            )

                        df_res = df_res.merge(
                            docs_merge,
                            on="doc_id",
                            how="left",
                        )
                    except Exception as e:
                        st.warning(
                            f"Could not merge translations by doc_id (error: {e}). "
                            "Showing results without translations."
                        )
                        df_res["en_translation"] = None
                else:
                    df_res["en_translation"] = None

                df_display = df_res.copy()
                df_display["text_preview"] = (
                    df_display["text"].astype(str).str.slice(0, 180) + "..."
                )
                if "en_translation" in df_display.columns:
                    df_display["en_translation_preview"] = (
                        df_display["en_translation"]
                        .astype(str)
                        .str.slice(0, 180)
                        + "..."
                    )
                else:
                    df_display["en_translation_preview"] = "[no translation]"

                df_display = df_display[
                    [
                        "rank",
                        "score",
                        "doc_id",
                        "lang",
                        "text_preview",
                        "en_translation_preview",
                    ]
                ]

                st.markdown(
                    f"### 📄 Results ({search_mode} mode, original + English translation if available)"
                )
                st.dataframe(df_display, use_container_width=True)

                top_row = df_res.iloc[0]
                with st.expander("👑 Top result (full original + English translation)", expanded=True):
                    st.markdown(
                        f"**doc_id:** `{top_row['doc_id']}`, "
                        f"**lang:** `{top_row['lang']}`, "
                        f"**score:** `{top_row['score']:.4f}`, "
                        f"**mode:** `{search_mode}`"
                    )
                    st.markdown("#### 📝 Original text")
                    st.write(top_row["text"])

                    st.markdown("#### 🇬🇧 English translation")
                    if pd.notna(top_row.get("en_translation")):
                        st.write(top_row["en_translation"])
                    else:
                        st.write("_No `en_translation` available for this document._")