# streamlit_app.py

import pandas as pd
import streamlit as st

import retrival_system as rs  # your retrival_system.py


st.set_page_config(page_title="Multilingual Retrieval System", layout="wide")
st.title("🌍 Multilingual Retrieval System (Chroma + Built-in English Translations)")

st.markdown(
    """
Upload a CSV with columns **`doc_id`**, **`lang`**, **`text`**, and **`en_translation`**,  
then type a query in *any* language to search across all documents.

The app will show:
- the original text, and  
- the **English translation** from the dataset (`en_translation` column, no external API).
"""
)

# -----------------------------
# Session state
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs_df" not in st.session_state:
    st.session_state.docs_df = None


# -----------------------------
# Sidebar: upload + settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="CSV must contain columns: doc_id, lang, text, en_translation",
    )

    top_k = st.slider("Top K results", min_value=1, max_value=20, value=5, step=1)

    build_button = st.button("🔄 Build / Rebuild Vector Store")


# -----------------------------
# Build vectorstore
# -----------------------------
def build_vectorstore_from_df(df: pd.DataFrame):
    required_cols = {"doc_id", "lang", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        return None

    df = df.copy()
    df["text"] = df["text"].astype(str).map(rs.preprocess_text)

    with st.spinner("Building vector store (this may take a while for large datasets)..."):
        vectorstore = rs.insert_docs(df)

    st.success(f"Vector store built with {len(df)} documents.")
    return vectorstore, df


if build_button:
    if uploaded_file is None:
        st.warning("Please upload a CSV file first.")
    else:
        try:
            df_uploaded = pd.read_csv(uploaded_file)

            # Check for translation column
            if "en_translation" not in df_uploaded.columns:
                st.warning(
                    "Column `en_translation` not found. "
                    "Translations will not be shown unless this column exists."
                )

            vs, df_preprocessed = build_vectorstore_from_df(df_uploaded)
            if vs is not None:
                st.session_state.vectorstore = vs
                st.session_state.docs_df = df_uploaded  # keep original with en_translation
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")


st.markdown("---")

# -----------------------------
# Main search UI
# -----------------------------
if st.session_state.vectorstore is None:
    st.info("👈 Upload a CSV and click **Build / Rebuild Vector Store** to get started.")
else:
    st.subheader("🔍 Search")

    query = st.text_input(
        "Enter your query (any language)",
        placeholder="مثال: مشین لرننگ سفارشاتی نظام میں کیسے استعمال ہوتی ہے؟",
    )

    if query:
        if st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                df_res = rs.search_multilingual(
                    st.session_state.vectorstore,
                    query,
                    top_k=top_k,
                )

            if df_res.empty:
                st.warning("No results found.")
            else:
                # -----------------------------
                # Attach English translations (if available in original df)
                # -----------------------------
                docs_df = st.session_state.docs_df

                if docs_df is not None and "en_translation" in docs_df.columns:
                    # Ensure doc_id types align for merge
                    try:
                        # align dtype if possible
                        docs_df_merge = docs_df[["doc_id", "en_translation"]].copy()
                        if docs_df_merge["doc_id"].dtype != df_res["doc_id"].dtype:
                            docs_df_merge["doc_id"] = docs_df_merge["doc_id"].astype(
                                df_res["doc_id"].dtype
                            )

                        df_res = df_res.merge(
                            docs_df_merge,
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

                # -----------------------------
                # Build display table
                # -----------------------------
                df_display = df_res.copy()
                df_display["text_preview"] = df_display["text"].astype(str).str.slice(0, 180) + "..."
                if "en_translation" in df_display.columns:
                    df_display["en_translation_preview"] = (
                        df_display["en_translation"].astype(str).str.slice(0, 180) + "..."
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

                st.markdown("### 📄 Results (original + English translation from dataset)")
                st.dataframe(df_display, width='stretch')

                # -----------------------------
                # Full original + translation for top result
                # -----------------------------
                top_row = df_res.iloc[0]
                with st.expander("👑 Top result (full original + English translation)", expanded=True):
                    st.markdown(
                        f"**doc_id:** `{top_row['doc_id']}`, "
                        f"**lang:** `{top_row['lang']}`, "
                        f"**score:** `{top_row['score']:.4f}`"
                    )
                    st.markdown("#### 📝 Original text")
                    st.write(top_row["text"])

                    st.markdown("#### 🇬🇧 English translation (from `en_translation`)")
                    if pd.notna(top_row.get("en_translation")):
                        st.write(top_row["en_translation"])
                    else:
                        st.write("_No `en_translation` available for this document._")
