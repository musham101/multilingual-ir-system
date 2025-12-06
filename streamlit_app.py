# streamlit_app.py

import pandas as pd
import streamlit as st
import datetime
import retrival_system as rs   # your backend

# ================================
# Pastel Theme Colors
# ================================
PASTEL_PRIMARY = "#79AEC7"    # Pastel Blue (buttons)
PASTEL_SUCCESS = "#A8DADC"    # Mint Green (accents)
PASTEL_INFO = "#D1E2F0"       # Light Blue background
PASTEL_WARNING = "#FEECC8"    # Soft Peach
PASTEL_TEXT = "#333333"       # Dark grey

# ================================
# Streamlit Page Config
# ================================
st.set_page_config(
    page_title="Multilingual Retrieval System",
    layout="wide",
    page_icon="🌐"
)

# ================================
# Custom CSS
# ================================
st.markdown(f"""
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Title Style */
    .st-emotion-cache-1j02j5g {{
        color: {PASTEL_TEXT};
        font-weight: 700;
        border-bottom: 2px solid {PASTEL_INFO};
        padding-bottom: 15px;
    }}

    /* BUTTONS — pastel blue */
    .stButton > button {{
        background-color: {PASTEL_PRIMARY} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        transition: 0.3s ease-in-out;
    }}
    .stButton > button:hover {{
        background-color: #6B97AD !important; /* slightly darker pastel */
    }}

    /* Text input labels */
    div[data-testid="stTextInput"] label {{
        font-weight: bold;
        color: {PASTEL_TEXT};
        font-size: 1.1em;
    }}

    /* Success Block */
    div[data-testid="stSuccess"] {{
        background-color: {PASTEL_SUCCESS} !important;
        color: {PASTEL_TEXT};
        border-left: 5px solid #79A6A6 !important;
    }}

    /* Info Block */
    div[data-testid="stInfo"] {{
        background-color: {PASTEL_INFO} !important;
        color: {PASTEL_TEXT};
        border-left: 5px solid #6C9BB3 !important;
    }}

    /* Warning Block */
    div[data-testid="stWarning"] {{
        background-color: {PASTEL_WARNING} !important;
        color: {PASTEL_TEXT};
        border-left: 5px solid #F5C77E !important;
    }}
</style>
""", unsafe_allow_html=True)

# ================================
# SESSION STATE
# ================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "docs_df" not in st.session_state:
    st.session_state.docs_df = None
if "last_build_time" not in st.session_state:
    st.session_state.last_build_time = "Never"

# ================================
# BUILD VECTORSTORE FUNCTION
# ================================
def build_vectorstore(df: pd.DataFrame):
    required_cols = {"doc_id", "lang", "text"}
    missing = required_cols - set(df.columns)

    if missing:
        st.error(f"CSV missing required columns: {missing}")
        return None, None

    df = df.copy()
    df["text"] = df["text"].astype(str).map(rs.preprocess_text)

    with st.spinner("Building vectorstore..."):
        vs = rs.insert_docs(df, persist_dir="chroma_store")

    st.success(f"Vectorstore built with {len(df)} documents.")
    return vs, df


# ================================
# HEADER
# ================================
st.title("🌐 Multilingual Retrieval System")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
        > Upload a CSV (`doc_id`, `lang`, `text`, `en_translation`)  
        > Search in **any language**, retrieve multilingual documents.
    """)

with col2:
    st.subheader("System Status")

    if st.session_state.vectorstore is None:
        st.metric("Indexed Documents", "0")
        st.caption("Last Built: Never")
        st.progress(0)
    else:
        st.metric("Indexed Documents", len(st.session_state.docs_df))
        st.caption(f"Last Built: {st.session_state.last_build_time}")
        st.progress(100)

st.markdown("---")

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("🛠️ Data & Settings")

    with st.expander("⬆ Upload & Build", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"]
        )
        build_btn = st.button("🏗 Build / Rebuild Vectorstore")

    top_k = st.slider(
        "Top K results",
        min_value=1, max_value=20,
        value=5
    )


# ================================
# HANDLE BUILD BUTTON
# ================================
if build_btn:
    if uploaded_file is None:
        st.warning("Please upload a CSV first.")
    else:
        try:
            df_u = pd.read_csv(uploaded_file)
            vs, df_clean = build_vectorstore(df_u)

            if vs is not None:
                st.session_state.vectorstore = vs
                st.session_state.docs_df = df_u
                st.session_state.last_build_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.rerun()

        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ================================
# SEARCH INTERFACE
# ================================
st.subheader("🔎 Search Interface")

if st.session_state.vectorstore is None:
    st.info("Upload data and build vectorstore to start searching.")
else:
    col_input, col_btn = st.columns([4, 1])

    with col_input:
        query = st.text_input(
            "Enter query (any language)",
            placeholder="مثال: مشین لرننگ کیا ہے؟"
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("Search")

    if search_btn and query.strip():
        try:
            with st.spinner("Searching..."):
                df_res = rs.search_multilingual(
                    st.session_state.vectorstore,
                    query,
                    top_k=top_k
                )

            if df_res.empty:
                st.warning("No results found.")
            else:
                # 🔄 Fix doc_id type mismatch for merging
                df_res["doc_id"] = df_res["doc_id"].astype(str)
                docs_df = st.session_state.docs_df.copy()
                docs_df["doc_id"] = docs_df["doc_id"].astype(str)

                # Try merging translation
                if "en_translation" in docs_df.columns:
                    df_res = df_res.merge(
                        docs_df[["doc_id", "en_translation"]],
                        on="doc_id",
                        how="left"
                    )
                else:
                    df_res["en_translation"] = None

                # Display table
                st.subheader(f"Top {top_k} Results")
                display_df = df_res.copy()
                display_df["Preview"] = display_df["text"].str[:150] + "..."

                st.dataframe(display_df[["rank", "score", "doc_id", "lang", "Preview"]])

                # Best Match
                top_row = df_res.iloc[0]
                st.markdown("---")
                with st.expander("🥇 Best Match Details", expanded=True):
                    st.markdown(f"""
                        **Document ID:** `{top_row['doc_id']}`  
                        **Language:** `{top_row['lang']}`  
                        **Score:** `{top_row['score']:.4f}`
                    """)

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("### Original Text")
                        st.info(top_row["text"])

                    with col_b:
                        st.markdown("### English Translation")
                        if top_row["en_translation"] and str(top_row["en_translation"]) != "nan":
                            st.success(top_row["en_translation"])
                        else:
                            st.warning("_No translation available._")

        except Exception as e:
            st.error(f"Search failed: {e}")
