import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import re

EMBED_MODEL = "snowflake-arctic-embed2"

embeddings = OllamaEmbeddings(model=EMBED_MODEL)


def read_file(file_path):
    try:
        return pd.read_csv(file_path)
    except:
        return None


def preprocess_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def convert_text_to_doc(df):
    docs = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=row["text"],
                metadata={"doc_id": row["doc_id"], "lang": row["lang"]},
            )
        )
    return docs


def insert_docs(df):
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    docs = convert_text_to_doc(df)
    vectorstore = Chroma.from_documents(documents=docs, embedding=emb)
    return vectorstore


def search_multilingual(vectorstore, query, top_k=5):
    q = preprocess_text(query)
    results = vectorstore.similarity_search_with_score(q, k=top_k)

    rows = []
    for rank, (doc, score) in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "score": float(score),
                "doc_id": doc.metadata.get("doc_id"),
                "lang": doc.metadata.get("lang"),
                "text": doc.page_content,
            }
        )
    return pd.DataFrame(rows)


def tokenize(text):
    text = preprocess_text(text).lower()
    return text.split()


def build_bm25_index(df):
    meta = df[["doc_id", "lang", "text"]].copy().reset_index(drop=True)
    corpus = meta["text"].astype(str).tolist()
    tokenized = [tokenize(t) for t in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25, meta


def search_bm25(bm25, meta, query, top_k=5):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)

    tmp = meta.copy()
    tmp["bm25_score"] = scores
    tmp = tmp.sort_values("bm25_score", ascending=False).head(top_k).reset_index(drop=True)
    tmp.insert(0, "rank", tmp.index + 1)

    return tmp[["rank", "bm25_score", "doc_id", "lang", "text"]]


def search_hybrid(vectorstore, bm25, meta, query, top_k=5, alpha=0.5):
    df_sem = search_multilingual(vectorstore, query, top_k=top_k * 2)
    df_sem = df_sem.rename(columns={"score": "sem_score"})

    df_bm = search_bm25(bm25, meta, query, top_k=top_k * 2)

    df = pd.merge(
        df_sem[["doc_id", "lang", "text", "sem_score"]],
        df_bm[["doc_id", "lang", "text", "bm25_score"]],
        on="doc_id",
        how="outer",
        suffixes=("_sem", "_bm"),
    )

    df["lang"] = df["lang_sem"].combine_first(df["lang_bm"])
    df["text"] = df["text_sem"].combine_first(df["text_bm"])

    sem_max = df["sem_score"].max()
    bm_max = df["bm25_score"].max()

    df["sem_norm"] = 0.0
    df["bm25_norm"] = 0.0

    if pd.notna(sem_max) and sem_max != 0:
        df["sem_norm"] = 1 - (df["sem_score"] / sem_max)

    if pd.notna(bm_max) and bm_max != 0:
        df["bm25_norm"] = df["bm25_score"] / bm_max

    df["hybrid_score"] = alpha * df["sem_norm"] + (1 - alpha) * df["bm25_norm"]

    df = df.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    return df[["rank", "hybrid_score", "doc_id", "lang", "text"]]
