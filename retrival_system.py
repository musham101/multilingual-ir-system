import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import re

EMBED_MODEL = "snowflake-arctic-embed2"
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

def read_file(file_path):
    try:
        return pd.read_csv(file_path)
    except:
        return None

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def convert_text_to_doc(df):
    documents = []
    for _, row in df.iterrows():
        documents.append(
            Document(
                page_content=row["text"],
                metadata={
                    "doc_id": str(row["doc_id"]),
                    "lang": row["lang"],
                },
            )
        )
    return documents

def embed_query(query):
    return embeddings.embed_query(query)

def insert_docs(df, persist_dir="chroma_store"):
    documents = convert_text_to_doc(df)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    vectorstore.persist()
    return vectorstore

def search_multilingual(vectorstore, query, top_k=5):
    query_clean = preprocess_text(query)

    results = vectorstore.similarity_search_with_score(query_clean, k=top_k)

    rows = []
    for rank, (doc, score) in enumerate(results, start=1):
        rows.append({
            "rank": rank,
            "score": float(score),
            "doc_id": str(doc.metadata.get("doc_id")),
            "lang": doc.metadata.get("lang"),
            "text": doc.page_content,
        })

    return pd.DataFrame(rows)
