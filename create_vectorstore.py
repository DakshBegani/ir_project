import json
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def load_chunks(path="data/preprocessed_chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return [Document(page_content=chunk) for chunk in chunks]

def create_vectorstore(documents, persist_path="vectorstore/"):
    os.makedirs(persist_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(persist_path)
    print(f"✅ Vector store saved to {persist_path}")

if __name__ == "__main__":
    docs = load_chunks()
    create_vectorstore(docs)
