import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

VECTOR_PATH = "vectorstore/"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

def ask(query: str):
    result = qa(query)
    return result["result"]

if __name__ == "__main__":
    while True:
        user_input = input("Ask a research question (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = ask(user_input)
        print("\nAnswer:", response)
