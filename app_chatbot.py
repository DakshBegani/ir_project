import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return db

#local llm
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource
def setup_qa():
    db = load_vectorstore()
    llm = load_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    return qa

qa = setup_qa()

st.set_page_config(page_title="arXiv Research Chatbot", page_icon="📚")
st.title("📚 Ask Research Questions")
st.caption("Talk to the latest arXiv papers you scraped!")

query = st.text_input("Ask a question about the research papers:")
if st.button("Submit") and query.strip():
    with st.spinner("Thinking..."):
        result = qa.run(query)
        st.markdown("### 📖 Answer")
        st.write(result)
