import os
import streamlit as st
import json
import arxiv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# TEMPORARY: HuggingFace token for Streamlit Cloud — REMOVE before sharing repo!
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# --- Streamlit App ---
st.set_page_config(page_title="arXiv Research Chatbot", page_icon="📚")
st.title("📚 arXiv Research Chatbot")

# --- Step 1: Get Topic ---
topic = st.text_input("Enter a research topic to fetch papers from arXiv:")

@st.cache_data(show_spinner="🔍 Fetching papers from arXiv...")
def scrape_arxiv(query, max_results=10):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    results = []
    for paper in search.results():
        paper_data = {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": paper.summary,
            "published": paper.published.strftime("%Y-%m-%d"),
            "url": paper.entry_id
        }
        results.append(paper_data)
    return results

@st.cache_resource(show_spinner="🤖 Loading language model...")
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource(show_spinner="⚙️ Creating vectorstore...")
def build_vectorstore(papers):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = []
    for paper in papers:
        text = f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nPublished: {paper['published']}\n\nAbstract:\n{paper['summary']}"
        documents.append(Document(page_content=text.strip()))
    return FAISS.from_documents(documents, embeddings)

# --- Step 2: Fetch & Preprocess Papers ---
if topic:
    papers = scrape_arxiv(topic)
    st.success(f"✅ Fetched {len(papers)} papers for topic: {topic}")

    vectorstore = build_vectorstore(papers)
    llm = load_llm()

    # Create LangChain QA system
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # --- Step 3: Ask Questions ---
    st.divider()
    st.subheader("💬 Ask a question based on the papers")
    user_query = st.text_input("Your question:")

    if user_query:
        with st.spinner("Thinking..."):
            answer = qa.run(user_query)
        st.markdown("**Answer:**")
        st.write(answer)
