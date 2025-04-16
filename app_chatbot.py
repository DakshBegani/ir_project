import os
import streamlit as st
import arxiv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.set_page_config(page_title="arXiv Research Chatbot", page_icon="🤖")
st.title("📚 arXiv Research Chatbot")

topic = st.text_input("Enter a research topic (e.g. 'graph neural networks'):")

@st.cache_data(show_spinner="🔍 Scraping arXiv...")
def fetch_papers(query):
    search = arxiv.Search(query=query, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    for result in search.results():
        paper = {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
        }
        papers.append(paper)
    return papers

@st.cache_resource(show_spinner="⚙️ Loading LLM...")
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_resource(show_spinner="📦 Building vectorstore...")
def build_vectorstore(papers):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = []
    for paper in papers:
        text = f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nDate: {paper['published']}\n\n{paper['summary']}"
        docs.append(Document(page_content=text))
    return FAISS.from_documents(docs, embeddings)

if topic:
    papers = fetch_papers(topic)
    st.success(f"✅ Retrieved {len(papers)} papers")

    vectorstore = build_vectorstore(papers)
    llm = load_llm()

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    st.divider()
    st.subheader("💬 Ask your question")
    query = st.text_input("Question about the papers:")
    if query:
        with st.spinner("Generating answer..."):
            response = qa.run(query)
        st.write
