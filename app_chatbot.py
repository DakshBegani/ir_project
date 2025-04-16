import os
import streamlit as st
import arxiv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.set_page_config(page_title="🧠 arXiv Chatbot", page_icon="📚")
st.title("📚 arXiv Research Chatbot")

if "vectorstore" not in st.session_state:
    topic = st.text_input("Enter a research topic (e.g. 'transformers in NLP'):")

    if topic:
        with st.spinner("🔍 Scraping arXiv..."):
            search = arxiv.Search(query=topic, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate)
            papers = []
            for result in search.results():
                papers.append({
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "published": result.published.strftime("%Y-%m-%d")
                })
            st.session_state["papers"] = papers

        with st.spinner("📦 Creating vectorstore with tighter chunks..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            docs = []

            for paper in st.session_state["papers"]:
                full_text = f"Title: {paper['title']}\n\n{paper['summary']}"
                chunks = splitter.split_text(full_text)
                docs.extend([Document(page_content=chunk) for chunk in chunks])

            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state["vectorstore"] = vectorstore

        with st.spinner("🤖 Loading Mistral-7B via HuggingFaceHub..."):
            llm = HuggingFaceHub(
                repo_id="mistralai/Mistral-7B-Instruct-v0.1",
                model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
            )
            st.session_state["llm"] = llm

        retriever = st.session_state["vectorstore"].as_retriever()
        st.session_state["qa_chain"] = RetrievalQA.from_chain_type(
            llm=st.session_state["llm"],
            retriever=retriever
        )

        st.success("✅ Chatbot ready! Scroll down to ask questions.")

if "qa_chain" in st.session_state:
    st.divider()
    st.subheader("💬 Chat with the papers")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a question about the papers..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Show retrieved context
                retrieved_docs = st.session_state["vectorstore"].as_retriever().get_relevant_documents(user_input)
                st.markdown("**Context Retrieved:**")
                if not retrieved_docs:
                    st.warning("No relevant context retrieved. The answer may be inaccurate.")
                for doc in retrieved_docs:
                    st.code(doc.page_content[:300])

                # Run QA chain with better model
                result = st.session_state["qa_chain"].run(user_input)
                st.markdown("**Answer:**")
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
