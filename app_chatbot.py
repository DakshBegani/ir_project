import os
import streamlit as st
import arxiv
import groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

GROQ_API_KEY = "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"
client = groq.Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="🧠 arXiv Chatbot (Groq API)", page_icon="🤖")
st.title("🧠 arXiv Research Chatbot")

if "vectorstore" not in st.session_state:
    topic = st.text_input("Enter a research topic (e.g. 'transformers in NLP'):")

    if topic:
        with st.spinner("Scraping arXiv..."):
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

        with st.spinner("Creating vectorstore..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            docs = []

            for paper in st.session_state["papers"]:
                full_text = f"Title: {paper['title']}\n\n{paper['summary']}"
                chunks = splitter.split_text(full_text)
                docs.extend([Document(page_content=chunk) for chunk in chunks])

            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state["vectorstore"] = vectorstore
            st.success("Papers embedded and ready!")

if "vectorstore" in st.session_state:
    st.divider()
    st.subheader("Chat with the papers")

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
                retriever = st.session_state["vectorstore"].as_retriever()
                retrieved_docs = retriever.get_relevant_documents(user_input)
                st.markdown("**Context Retrieved:**")
                if not retrieved_docs:
                    st.warning("No relevant context retrieved. The answer may be inaccurate.")
                for doc in retrieved_docs:
                    st.code(doc.page_content[:300])

                combined_context = "\n\n".join(doc.page_content for doc in retrieved_docs[:5])
                prompt = f"Answer the following question based on the context below:\n\nContext:\n{combined_context}\n\nQuestion: {user_input}\nAnswer:"

                chat_completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}]
                )

                result = chat_completion.choices[0].message.content
                st.markdown("**Answer:**")
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
