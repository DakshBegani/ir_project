import os
import streamlit as st
import arxiv
import groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

GROQ_API_KEY = "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"
client = groq.Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="🧠 arXiv Chatbot (Groq API)", page_icon="🤖")
st.title("🧠 arXiv Research Chatbot")

if st.sidebar.button("✨ Start New Chat"):
    keys_to_clear = ["vectorstore", "papers", "messages", "topic", "sort_criterion"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("New chat started! Enter a new Research topic below.")
    time.sleep(1)
    st.rerun() #Refresh the state

if "vectorstore" not in st.session_state:
    st.sidebar.header("Find Papers")

    topic = st.sidebar.text_input(
        "Enter a research topic:",
        key="topic_input",
        placeholder="e.g., 'transformers in NLP'"
        )

    sort_options = {
        "Relevance": arxiv.SortCriterion.Relevance,
        "Submission Date (Newest First)": arxiv.SortCriterion.SubmittedDate,
        "Last Updated Date (Newest First)": arxiv.SortCriterion.LastUpdatedDate
    }
    sort_selection_label = st.sidebar.selectbox(
        "Sort papers by:",
        options=list(sort_options.keys()),
        index=2,  # Default to LastUpdateDate
        key="sort_select"
    )
    selected_sort_criterion = sort_options[sort_selection_label]

    if st.sidebar.button("Search and Embed Papers", disabled=not topic):
        st.session_state["topic"] = topic
        st.session_state["sort_criterion"] = selected_sort_criterion

        with st.spinner(f"Scraping arXiv for '{topic}'"):
            search = arxiv.Search(
                query=st.session_state["topic"],
                max_results=10,
                sort_by=st.session_state["sort_criterion"]
            )
            papers_data = []
            for result in search.results():
                papers_data.append({
                    "entry_id": result.entry_id,
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary.replace("\n", " "),  # Replace newlines for cleaner text
                    "published": result.published.strftime("%Y-%m-%d") if result.published else "N/A",
                    "pdf_url": result.pdf_url
                })

            if not papers_data:
                st.warning(f"No papers found for '{topic}'.")
                keys_to_clear = ["vectorstore", "papers", "messages", "topic", "sort_criterion"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.stop() # Also need to try st.rerun()

            st.session_state["papers"] = papers_data
            st.success(f"Found {len(papers_data)} papers.")
            with st.sidebar.expander("View Found Papers", expanded=False):
                for i, p in enumerate(st.session_state["papers"]):
                    st.write(f"**{i + 1}. {p['title']}** ({p['published']})")
                    st.caption(f"Authors: {', '.join(p['authors'])}")
                    st.caption(f"[Link]({p['entry_id']})")  # Link to abstract page

        with st.spinner("Creating vectorstore..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            docs = []

            for paper in st.session_state["papers"]:
                full_text = f"Title: {paper['title']}\n\n{paper['summary']}"
                chunks = splitter.split_text(full_text)
                for chunk in chunks: #Adds more data to the prev docs list
                    docs.append(Document(page_content=chunk, metadata={
                        "title": paper['title'],
                        "published": paper['published'],
                        "authors": ", ".join(paper['authors']),
                        "source_id": paper['entry_id']
                    }))
                if not docs:
                    st.error("Could not extract text from papers to create embeddings.")
                    if "papers" in st.session_state: del st.session_state["papers"]
                    if "topic" in st.session_state: del st.session_state["topic"]
                    st.stop()

            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state["vectorstore"] = vectorstore
            st.success("Papers embedded and ready!")
            st.rerun()

if "vectorstore" in st.session_state:
    st.header(f"Chat about: {st.session_state.get('topic', 'the selected papers')}")
    st.sidebar.success("Papers loaded and embedded.")

    with st.sidebar.expander("View Loaded Papers", expanded=False):
        if "papers" in st.session_state:
            for i, p in enumerate(st.session_state["papers"]):
                st.write(f"**{i + 1}. {p['title']}** ({p['published']})")
                st.caption(f"[Link]({p['entry_id']})")
        else:
            st.write("Paper details not available.")

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
                retriever = st.session_state["vectorstore"].as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                retrieved_docs = retriever.invoke(user_input)
                st.markdown("**Context Retrieved:**")
                if not retrieved_docs:
                    st.warning("No relevant context retrieved. The answer may be inaccurate.")
                else:
                    with st.expander("Show Retrieved Context"):
                        for i, doc in enumerate(retrieved_docs):
                            st.caption(f"--- Context Chunk {i+1} ---")
                            st.markdown(f"**Source:** {doc.metadata.get('title', 'N/A')} ({doc.metadata.get('published', 'N/A')})")
                            st.markdown(f"> {doc.page_content[:300]}...")
                            st.markdown(f"[Link to Paper]({doc.metadata.get('source_id', '#')})")
                    combined_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                prompt = f"""You are an AI assistant specialized in answering questions based *only* on the provided scientific paper summaries.
                                    Use the following context retrieved from arXiv paper summaries to answer the question.
                                    If the context does not contain the information to answer the question, clearly state that the information is not available in the provided summaries.
                                    Do not make up information or use external knowledge outside of the provided context. Be concise and accurate.
                                    Context:
                                    ---
                                    {combined_context}
                                    ---

                                    Question: {user_input}

                                    Answer:"""
                chat_completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}]
                )

                result = chat_completion.choices[0].message.content
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})

else:
    st.info("Enter a research topic and select sorting criteria in the sidebar to begin.")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by arXiv, Groq (Llama 3), LangChain, FAISS, and Streamlit.")