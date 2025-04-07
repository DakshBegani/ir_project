import streamlit as st
from scrape_arxiv import fetch_arxiv_papers, save_to_json

st.title("arXiv Paper Fetcher")
st.write("Enter a topic to fetch recent research papers from arXiv.")

query = st.text_input("Enter your query:", value="machine learning")
max_results = st.slider("Number of results to fetch:", min_value=5, max_value=50, value=10)

if st.button("Fetch Papers"):
    with st.spinner("Fetching papers from arXiv..."):
        papers = fetch_arxiv_papers(query=query, max_results=max_results)
        save_to_json(papers)
    st.success(f"Fetched and saved {len(papers)} papers.")
    st.download_button(
        label="Download JSON",
        data=open("data/arxiv_papers.json", "rb"),
        file_name="arxiv_papers.json",
        mime="application/json"
    )
