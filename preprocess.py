import json
import os
from typing import List, Dict

def load_papers(path="data/arxiv_papers.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_papers(papers: List[Dict]) -> List[str]:
    processed = []
    for paper in papers:
        text = f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nPublished: {paper['published']}\n\nAbstract:\n{paper['summary']}"
        # Optional cleanup: remove LaTeX-style equations, extra spaces, etc.
        text = text.replace("\n", " ").strip()
        processed.append(text)
    return processed

def save_chunks(chunks: List[str], path="data/preprocessed_chunks.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(chunks)} preprocessed chunks to {path}")

if __name__ == "__main__":
    papers = load_papers()
    chunks = preprocess_papers(papers)
    save_chunks(chunks)
