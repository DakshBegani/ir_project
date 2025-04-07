import arxiv
import json

def fetch_arxiv_papers(query="machine learning", max_results=10):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in search.results():
        paper = {
            "title": result.title.strip(),
            "authors": [author.name for author in result.authors],
            "summary": result.summary.strip(),
            "published": result.published.strftime("%Y-%m-%d"),
            "arxiv_url": result.entry_id
        }
        papers.append(paper)

    return papers

def save_to_json(papers, path="data/arxiv_papers.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
