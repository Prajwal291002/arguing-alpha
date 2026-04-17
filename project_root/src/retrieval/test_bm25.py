import json
from src.retrieval.bm25_retriever import BM25Retriever


retriever = BM25Retriever()

retriever.load_chunks("data/processed_chunks/")
retriever.tokenize_corpus()
retriever.build_index()

with open("configs/retrieval_queries.json", "r") as f:
    queries = json.load(f)

query = queries[0]

results = retriever.search(query, top_k=5)

print("\nQUERY:", query)

for i, res in enumerate(results):
    print(f"\nResult {i+1} | Score: {res['score']}")
    print(res["text"][:300])