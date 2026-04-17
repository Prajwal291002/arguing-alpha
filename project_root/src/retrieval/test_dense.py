import json
from src.retrieval.dense_retriever import DenseRetriever


retriever = DenseRetriever(
    embeddings_path="data/embeddings/embeddings.npy",
    metadata_path="data/embeddings/metadata.json"
)

with open("configs/retrieval_queries.json", "r") as f:
    queries = json.load(f)

query = queries[0]

results = retriever.search(query, top_k=5)

print("\nQUERY:", query)

for i, res in enumerate(results):
    print(f"\nResult {i+1} | Score: {res['score']}")
    print(res["text"][:300])