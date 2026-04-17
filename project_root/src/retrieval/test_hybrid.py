import json

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


# Initialize BM25
bm25 = BM25Retriever()
bm25.load_chunks("data/processed_chunks/")
bm25.tokenize_corpus()
bm25.build_index()

# Initialize Dense
dense = DenseRetriever(
    embeddings_path="data/embeddings/embeddings.npy",
    metadata_path="data/embeddings/metadata.json"
)

# Hybrid
hybrid = HybridRetriever(bm25, dense)

# Load queries
with open("configs/retrieval_queries.json", "r") as f:
    queries = json.load(f)

query = queries[0]

results = hybrid.search(query, top_k=10)

print("\nQUERY:", query)

for i, res in enumerate(results):
    print(f"\nResult {i+1} | Score: {res['score']}")
    print(res["text"][:300])