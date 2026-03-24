from collections import defaultdict

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


class HybridRetriever:

    def __init__(self, k: int = 60):
        self.k = k

        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()

    def initialize(self):

        print("Loading BM25...")
        self.bm25.load_chunks("data/processed_chunks/")
        self.bm25.build_index()

        print("Loading Dense Index...")
        self.dense.load_index(
            index_path="data/retrieval_index/faiss_index.bin",
            metadata_path="data/retrieval_index/metadata.pkl"
        )

    def reciprocal_rank_fusion(self, bm25_results, dense_results):

        scores = defaultdict(float)
        metadata_map = {}

        # BM25 contribution
        for rank, result in enumerate(bm25_results):
            chunk_id = result["metadata"]["chunk_id"]
            scores[chunk_id] += 1 / (self.k + rank)
            metadata_map[chunk_id] = result

        # Dense contribution
        for rank, result in enumerate(dense_results):
            chunk_id = result["metadata"]["chunk_id"]
            scores[chunk_id] += 1 / (self.k + rank)
            metadata_map[chunk_id] = result

        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []

        for chunk_id, score in ranked:
            entry = metadata_map[chunk_id]
            entry["rrf_score"] = score
            final_results.append(entry)

        return final_results

    def retrieve(self, query: str, final_k: int = 5):

        bm25_results = self.bm25.retrieve(query, top_k=20)
        dense_results = self.dense.retrieve(query, top_k=20)

        fused_results = self.reciprocal_rank_fusion(bm25_results, dense_results)

        return fused_results[:final_k]


if __name__ == "__main__":

    retriever = HybridRetriever()
    retriever.initialize()

    query = "liquidity risk and cash flow problems"

    results = retriever.retrieve(query, final_k=5)

    for i, res in enumerate(results):
        print(f"\nResult {i+1} | RRF Score: {res['rrf_score']}")
        print(res["chunk_text"][:300])