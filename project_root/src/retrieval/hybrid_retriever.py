import numpy as np


class HybridRetriever:

    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

    def normalize_scores(self, scores):
        scores = np.array(scores)

        if np.max(scores) == np.min(scores):
            return np.zeros_like(scores)

        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    def search(self, query: str, top_k: int = 10):

        # BM25 results
        bm25_results = self.bm25.search(query, top_k=top_k * 5)

        # Dense results
        dense_results = self.dense.search(query, top_k=top_k * 5)

        # Build index mapping
        combined = {}

        # BM25 indexing
        for res in bm25_results:
            key = res["metadata"]["chunk_id"]
            combined[key] = {
                "text": res["text"],
                "metadata": res["metadata"],
                "bm25_score": res["score"],
                "dense_score": 0.0
            }

        # Dense indexing
        for res in dense_results:
            key = res["metadata"]["chunk_id"]

            if key not in combined:
                combined[key] = {
                    "text": res["text"],
                    "metadata": res["metadata"],
                    "bm25_score": 0.0,
                    "dense_score": res["score"]
                }
            else:
                combined[key]["dense_score"] = res["score"]

        # Normalize scores
        bm25_scores = [v["bm25_score"] for v in combined.values()]
        dense_scores = [v["dense_score"] for v in combined.values()]

        norm_bm25 = self.normalize_scores(bm25_scores)
        norm_dense = self.normalize_scores(dense_scores)

        # Combine scores
        results = []
        for i, (key, value) in enumerate(combined.items()):
            final_score = 0.5 * norm_bm25[i] + 0.5 * norm_dense[i]

            results.append({
                "score": float(final_score),
                "text": value["text"],
                "metadata": value["metadata"]
            })

        # Sort final results
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        return results