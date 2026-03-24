import numpy as np
import re

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


class RetrievalEvaluator:

    def __init__(self):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.hybrid = HybridRetriever()

        self.stopwords = {"and", "or", "the", "of", "in", "risk"}

    def initialize(self):
        print("Initializing BM25...")
        self.bm25.load_chunks("data/processed_chunks/")
        self.bm25.build_index()

        print("Initializing Dense...")
        self.dense.load_index(
            "data/retrieval_index/faiss_index.bin",
            "data/retrieval_index/metadata.pkl"
        )

        print("Initializing Hybrid...")
        self.hybrid.initialize()

    def clean_and_tokenize(self, text: str):
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if w not in self.stopwords]

    def is_relevant(self, query: str, text: str) -> bool:

        query_terms = self.clean_and_tokenize(query)
        text_terms = set(self.clean_and_tokenize(text))

        if len(query_terms) < 2:
            return False

        # Require at least 2 strong matches
        match_terms = [term for term in query_terms if term in text_terms]

        return len(match_terms) >= 2

    def estimate_total_relevant(self, query: str):

        # Use BM25 top 50 as approximation of relevant pool
        candidates = self.bm25.retrieve(query, top_k=50)

        relevant_count = sum(
            1 for res in candidates
            if self.is_relevant(query, res["chunk_text"])
        )

        return max(relevant_count, 1)  # avoid divide by zero

    def precision_at_k(self, results, query, k=5):
        relevant = sum(
            1 for res in results[:k]
            if self.is_relevant(query, res["chunk_text"])
        )
        return relevant / k

    def recall_at_k(self, results, query, total_relevant, k=5):
        relevant = sum(
            1 for res in results[:k]
            if self.is_relevant(query, res["chunk_text"])
        )
        return relevant / total_relevant

    def ndcg_at_k(self, results, query, k=5):
        dcg = 0
        for i, res in enumerate(results[:k]):
            rel = 1 if self.is_relevant(query, res["chunk_text"]) else 0
            dcg += rel / np.log2(i + 2)

        # Ideal DCG (all relevant)
        ideal_rels = sorted(
            [1 if self.is_relevant(query, r["chunk_text"]) else 0 for r in results[:k]],
            reverse=True
        )

        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0

    def evaluate_query(self, query):

        total_relevant = self.estimate_total_relevant(query)

        bm25_results = self.bm25.retrieve(query, top_k=10)
        dense_results = self.dense.retrieve(query, top_k=10)
        hybrid_results = self.hybrid.retrieve(query, final_k=10)

        metrics = {}

        for name, results in zip(
            ["BM25", "Dense", "Hybrid"],
            [bm25_results, dense_results, hybrid_results]
        ):
            metrics[name] = {
                "precision": self.precision_at_k(results, query),
                "recall": self.recall_at_k(results, query, total_relevant),
                "ndcg": self.ndcg_at_k(results, query)
            }

        return metrics

    def evaluate_all(self, queries):

        all_results = []

        for query in queries:
            print(f"\nEvaluating: {query}")

            result = self.evaluate_query(query)

            for model_name, metrics in result.items():
                all_results.append({
                    "query": query,
                    "model": model_name,
                    "precision": round(metrics["precision"], 3),
                    "recall": round(metrics["recall"], 3),
                    "ndcg": round(metrics["ndcg"], 3)
                })

        return all_results


if __name__ == "__main__":

    evaluator = RetrievalEvaluator()
    evaluator.initialize()

    queries = [
        "liquidity risk indicators",
        "cash flow shortages",
        "credit risk exposure",
        "counterparty default",
        "cybersecurity breaches and data loss",
        "internal control failures",
        "supply chain disruption risk",
        "raw material shortages",
        "regulatory investigation risk",
        "interest rate fluctuations"
    ]

    results = evaluator.evaluate_all(queries)

    print("\n===== FINAL RESULTS =====\n")

    for row in results:
        print(row)