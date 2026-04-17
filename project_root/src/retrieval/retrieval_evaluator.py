import json
import numpy as np
from collections import defaultdict

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


class RetrievalEvaluator:

    def __init__(self, processed_path, embeddings_path, metadata_path):

        self.bm25 = BM25Retriever()
        self.bm25.load_chunks(processed_path)
        self.bm25.tokenize_corpus()
        self.bm25.build_index()

        self.dense = DenseRetriever(embeddings_path, metadata_path)
        self.hybrid = HybridRetriever(self.bm25, self.dense)

    def get_keywords(self, query: str):

        keyword_map = {
            "liquidity": ["liquidity", "cash flow", "funding", "liquidity risk"],
            "credit": ["credit", "debt", "default", "credit risk"],
            "operational": ["operational", "process failure", "system failure"],
            "supply": ["supply chain", "supplier", "logistics"],
            "market": ["market demand", "revenue decline", "market risk"],
            "macro": ["inflation", "interest rate", "economic conditions"],
            "regulatory": ["regulation", "compliance", "legal risk"]
        }

        for key in keyword_map:
            if key in query.lower():
                return keyword_map[key]

        return query.lower().split()

    def is_relevant(self, text: str, keywords: list):

        text_lower = text.lower()

        # Rule 1: Exact phrase match
        for kw in keywords:
            if kw in text_lower:
                return True

        # Rule 2: At least 2 keyword matches
        match_count = sum(1 for kw in keywords if kw in text_lower)
        if match_count >= 2:
            return True

        return False

    def compute_metrics(self, results, keywords, k):

        hits = 0
        reciprocal_rank = 0

        for i, res in enumerate(results[:k]):

            if self.is_relevant(res["text"], keywords):
                hits += 1

                if reciprocal_rank == 0:
                    reciprocal_rank = 1 / (i + 1)

        precision = hits / k
        recall = hits / k  # proxy

        return precision, recall, reciprocal_rank

    def evaluate_query(self, query):

        keywords = self.get_keywords(query)

        methods = {
            "bm25": self.bm25.search(query, top_k=10),
            "dense": self.dense.search(query, top_k=10),
            "hybrid": self.hybrid.search(query, top_k=10),
        }

        results = {}

        for method_name, method_results in methods.items():

            metrics = {}

            for k in [5, 10]:
                precision, recall, mrr = self.compute_metrics(
                    method_results, keywords, k
                )

                metrics[f"P@{k}"] = precision
                metrics[f"R@{k}"] = recall
                metrics[f"MRR@{k}"] = mrr

            results[method_name] = metrics

        return results

    def run_evaluation(self, queries):

        final_results = defaultdict(list)

        for query in queries:
            query_result = self.evaluate_query(query)

            for method in query_result:
                final_results[method].append(query_result[method])

        averaged = {}

        for method, metrics_list in final_results.items():

            avg_metrics = {}

            for key in metrics_list[0]:
                avg_metrics[key] = np.mean([m[key] for m in metrics_list])

            averaged[method] = avg_metrics

        return averaged


if __name__ == "__main__":

    with open("configs/retrieval_queries.json", "r") as f:
        queries = json.load(f)

    evaluator = RetrievalEvaluator(
        processed_path="data/processed_chunks/",
        embeddings_path="data/embeddings/embeddings.npy",
        metadata_path="data/embeddings/metadata.json"
    )

    results = evaluator.run_evaluation(queries)

    print("\n===== FINAL RESULTS (STRICT EVAL) =====\n")

    for method, metrics in results.items():
        print(f"\n{method.upper()}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")