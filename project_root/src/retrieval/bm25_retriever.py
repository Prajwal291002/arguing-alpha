import os
import json
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        self.corpus = []
        self.metadata = []
        self.tokenized_corpus = []
        self.bm25 = None

    def load_chunks(self, processed_chunks_path: str):

        for root, _, files in os.walk(processed_chunks_path):
            for file in files:
                if not file.endswith(".json"):
                    continue

                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    records = json.load(f)

                for record in records:
                    text = record["chunk_text"]

                    self.corpus.append(text)
                    self.metadata.append(record)

        print(f"Loaded {len(self.corpus)} chunks.")

    def tokenize(self, text: str):
        return text.lower().split()

    def build_index(self):
        self.tokenized_corpus = [self.tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 index built.")

    def retrieve(self, query: str, top_k: int = 5):

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in ranked_indices:
            results.append({
                "score": float(scores[idx]),
                "chunk_text": self.corpus[idx],
                "metadata": self.metadata[idx]
            })

        return results


if __name__ == "__main__":

    retriever = BM25Retriever()

    retriever.load_chunks("data/processed_chunks/")
    retriever.build_index()

    query = "liquidity risk and cash flow problems"

    results = retriever.retrieve(query, top_k=5)

    for i, res in enumerate(results):
        print(f"\nResult {i+1} | Score: {res['score']}")
        print(res["chunk_text"][:300])