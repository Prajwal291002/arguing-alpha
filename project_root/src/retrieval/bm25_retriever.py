import os
import json
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self):
        self.corpus = []
        self.chunk_metadata = []
        self.tokenized_corpus = []
        self.bm25 = None

    def load_chunks(self, processed_chunks_path: str):

        total_files = 0
        total_loaded = 0
        failed_files = 0

        for root, _, files in os.walk(processed_chunks_path):
            for file in files:

                if not file.endswith(".json"):
                    continue

                total_files += 1
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        records = json.load(f)

                        for record in records:
                            text = record.get("chunk_text", "").strip()

                            if not text:
                                continue

                            self.corpus.append(text)
                            self.chunk_metadata.append(record)
                            total_loaded += 1

                except Exception as e:
                    failed_files += 1
                    print(f"Failed to load: {file_path} | Error: {e}")

        print("\n===== BM25 LOAD SUMMARY =====")
        print(f"Total JSON files read : {total_files}")
        print(f"Total chunks loaded   : {total_loaded}")
        print(f"Failed files          : {failed_files}")

    def tokenize_corpus(self):
        print("Tokenizing corpus...")
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]

    def build_index(self):
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []

        for idx in top_indices:
            results.append({
                "score": scores[idx],
                "text": self.corpus[idx],
                "metadata": self.chunk_metadata[idx]
            })

        return results