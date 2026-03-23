import os
import json
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.metadata = []
        self.index = None

    def load_chunks(self, processed_chunks_path: str):

        corpus = []

        for root, _, files in os.walk(processed_chunks_path):
            for file in files:
                if not file.endswith(".json"):
                    continue

                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    records = json.load(f)

                for record in records:
                    corpus.append(record["chunk_text"])
                    self.metadata.append(record)

        print(f"Loaded {len(corpus)} chunks.")
        return corpus

    def build_embeddings(self, corpus: list, batch_size: int = 32):

        print("Generating embeddings...")

        embeddings = self.model.encode(
            corpus,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        self.embeddings = embeddings.astype("float32")

        print("Embeddings generated.")

    def build_faiss_index(self):

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

        print("FAISS index built.")

    def save_index(self, index_path: str, metadata_path: str):

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)

        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print("Index and metadata saved.")

    def load_index(self, index_path: str, metadata_path: str):

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        print("Index and metadata loaded.")

    def retrieve(self, query: str, top_k: int = 5):

        query_embedding = self.model.encode([query]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for i, idx in enumerate(indices[0]):
            results.append({
                "score": float(distances[0][i]),
                "chunk_text": self.metadata[idx]["chunk_text"],
                "metadata": self.metadata[idx]
            })

        return results


if __name__ == "__main__":

    retriever = DenseRetriever()

    corpus = retriever.load_chunks("data/processed_chunks/")

    retriever.build_embeddings(corpus)
    retriever.build_faiss_index()

    retriever.save_index(
        index_path="data/retrieval_index/faiss_index.bin",
        metadata_path="data/retrieval_index/metadata.pkl"
    )

    # Test query
    query = "liquidity risk and cash flow problems"

    results = retriever.retrieve(query, top_k=5)

    for i, res in enumerate(results):
        print(f"\nResult {i+1} | Distance: {res['score']}")
        print(res["chunk_text"][:300])