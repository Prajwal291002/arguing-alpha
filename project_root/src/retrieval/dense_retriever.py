import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(self, embeddings_path: str, metadata_path: str,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):

        print("Loading embeddings...")
        self.embeddings = np.load(embeddings_path)

        print("Loading metadata...")
        import json
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print("Loading model...")
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5):

        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # Cosine similarity
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []

        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "text": self.metadata[idx]["chunk_text"],
                "metadata": self.metadata[idx]
            })

        return results