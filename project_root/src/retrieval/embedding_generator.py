import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def load_chunks(self, processed_chunks_path: str):

        corpus = []
        metadata = []

        for root, _, files in os.walk(processed_chunks_path):
            for file in files:
                if not file.endswith(".json"):
                    continue

                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        records = json.load(f)

                        for record in records:
                            text = record.get("chunk_text", "").strip()

                            if not text:
                                continue

                            corpus.append(text)
                            metadata.append(record)

                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")

        print(f"Total chunks loaded: {len(corpus)}")
        return corpus, metadata

    def generate_embeddings(self, corpus: list, batch_size: int = 128):

        embeddings = []

        for i in tqdm(range(0, len(corpus), batch_size)):
            batch = corpus[i:i + batch_size]

            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)

        return embeddings

    def save_outputs(self, embeddings: np.ndarray, metadata: list, output_dir: str):

        os.makedirs(output_dir, exist_ok=True)

        embeddings_path = os.path.join(output_dir, "embeddings.npy")
        metadata_path = os.path.join(output_dir, "metadata.json")

        np.save(embeddings_path, embeddings)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        print(f"Saved embeddings to {embeddings_path}")
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":

    generator = EmbeddingGenerator()

    corpus, metadata = generator.load_chunks("data/processed_chunks/")

    embeddings = generator.generate_embeddings(corpus, batch_size=128)

    generator.save_outputs(
        embeddings,
        metadata,
        output_dir="data/embeddings/"
    )