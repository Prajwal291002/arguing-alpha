import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.generation.detector_agent import DetectorAgent
from src.generation.skeptic_agent import SkepticAgent
from src.generation.synthesizer_agent import SynthesizerAgent


class LLMBatchProcessor:

    def __init__(self, taxonomy_path, batch_size=8, max_workers=4, max_chunks_per_file=80):

        self.detector = DetectorAgent(taxonomy_path)
        self.skeptic = SkepticAgent()
        self.synthesizer = SynthesizerAgent()

        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_chunks_per_file = max_chunks_per_file

    def batch_chunks(self, chunks):
        for i in range(0, len(chunks), self.batch_size):
            yield chunks[i:i + self.batch_size]

    def process_batch(self, batch):

        combined_text = "\n\n".join(
            chunk.get("chunk_text", "")[:2000] for chunk in batch
        )

        if not combined_text.strip():
            return []

        try:
            detected = self.detector.run(combined_text)
            if not detected:
                return []

            validated = self.skeptic.run(combined_text, detected)
            if not validated:
                return []

            final = self.synthesizer.run(validated)
            return final if final else []

        except Exception as e:
            print(f"[BATCH ERROR] {e}")
            return []

    def process_file(self, file_path, output_path):

        start_time = time.time()

        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        original_chunk_count = len(chunks)

        # LIMIT chunks for performance
        if len(chunks) > self.max_chunks_per_file:
            chunks = chunks[:self.max_chunks_per_file]

        print(f"\nProcessing file: {file_path}")
        print(f"Chunks used: {len(chunks)} / {original_chunk_count}")

        batches = list(self.batch_chunks(chunks))
        total_batches = len(batches)

        all_risks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            future_to_batch = {
                executor.submit(self.process_batch, batch): idx
                for idx, batch in enumerate(batches)
            }

            for i, future in enumerate(as_completed(future_to_batch), 1):

                batch_index = future_to_batch[future]

                try:
                    result = future.result()
                    if result:
                        all_risks.extend(result)

                except Exception as e:
                    print(f"[ERROR] Batch {batch_index}: {e}")

                print(f"Progress: {i}/{total_batches} batches completed", end="\r")

        elapsed_time = time.time() - start_time

        print(f"\nCompleted file in {elapsed_time:.2f} seconds")
        print(f"Total risks extracted: {len(all_risks)}")

        if not all_risks:
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_risks, f, indent=2)

    def run(self, input_dir, output_dir):

        total_start = time.time()

        for root, _, files in os.walk(input_dir):
            for file in files:

                if not file.endswith(".json"):
                    continue

                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                try:
                    self.process_file(input_path, output_path)
                except Exception as e:
                    print(f"[FILE ERROR] {input_path}: {e}")

        total_time = time.time() - total_start
        print(f"\nTOTAL PIPELINE TIME: {total_time/60:.2f} minutes")


if __name__ == "__main__":

    processor = LLMBatchProcessor(
        taxonomy_path="configs/risk_taxonomy.json",
        batch_size=8,
        max_workers=6,              # PARALLELISM
        max_chunks_per_file=40      # HARD LIMIT
    )

    processor.run(
        input_dir="data/filtered_chunks/",
        output_dir="data/llm_outputs/"
    )