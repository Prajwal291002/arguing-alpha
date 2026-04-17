import os
import json
import shutil

INPUT_DIR = "data/processed_chunks/"
OUTPUT_DIR = "data/filtered_chunks/"
MAX_CHUNKS = 40
MAX_FILES = 250


def count_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return len(data)


selected = 0

for root, _, files in os.walk(INPUT_DIR):

    for file in files:

        if not file.endswith(".json"):
            continue

        input_path = os.path.join(root, file)

        try:
            chunk_count = count_chunks(input_path)
        except:
            continue

        if chunk_count <= MAX_CHUNKS:

            relative_path = os.path.relpath(input_path, INPUT_DIR)
            output_path = os.path.join(OUTPUT_DIR, relative_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(input_path, output_path)

            selected += 1

            print(f"Selected: {file} ({chunk_count} chunks)")

        if selected >= MAX_FILES:
            break

    if selected >= MAX_FILES:
        break

print(f"\nTotal selected files: {selected}")