import os
import json
import uuid


class ChunkGenerator:

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_into_words(self, text: str) -> list:
        return text.split()

    def generate_chunks(self, text: str) -> list:
        words = self.split_into_words(text)

        chunks = []
        start_index = 0

        while start_index < len(words):
            end_index = start_index + self.chunk_size

            chunk_words = words[start_index:end_index]
            chunk_text = " ".join(chunk_words)

            chunks.append(chunk_text)

            start_index += (self.chunk_size - self.overlap)

        return chunks

    def create_chunk_records(self, chunks: list, metadata: dict) -> list:
        records = []

        for chunk in chunks:
            record = {
                "company_identifier": metadata["company_identifier"],
                "filing_date": metadata["filing_date"],
                "section_type": metadata["section_type"],
                "chunk_id": str(uuid.uuid4()),
                "chunk_text": chunk
            }
            records.append(record)

        return records

    def save_chunks(self, records: list, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(records, file, indent=2)


if __name__ == "__main__":
    from src.preprocessing.section_extraction.document_parser import SECDocumentParser
    from src.preprocessing.section_extraction.section_extractor import SectionExtractor

    file_path = "data/raw_filings/A/2018/0001090872-18-000004_10-Q.html"

    company_identifier = "TEST_COMPANY"
    filing_date = "2018"

    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    parser = SECDocumentParser()
    extractor = SectionExtractor()
    chunker = ChunkGenerator()

    primary_text = parser.get_primary_filing_text(raw_text)

    if not primary_text:
        print("No valid filing found.")
        exit()

    cleaned_text = extractor.clean_html(primary_text)

    sections = {
        "risk_factors": extractor.extract_risk_factors(cleaned_text),
        "mdna": extractor.extract_mdna(cleaned_text)
    }

    all_records = []

    for section_type, section_text in sections.items():
        if not section_text:
            continue

        chunks = chunker.generate_chunks(section_text)

        metadata = {
            "company_identifier": company_identifier,
            "filing_date": filing_date,
            "section_type": section_type
        }

        records = chunker.create_chunk_records(chunks, metadata)
        all_records.extend(records)

    output_path = "data/processed_chunks/test_output.json"
    chunker.save_chunks(all_records, output_path)

    print(f"Saved {len(all_records)} chunks to {output_path}")