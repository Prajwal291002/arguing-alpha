import os
import json
from datetime import datetime

from src.preprocessing.section_extraction.document_parser import SECDocumentParser
from src.preprocessing.section_extraction.section_extractor import SectionExtractor
from src.preprocessing.chunking.chunk_generator import ChunkGenerator


class BatchChunkProcessor:

    def __init__(self, raw_data_path: str, output_path: str, log_path: str):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.log_path = log_path

        self.parser = SECDocumentParser()
        self.extractor = SectionExtractor()
        self.chunker = ChunkGenerator()

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")

    def process_single_file(self, file_path: str, ticker: str, year: str):

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                raw_text = file.read()

            primary_text = self.parser.get_primary_filing_text(raw_text)

            if not primary_text:
                self.log(f"SKIPPED (no primary doc): {file_path}")
                return

            cleaned_text = self.extractor.clean_html(primary_text)

            sections = {
                "risk_factors": self.extractor.extract_risk_factors(cleaned_text),
                "mdna": self.extractor.extract_mdna(cleaned_text)
            }

            all_records = []

            for section_type, section_text in sections.items():
                if not section_text:
                    continue

                chunks = self.chunker.generate_chunks(section_text)

                metadata = {
                    "company_identifier": ticker,
                    "filing_date": year,
                    "section_type": section_type
                }

                records = self.chunker.create_chunk_records(chunks, metadata)
                all_records.extend(records)

            if not all_records:
                self.log(f"SKIPPED (no sections): {file_path}")
                return

            file_name = os.path.basename(file_path).replace(".txt", ".json").replace(".html", ".json")

            save_dir = os.path.join(self.output_path, ticker, year)
            os.makedirs(save_dir, exist_ok=True)

            output_file_path = os.path.join(save_dir, file_name)

            with open(output_file_path, "w", encoding="utf-8") as output_file:
                json.dump(all_records, output_file, indent=2)

            self.log(f"SUCCESS: {file_path} → {len(all_records)} chunks")

        except Exception as error:
            self.log(f"ERROR: {file_path} → {str(error)}")

    def run(self):

        total_files = 0

        for ticker in os.listdir(self.raw_data_path):
            ticker_path = os.path.join(self.raw_data_path, ticker)

            if not os.path.isdir(ticker_path):
                continue

            for year in os.listdir(ticker_path):
                year_path = os.path.join(ticker_path, year)

                if not os.path.isdir(year_path):
                    continue

                for file_name in os.listdir(year_path):
                    file_path = os.path.join(year_path, file_name)

                    if not file_name.endswith((".txt", ".html")):
                        continue

                    print(f"Processing: {file_path}")

                    self.process_single_file(file_path, ticker, year)

                    total_files += 1

        self.log(f"COMPLETED: Processed {total_files} files")


if __name__ == "__main__":

    processor = BatchChunkProcessor(
        raw_data_path="data/raw_filings/",
        output_path="data/processed_chunks/",
        log_path="logs/preprocessing_log.txt"
    )

    processor.run()