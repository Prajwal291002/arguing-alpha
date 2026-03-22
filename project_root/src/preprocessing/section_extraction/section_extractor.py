import re
from bs4 import BeautifulSoup


class SectionExtractor:

    def __init__(self):
        pass

    def clean_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "lxml")

        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator=" ")

        text = re.sub(r"\s+", " ", text)

        return text.strip()
    def extract_section(self, text: str, start_pattern: str, end_patterns: list) -> str:
        
        matches = list(re.finditer(start_pattern, text, re.IGNORECASE))
        
        if not matches:
            return None

        # Use LAST occurrence (skip Table of Contents)
        start_match = matches[-1]
        start_index = start_match.start()

        end_index = len(text)

        for pattern in end_patterns:
            match = re.search(pattern, text[start_index:], re.IGNORECASE)
            if match:
                candidate_end = start_index + match.start()
                
                # choose nearest valid end
                if candidate_end < end_index:
                    end_index = candidate_end

        extracted_text = text[start_index:end_index].strip()

        # sanity filter (avoid very short outputs like TOC)
        if len(extracted_text) < 500:
            return None

        return extracted_text

    def extract_risk_factors(self, text: str) -> str:
        return self.extract_section(
            text,
            start_pattern=r"item\s+1a\.?\s+risk\s+factors",
            end_patterns=[
                r"item\s+1b",
                r"item\s+2",
                r"item\s+3"
            ]
        )

    def extract_mdna(self, text: str) -> str:
        return self.extract_section(
            text,
            start_pattern=r"item\s+(7|2)\.?\s+management[’'s\s]+discussion",
            end_patterns=[
                r"item\s+8",
                r"item\s+3",
                r"item\s+4"
            ]
        )


if __name__ == "__main__":
    from src.preprocessing.section_extraction.document_parser import SECDocumentParser

    file_path = "data/raw_filings/A/2018/0001090872-18-000004_10-Q.html"

    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    parser = SECDocumentParser()
    extractor = SectionExtractor()

    primary_text = parser.get_primary_filing_text(raw_text)

    cleaned_text = extractor.clean_html(primary_text)

    risk_section = extractor.extract_risk_factors(cleaned_text)
    mdna_section = extractor.extract_mdna(cleaned_text)

    print("\n--- RISK FACTORS ---\n")
    print(risk_section[:1000] if risk_section else "Not Found")

    print("\n--- MD&A ---\n")
    print(mdna_section[:1000] if mdna_section else "Not Found")