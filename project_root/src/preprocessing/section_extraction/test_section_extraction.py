from src.preprocessing.section_extraction.document_parser import SECDocumentParser
from src.preprocessing.section_extraction.section_extractor import SectionExtractor


file_path = "data/raw_filings/A/2016/0001090872-16-000076_10-Q.html"

with open(file_path, "r", encoding="utf-8") as file:
    raw_text = file.read()

parser = SECDocumentParser()
extractor = SectionExtractor()

primary_text = parser.get_primary_filing_text(raw_text)

if not primary_text:
    print("No valid 10-K/10-Q found.")
    exit()

cleaned_text = extractor.clean_html(primary_text)

risk_section = extractor.extract_risk_factors(cleaned_text)
mdna_section = extractor.extract_mdna(cleaned_text)

print("\n--- RISK FACTORS ---\n")
print(risk_section[:500] if risk_section else "Not Found")

print("\n--- MD&A ---\n")
print(mdna_section[:500] if mdna_section else "Not Found")