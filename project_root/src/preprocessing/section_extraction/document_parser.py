import re


class SECDocumentParser:
    
    def __init__(self):
        pass

    def split_into_documents(self, raw_text: str) -> list:
        document_pattern = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)
        documents = document_pattern.findall(raw_text)
        return documents

    def extract_document_type(self, document: str) -> str:
        match = re.search(r"<TYPE>(.*?)\n", document)
        if match:
            return match.group(1).strip()
        return None

    def extract_text_block(self, document: str) -> str:
        match = re.search(r"<TEXT>(.*?)</TEXT>", document, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def get_primary_filing_text(self, raw_text: str) -> str:
        documents = self.split_into_documents(raw_text)
        
        for document in documents:
            document_type = self.extract_document_type(document)
            
            if document_type in ["10-K", "10-Q"]:
                text_block = self.extract_text_block(document)
                return text_block
        
        return None

if __name__ == "__main__":
    sample_file_path = "data/raw_filings/A/2016/0001090872-16-000076_10-Q.html"
    
    with open(sample_file_path, "r", encoding="utf-8") as file:
        raw_content = file.read()
    
    parser = SECDocumentParser()
    
    primary_text = parser.get_primary_filing_text(raw_content)
    
    if primary_text:
        print("Primary filing extracted successfully.\n")
        print(primary_text[:1000])
    else:
        print("No valid 10-K or 10-Q document found.")

