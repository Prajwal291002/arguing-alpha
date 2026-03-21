import os
import time
import requests
import pandas as pd
import yaml

class SECFilingsDownloader:
    
    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.headers = {
            "User-Agent": self.config["project_settings"]["user_agent"]
        }
        
        self.base_url = "https://data.sec.gov/submissions/"
        self.raw_data_path = self.config["storage"]["raw_data_path"]
        os.makedirs(self.raw_data_path, exist_ok=True)
        
        print("Fetching master CIK dictionary from SEC...")
        self.ticker_to_cik = self._build_cik_mapping()

    def _build_cik_mapping(self) -> dict:
        """Downloads the SEC ticker list once and returns a dictionary mapping."""
        ticker_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(ticker_url, headers=self.headers)
        response.raise_for_status() 
        
        data = response.json()
        mapping = {}
        for entry in data.values():
            mapping[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
            
        time.sleep(0.1) 
        return mapping

    def get_cik_from_ticker(self, ticker: str) -> str:
        return self.ticker_to_cik.get(ticker.upper())

    def fetch_company_filings(self, cik: str) -> dict:
        url = f"{self.base_url}CIK{cik}.json"
        response = requests.get(url, headers=self.headers)
        
        time.sleep(0.1) 
        
        if response.status_code != 200:
            return None
        
        return response.json()

    def download_filings(self, ticker: str, cik: str):
        filings_data = self.fetch_company_filings(cik)
        
        if filings_data is None:
            print(f"Failed to fetch data for {ticker}")
            return
        
        recent_filings = filings_data.get("filings", {}).get("recent", {})
        
        forms = recent_filings.get("form", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])
        
        for form, accession, date in zip(forms, accession_numbers, filing_dates):
            
            if form not in self.config["data_collection"]["filing_types"]:
                continue
            
            accession_clean = accession.replace("-", "")
            
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{accession}.txt"
            
            response = requests.get(filing_url, headers=self.headers)
            time.sleep(0.25) 
            
            if response.status_code != 200:
                print(f"  Failed {form} for {ticker} | Error Code: {response.status_code}")
                print(f"  URL: {filing_url}") 
                continue
            
            year = date.split("-")[0]
            save_dir = os.path.join(self.raw_data_path, ticker, year)
            os.makedirs(save_dir, exist_ok=True)
            
            file_path = os.path.join(save_dir, f"{accession}_{form}.html")
            
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(response.text)
            
            print(f"  Saved: {file_path}")

    def run_batch_download(self, companies_df: pd.DataFrame):
        for _, row in companies_df.iterrows():

            ticker = row.get("Symbol", row.get("ticker", None))
            
            if not ticker:
                continue
                
            print(f"\nProcessing {ticker}...")
            
            cik = self.get_cik_from_ticker(ticker)
            
            if cik is None:
                print(f"  CIK not found for {ticker}")
                continue
            
            self.download_filings(ticker, cik)



if __name__ == "__main__":
    downloader = SECFilingsDownloader(config_path="configs/data_config.yaml")
    
    from src.data_ingestion.company_selector import fetch_sp500_companies
    
    companies = fetch_sp500_companies(
        max_companies=downloader.config["data_collection"]["max_companies"]
    )
    
    downloader.run_batch_download(companies)