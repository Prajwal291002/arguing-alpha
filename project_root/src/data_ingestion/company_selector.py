import requests
import pandas as pd

def fetch_sp500_companies(max_companies: int) -> pd.DataFrame:
    wikipedia_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    headers = {
        "User-Agent": "Prajwal2k02 bhandarkarprajwal.02@gmail.com"
    }
    
    response = requests.get(wikipedia_url, headers=headers)
    response.raise_for_status() 
    
    tables = pd.read_html(response.text)
    sp500_table = tables[0]
    
    selected_companies = sp500_table[['Symbol', 'Security']].head(max_companies)
    selected_companies.columns = ['ticker', 'company_name']
    
    return selected_companies

if __name__ == "__main__":
    companies_dataframe = fetch_sp500_companies(max_companies=50)
    
    print("Total companies selected:", len(companies_dataframe))
    print(companies_dataframe.head())