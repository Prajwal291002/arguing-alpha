import pandas as pd
import re


class MetadataExtractor:

    def extract_metadata(self, df):

        cik_list = []
        year_list = []

        for file_name in df["source_file"]:

            # Extract CIK
            cik_match = re.search(r"(\d{10})", file_name)
            cik = cik_match.group(1) if cik_match else None

            # Extract year (approx from file name)
            year_match = re.search(r"-(\d{2})-", file_name)
            year = int("20" + year_match.group(1)) if year_match else None

            cik_list.append(cik)
            year_list.append(year)

        df["cik"] = cik_list
        df["year"] = year_list

        return df


if __name__ == "__main__":

    df = pd.read_csv("data/features/feature_dataset.csv")

    extractor = MetadataExtractor()
    df = extractor.extract_metadata(df)

    df.to_csv("data/features/feature_dataset_with_meta.csv", index=False)

    print(df.head())