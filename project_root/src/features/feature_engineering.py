import os
import json
import pandas as pd
from collections import Counter


class FeatureEngineer:

    def __init__(self):

        self.categories = [
            "Liquidity Risk",
            "Credit Risk",
            "Operational Risk",
            "Market Risk",
            "Regulatory Risk",
            "Supply Chain Risk",
            "Technological Risk",
            "Macroeconomic Risk"
        ]

    def process_file(self, file_path):

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            return None

        category_counts = Counter()
        confidences = []

        for item in data:
            category = item.get("risk_category")
            confidence = item.get("confidence", 0)

            if category:
                category_counts[category] += 1

            confidences.append(confidence)

        feature_row = {}

        # Category features
        for cat in self.categories:
            feature_row[f"{cat.lower().replace(' ', '_')}_count"] = category_counts.get(cat, 0)

        # Aggregate features
        feature_row["risk_diversity"] = len(category_counts)
        feature_row["total_risk_count"] = sum(category_counts.values())
        feature_row["mean_confidence"] = sum(confidences) / len(confidences)

        return feature_row

    def process_directory(self, input_dir):

        all_rows = []

        for root, _, files in os.walk(input_dir):
            for file in files:
                if not file.endswith(".json"):
                    continue

                file_path = os.path.join(root, file)

                features = self.process_file(file_path)

                if features:
                    features["source_file"] = file
                    all_rows.append(features)

        return pd.DataFrame(all_rows)


if __name__ == "__main__":

    engineer = FeatureEngineer()

    df = engineer.process_directory("data/llm_outputs/")

    df.to_csv("data/features/feature_dataset.csv", index=False)

    print(df.head())