import pandas as pd
import numpy as np


class SequenceBuilder:

    def __init__(self, sequence_length=4):
        self.sequence_length = sequence_length

    def create_label(self, value):
        # Proxy distress label
        return 1 if value >= 3 else 0

    def build_sequences(self, df):

        sequences = []
        labels = []

        feature_cols = [
            col for col in df.columns
            if col not in ["source_file", "cik", "year"]
        ]

        grouped = df.sort_values("year").groupby("cik")

        for cik, group in grouped:

            group = group.sort_values("year")

            for i in range(len(group) - self.sequence_length):

                seq = group.iloc[i:i+self.sequence_length][feature_cols].values

                future_value = group.iloc[i+self.sequence_length]["total_risk_count"]

                label = self.create_label(future_value)

                sequences.append(seq)
                labels.append(label)

        return np.array(sequences), np.array(labels)


if __name__ == "__main__":

    df = pd.read_csv("data/features/feature_dataset_with_meta.csv")

    builder = SequenceBuilder(sequence_length=4)

    X, y = builder.build_sequences(df)

    print(f"Total sequences: {len(X)}")
    print(f"Sequence shape: {X[0].shape}")
    print(f"Labels distribution: {np.bincount(y)}")