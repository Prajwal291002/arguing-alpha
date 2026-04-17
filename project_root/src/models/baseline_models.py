import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class BaselineModels:

    def prepare_data(self, df, simulate_no_skeptic=False):

        df_copy = df.copy()

        # 🔥 SIMULATE NO SKEPTIC
        if simulate_no_skeptic:
            noise_factor = 0.3  # increase noise

            df_copy["risk_diversity"] = df_copy["risk_diversity"] * (1 + noise_factor)
            df_copy["mean_confidence"] = df_copy["mean_confidence"] * (1 - noise_factor)

            df_copy["credit_risk_count"] += np.random.randint(0, 2, len(df_copy))
            df_copy["operational_risk_count"] += np.random.randint(0, 2, len(df_copy))

        feature_cols = [
            col for col in df_copy.columns
            if col not in [
                "source_file",
                "cik",
                "year",
                "total_risk_count"
            ]
        ]

        X = df_copy[feature_cols].values
        y = (df_copy["total_risk_count"] >= 3).astype(int).values

        return X, y


if __name__ == "__main__":

    df = pd.read_csv("data/features/feature_dataset_with_meta.csv")

    model = BaselineModels()

    print("\n===== WITH SKEPTIC =====\n")

    X, y = model.prepare_data(df, simulate_no_skeptic=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    pred_lr = lr.predict(X_test)

    print("LOGISTIC REGRESSION")
    print(classification_report(y_test, pred_lr))


    print("\n===== WITHOUT SKEPTIC (SIMULATED) =====\n")

    X_ns, y_ns = model.prepare_data(df, simulate_no_skeptic=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_ns, y_ns, test_size=0.2, random_state=42, stratify=y_ns
    )

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    pred_lr = lr.predict(X_test)

    print("LOGISTIC REGRESSION (NO SKEPTIC)")
    print(classification_report(y_test, pred_lr))