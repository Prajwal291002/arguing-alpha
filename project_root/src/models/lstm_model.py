import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class LSTMModel:

    def build_model(self, input_shape):

        model = Sequential()

        model.add(LSTM(32, input_shape=input_shape))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model


if __name__ == "__main__":

    from src.features.sequence_builder import SequenceBuilder
    import pandas as pd

    df = pd.read_csv("data/features/feature_dataset_with_meta.csv")

    builder = SequenceBuilder(sequence_length=4)
    X, y = builder.build_sequences(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weights
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = dict(zip(classes, weights))

    print(f"Class Weights: {class_weights}")

    model = LSTMModel().build_model(input_shape=(X.shape[1], X.shape[2]))

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=8,
        class_weight=class_weights
    )

    predictions = (model.predict(X_test) > 0.5).astype(int)

    print("\n=== CLASSIFICATION REPORT ===\n")
    print(classification_report(y_test, predictions))