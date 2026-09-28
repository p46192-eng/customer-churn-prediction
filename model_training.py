"""Train and evaluate the customer churn model.

Dataset source:
UCI Machine Learning Repository - Iranian Churn, dataset ID 563.
https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
"""

import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

RANDOM_STATE = 42


def load_data():
    dataset = fetch_ucirepo(id=563)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Keep a clean, consistent target name.
    y.columns = ["Churn"]

    # Remove an identifier if the repository version exposes one.
    id_columns = [c for c in X.columns if str(c).strip().lower() in {"customer id", "customer_id", "id"}]
    if id_columns:
        X = X.drop(columns=id_columns)

    return X, y["Churn"].astype(int)


def build_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_probability),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
    }

    print("Rows:", len(X))
    print("Features:", list(X.columns))
    print("Churn distribution:")
    print(y.value_counts())
    print("\nModel performance:")
    for key, value in metrics.items():
        if key not in {"confusion_matrix", "classification_report"}:
            print(f"{key}: {value:.3f}")

    print("\nConfusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    # Saved for reproducibility and future deployment use.
    joblib.dump(
        {
            "model": model,
            "feature_names": list(X.columns),
            "metrics": metrics,
        },
        "customer_churn_model.pkl",
    )
    print("\nSaved customer_churn_model.pkl")


if __name__ == "__main__":
    main()
