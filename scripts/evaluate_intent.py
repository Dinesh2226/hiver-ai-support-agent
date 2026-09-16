from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


MODEL_PATH = Path(
    "data/processed/intent_model/intent_classifier.joblib"
)

DEV_PATH = Path(
    "data/processed/dev_annotations.csv"
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DEV_PATH.exists():
        raise FileNotFoundError(
            f"Development file not found: {DEV_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(
        DEV_PATH,
        dtype=str
    ).fillna("")

    df["customer_text"] = (
        df["customer_text"]
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["customer_text"] != "")
        & (df["intent"] != "")
    ].copy()

    X = df["customer_text"]
    y_true = df["intent"]

    y_pred = model.predict(X)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("=" * 80)
    print("INTENT CLASSIFIER EVALUATION")
    print("=" * 80)

    print(f"Development examples: {len(df)}")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Macro F1:             {macro_f1:.4f}")
    print(f"Weighted F1:          {weighted_f1:.4f}")

    print("\n" + "=" * 80)
    print("PER-INTENT RESULTS")
    print("=" * 80)

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    header = "True \\ Pred".ljust(24)

    for label in labels:
        header += label[:16].ljust(18)

    print(header)

    for i, label in enumerate(labels):
        row = label.ljust(24)

        for j in range(len(labels)):
            row += str(cm[i, j]).ljust(18)

        print(row)


if __name__ == "__main__":
    main()