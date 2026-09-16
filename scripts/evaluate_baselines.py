from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


BASE = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE / "data" / "processed" / "train_annotations.csv"
GOLDEN_FILE = BASE / "data" / "golden" / "golden_set.csv"
OUT_FILE = BASE / "data" / "golden" / "baseline_results.txt"


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def evaluate(y_true, y_pred, name):
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1:          {f1:.4f}")
    print(f"Weighted Precision:{weighted_precision:.4f}")
    print(f"Weighted Recall:   {weighted_recall:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")

    print()
    print(classification_report(
        y_true,
        y_pred,
        zero_division=0,
    ))

    return {
        "name": name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": precision,
        "weighted_f1": weighted_f1,
    }


def main():
    train = pd.read_csv(TRAIN_FILE)
    golden = pd.read_csv(GOLDEN_FILE)

    train_text_col = find_column(
        train,
        ["customer_text", "text", "message"],
    )

    train_label_col = find_column(
        train,
        ["intent", "label"],
    )

    gold_text_col = find_column(
        golden,
        ["customer_text", "text", "message"],
    )

    gold_label_col = find_column(
        golden,
        ["intent", "label"],
    )

    train = train.dropna(
        subset=[train_text_col, train_label_col]
    ).copy()

    golden = golden.dropna(
        subset=[gold_text_col, gold_label_col]
    ).copy()

    train[train_text_col] = (
        train[train_text_col]
        .astype(str)
        .str.strip()
    )

    golden[gold_text_col] = (
        golden[gold_text_col]
        .astype(str)
        .str.strip()
    )

    train[train_label_col] = (
        train[train_label_col]
        .astype(str)
        .str.strip()
    )

    golden[gold_label_col] = (
        golden[gold_label_col]
        .astype(str)
        .str.strip()
    )

    X_train = train[train_text_col]
    y_train = train[train_label_col]

    X_gold = golden[gold_text_col]
    y_gold = golden[gold_label_col]

    print("=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    print(f"Training examples: {len(train)}")
    print(f"Golden examples:   {len(golden)}")
    print()

    # ------------------------------------------------------------
    # 1. TRIVIAL BASELINE
    # ------------------------------------------------------------
    majority_intent = y_train.value_counts().idxmax()

    print(f"Majority training intent: {majority_intent}")
    print()

    majority_predictions = [
        majority_intent
        for _ in range(len(y_gold))
    ]

    majority_metrics = evaluate(
        y_gold,
        majority_predictions,
        "1. TRIVIAL BASELINE — MAJORITY CLASS",
    )

    # ------------------------------------------------------------
    # 2. SIMPLE BASELINE
    # ------------------------------------------------------------
    tfidf_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print()
    print("Training TF-IDF + Logistic Regression...")

    tfidf_model.fit(X_train, y_train)

    tfidf_predictions = tfidf_model.predict(X_gold)

    simple_metrics = evaluate(
        y_gold,
        tfidf_predictions,
        "2. SIMPLE BASELINE — TF-IDF + LOGISTIC REGRESSION",
    )

    # ------------------------------------------------------------
    # 3. CONFUSION MATRIX
    # ------------------------------------------------------------
    labels = sorted(y_gold.unique())

    cm = confusion_matrix(
        y_gold,
        tfidf_predictions,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"gold_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )

    print()
    print("=" * 70)
    print("TF-IDF CONFUSION MATRIX")
    print("=" * 70)
    print(cm_df.to_string())

    # ------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------
    lines = []

    lines.append("BASELINE EVALUATION")
    lines.append("=" * 70)
    lines.append(f"Training examples: {len(train)}")
    lines.append(f"Golden examples: {len(golden)}")
    lines.append("")
    lines.append(
        f"Majority training intent: {majority_intent}"
    )
    lines.append("")

    for result in [majority_metrics, simple_metrics]:
        lines.append(result["name"])
        lines.append("-" * 70)
        lines.append(
            f"Accuracy: {result['accuracy']:.4f}"
        )
        lines.append(
            f"Macro Precision: {result['macro_precision']:.4f}"
        )
        lines.append(
            f"Macro Recall: {result['macro_recall']:.4f}"
        )
        lines.append(
            f"Macro F1: {result['macro_f1']:.4f}"
        )
        lines.append(
            f"Weighted F1: {result['weighted_f1']:.4f}"
        )
        lines.append("")

    lines.append("TF-IDF CONFUSION MATRIX")
    lines.append("-" * 70)
    lines.append(cm_df.to_string())

    OUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(f"Results saved to: {OUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()