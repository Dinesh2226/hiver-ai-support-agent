from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path(
    "data/processed/train_annotations.csv"
)

MODEL_DIR = Path("data/processed/intent_model")
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}"
        )

    df = pd.read_csv(
        TRAIN_PATH,
        dtype=str
    ).fillna("")

    # Keep only valid labeled rows.
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

    print("=" * 80)
    print("TRAINING INTENT CLASSIFIER")
    print("=" * 80)

    print(f"Training examples: {len(df):,}")
    print(f"Intent classes: {df['intent'].nunique()}")

    print("\nClass distribution:")
    print(
        df["intent"]
        .value_counts()
        .to_string()
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            )
        ),
    ])

    pipeline.fit(
        df["customer_text"],
        df["intent"]
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(f"Model saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()