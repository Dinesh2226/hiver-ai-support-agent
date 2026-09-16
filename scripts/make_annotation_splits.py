from pathlib import Path
import pandas as pd


INPUT_PATH = Path(
    "data/processed/intent_review_sample.csv"
)

TRAIN_PATH = Path(
    "data/processed/train_annotations.csv"
)

DEV_PATH = Path(
    "data/processed/dev_annotations.csv"
)

GOLDEN_PATH = Path(
    "data/golden/golden_set.csv"
)


TRAIN_SIZE = 700
DEV_SIZE = 100
GOLDEN_SIZE = 200


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    # Remove duplicate customer tweets.
    df = df.drop_duplicates(
        subset=["customer_tweet_id"]
    ).reset_index(drop=True)

    required_size = TRAIN_SIZE + DEV_SIZE + GOLDEN_SIZE

    if len(df) < required_size:
        raise ValueError(
            f"Need {required_size} rows, "
            f"but only {len(df)} are available."
        )

    # Stable random split.
    df = df.sample(
        n=required_size,
        random_state=2026
    ).reset_index(drop=True)

    golden = df.iloc[
        :GOLDEN_SIZE
    ].copy()

    dev = df.iloc[
        GOLDEN_SIZE:GOLDEN_SIZE + DEV_SIZE
    ].copy()

    train = df.iloc[
        GOLDEN_SIZE + DEV_SIZE:
    ].copy()

    # Keep annotation columns.
    for data in (train, dev, golden):
        data["intent"] = ""
        data["intent_notes"] = ""

    # Golden set gets extra evaluation fields.
    golden["escalation"] = ""
    golden["escalation_reason"] = ""
    golden["ideal_reply_points"] = ""
    golden["difficulty"] = ""

    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    DEV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    GOLDEN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_PATH,
        index=False,
        encoding="utf-8"
    )

    dev.to_csv(
        DEV_PATH,
        index=False,
        encoding="utf-8"
    )

    golden.to_csv(
        GOLDEN_PATH,
        index=False,
        encoding="utf-8"
    )

    print("=" * 80)
    print("ANNOTATION SPLIT CREATED")
    print("=" * 80)

    print(f"Training: {len(train):,}")
    print(f"Development: {len(dev):,}")
    print(f"Golden: {len(golden):,}")

    print("\nFiles:")
    print(f"  {TRAIN_PATH}")
    print(f"  {DEV_PATH}")
    print(f"  {GOLDEN_PATH}")

    print("\nImportant:")
    print("The golden set must NOT be used for training or tuning.")


if __name__ == "__main__":
    main()