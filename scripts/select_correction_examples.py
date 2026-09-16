from pathlib import Path

import pandas as pd


TRAIN_PATH = Path(
    "data/processed/train_annotations.csv"
)

OUTPUT_PATH = Path(
    "data/processed/correction_set.csv"
)

TARGET_COUNTS = {
    "PAYMENT_BILLING": 25,
    "PRICING_PROMOTIONS": 20,
    "TECHNICAL_SUPPORT": 25,
    "ACCOUNT_ACCESS": 20,
    "PRIME_MEMBERSHIP": 20,
    "PRODUCT_CONTENT": 20,
    "RETURN_REFUND": 20,
    "COMPLAINT_FEEDBACK": 10,
}


def main():
    df = pd.read_csv(
        TRAIN_PATH,
        dtype=str
    ).fillna("")

    selected = []

    for intent, count in TARGET_COUNTS.items():
        subset = df[
            df["intent"].str.strip() == intent
        ]

        subset = subset.sample(
            n=min(count, len(subset)),
            random_state=2026
        )

        selected.append(subset)

    correction = pd.concat(
        selected,
        ignore_index=True
    )

    correction = correction.sample(
        frac=1,
        random_state=2026
    ).reset_index(drop=True)

    correction["verified_intent"] = ""
    correction["verification_notes"] = ""

    correction.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("=" * 80)
    print("CORRECTION SET CREATED")
    print("=" * 80)

    print(f"Rows: {len(correction)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nCurrent-label distribution:")
    print(
        correction["intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()