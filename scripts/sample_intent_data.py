from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/processed/amazonhelp_pairs.csv")
OUTPUT_PATH = Path("data/processed/intent_review_sample.csv")

SAMPLE_SIZE = 1000


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    # Remove duplicate customer messages so repeated
    # messages from the same tweet do not dominate review.
    df = df.drop_duplicates(
        subset=["customer_tweet_id"]
    )

    sample_size = min(SAMPLE_SIZE, len(df))

    sample = df.sample(
        n=sample_size,
        random_state=42
    ).copy()

    sample = sample[
        [
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_text",
            "brand_reply_text",
        ]
    ]

    # Empty annotation columns for manual labeling.
    sample["intent"] = ""
    sample["intent_notes"] = ""

    sample.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("=" * 80)
    print("INTENT REVIEW SAMPLE CREATED")
    print("=" * 80)
    print(f"Source rows: {len(df):,}")
    print(f"Sample rows: {len(sample):,}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nIntent labels to use:")
    print("ORDER_DELIVERY")
    print("RETURN_REFUND")
    print("ACCOUNT_ACCESS")
    print("PAYMENT_BILLING")
    print("TECHNICAL_SUPPORT")
    print("PRODUCT_CONTENT")
    print("PRICING_PROMOTIONS")
    print("COMPLAINT_FEEDBACK")
    print("OTHER")


if __name__ == "__main__":
    main()