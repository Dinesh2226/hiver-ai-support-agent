from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/processed/amazonhelp_pairs.csv")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("=" * 80)
    print("AMAZONHELP DATASET INSPECTION")
    print("=" * 80)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print(
        f"\nUnique customer tweets: "
        f"{df['customer_tweet_id'].nunique():,}"
    )

    print(
        f"Unique customers: "
        f"{df['customer_author_id'].nunique():,}"
    )

    print("\nCustomer message length statistics:")
    print(
        df["customer_text"]
        .astype(str)
        .str.len()
        .describe()
        .to_string()
    )

    print("\n" + "=" * 80)
    print("50 RANDOM CUSTOMER → AMAZONHELP EXAMPLES")
    print("=" * 80)

    sample_size = min(50, len(df))

    sample = df[
        ["customer_text", "brand_reply_text"]
    ].sample(
        n=sample_size,
        random_state=42
    )

    for i, (_, row) in enumerate(
        sample.iterrows(),
        start=1
    ):
        print(f"\n--- Example {i} ---")
        print(f"Customer : {row['customer_text']}")
        print(f"AmazonHelp: {row['brand_reply_text']}")


if __name__ == "__main__":
    main()