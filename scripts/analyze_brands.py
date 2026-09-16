from pathlib import Path
from collections import Counter
import pandas as pd


DATA_PATH = Path("data/raw/twcs.csv")

CHUNK_SIZE = 100_000


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    brand_counts = Counter()
    inbound_counts = Counter()
    outbound_counts = Counter()

    total_rows = 0
    total_inbound = 0
    total_outbound = 0

    print("Starting brand analysis...")
    print(f"Dataset: {DATA_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print()

    for chunk_number, chunk in enumerate(
        pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
        start=1
    ):
        total_rows += len(chunk)

        inbound_mask = chunk["inbound"].eq(True)
        outbound_mask = chunk["inbound"].eq(False)

        inbound = chunk[inbound_mask]
        outbound = chunk[outbound_mask]

        # author_id identifies the brand on outbound tweets.
        # On inbound tweets, it identifies the customer.
        for author in outbound["author_id"].dropna():
            brand_counts[str(author)] += 1

        for author in inbound["author_id"].dropna():
            inbound_counts[str(author)] += 1

        for author in outbound["author_id"].dropna():
            outbound_counts[str(author)] += 1

        total_inbound += len(inbound)
        total_outbound += len(outbound)

        if chunk_number % 5 == 0:
            print(
                f"Processed {total_rows:,} rows | "
                f"inbound={total_inbound:,} | "
                f"outbound={total_outbound:,}"
            )

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(f"Total rows:     {total_rows:,}")
    print(f"Inbound tweets: {total_inbound:,}")
    print(f"Outbound tweets:{total_outbound:,}")

    print("\n" + "=" * 80)
    print("TOP BRANDS BY OUTBOUND TWEETS")
    print("=" * 80)

    top_brands = brand_counts.most_common(30)

    print(
        f"{'Rank':<6}"
        f"{'Brand':<30}"
        f"{'Outbound':>15}"
        f"{'Ratio':>12}"
    )

    print("-" * 65)

    for rank, (brand, count) in enumerate(top_brands, start=1):
        ratio = (count / total_outbound) * 100 if total_outbound else 0

        print(
            f"{rank:<6}"
            f"{brand:<30}"
            f"{count:>15,}"
            f"{ratio:>11.2f}%"
        )

    print("\n" + "=" * 80)
    print("KNOWN CANDIDATE BRANDS")
    print("=" * 80)

    candidates = [
        "AmazonHelp",
        "AppleSupport",
        "Uber_Support",
        "SpotifyCares",
        "MicrosoftHelps",
        "AskPlayStation",
        "sprintcare",
    ]

    print(
        f"{'Brand':<25}"
        f"{'Outbound':>15}"
        f"{'Inbound':>15}"
    )

    print("-" * 55)

    for brand in candidates:
        print(
            f"{brand:<25}"
            f"{outbound_counts.get(brand, 0):>15,}"
            f"{inbound_counts.get(brand, 0):>15,}"
        )


if __name__ == "__main__":
    main()