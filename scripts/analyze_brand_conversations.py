from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/twcs.csv")

CHUNK_SIZE = 100_000

CANDIDATES = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "TMobileHelp",
    "comcastcares",
    "XboxSupport",
    "AskPlayStation",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("=" * 80)
    print("BRAND CONVERSATION ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {DATA_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print()

    # ------------------------------------------------------------------
    # PASS 1:
    # Collect outbound brand tweets and the customer tweet IDs they
    # directly respond to.
    # ------------------------------------------------------------------

    brand_reply_to_customer = {
        brand: set()
        for brand in CANDIDATES
    }

    brand_outbound_count = {
        brand: 0
        for brand in CANDIDATES
    }

    total_rows = 0

    print("PASS 1: Finding brand replies...")

    usecols = [
        "tweet_id",
        "author_id",
        "inbound",
        "in_response_to_tweet_id",
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=usecols,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):
        total_rows += len(chunk)

        outbound = chunk[chunk["inbound"] == False]

        for brand in CANDIDATES:
            brand_rows = outbound[outbound["author_id"] == brand]

            if brand_rows.empty:
                continue

            brand_outbound_count[brand] += len(brand_rows)

            response_ids = (
                pd.to_numeric(
                    brand_rows["in_response_to_tweet_id"],
                    errors="coerce"
                )
                .dropna()
                .astype("int64")
                .tolist()
            )

            brand_reply_to_customer[brand].update(response_ids)

        if chunk_number % 5 == 0:
            print(f"Processed {total_rows:,} rows...")

    print("\nPASS 1 complete.")

    # ------------------------------------------------------------------
    # PASS 2:
    # Verify that the referenced tweets are actually inbound/customer
    # tweets.
    # ------------------------------------------------------------------

    customer_ids = set()

    for brand in CANDIDATES:
        customer_ids.update(
            brand_reply_to_customer[brand]
        )

    print()
    print(f"Unique customer tweet IDs referenced: {len(customer_ids):,}")
    print()

    verified_customer_ids = set()

    print("PASS 2: Verifying customer tweets...")

    total_rows = 0

    usecols = [
        "tweet_id",
        "inbound",
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=usecols,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):
        total_rows += len(chunk)

        inbound = chunk[chunk["inbound"] == True]

        matches = inbound[
            inbound["tweet_id"].isin(customer_ids)
        ]

        if not matches.empty:
            verified_customer_ids.update(
                matches["tweet_id"].astype("int64").tolist()
            )

        if chunk_number % 5 == 0:
            print(f"Verified {total_rows:,} rows...")

    print("\nPASS 2 complete.")

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("BRAND RESULTS")
    print("=" * 80)

    header = (
        f"{'Brand':<22}"
        f"{'Replies':>12}"
        f"{'Unique Customers':>20}"
        f"{'Pair Coverage':>18}"
    )

    print(header)
    print("-" * len(header))

    results = []

    for brand in CANDIDATES:
        replies = brand_outbound_count[brand]

        referenced = brand_reply_to_customer[brand]

        verified = referenced.intersection(
            verified_customer_ids
        )

        unique_customers = len(verified)

        coverage = (
            unique_customers / replies * 100
            if replies
            else 0
        )

        results.append(
            (
                brand,
                replies,
                unique_customers,
                coverage
            )
        )

    results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    for brand, replies, customers, coverage in results:
        print(
            f"{brand:<22}"
            f"{replies:>12,}"
            f"{customers:>20,}"
            f"{coverage:>17.2f}%"
        )

    print("\n" + "=" * 80)
    print("RECOMMENDATION INPUT")
    print("=" * 80)

    print(
        "\nWe will use the following information to choose the final brand:"
    )

    print(
        "1. Number of outbound support replies"
    )
    print(
        "2. Number of verified customer messages"
    )
    print(
        "3. Number of customer→brand pairs"
    )
    print(
        "4. Conversation diversity and quality"
    )


if __name__ == "__main__":
    main()