from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/processed/amazonhelp_pairs.csv")

BRAND = "AmazonHelp"
CHUNK_SIZE = 100_000


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXTRACTING AMAZONHELP CUSTOMER → BRAND PAIRS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # PASS 1
    # Find all AmazonHelp replies and the customer tweet IDs they
    # respond to.
    # ------------------------------------------------------------------

    reply_map = {}

    total_brand_replies = 0

    columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id",
    ]

    print("\nPASS 1: Finding AmazonHelp replies...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=columns,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        brand_rows = chunk[
            (chunk["inbound"] == False)
            & (chunk["author_id"] == BRAND)
        ]

        for _, row in brand_rows.iterrows():
            parent_id = pd.to_numeric(
                row["in_response_to_tweet_id"],
                errors="coerce",
            )

            if pd.isna(parent_id):
                continue

            parent_id = int(parent_id)

            reply_map.setdefault(parent_id, []).append(
                {
                    "reply_tweet_id": int(row["tweet_id"]),
                    "reply_text": str(row["text"]),
                    "reply_created_at": str(row["created_at"]),
                }
            )

            total_brand_replies += 1

        if chunk_number % 5 == 0:
            print(
                f"Processed {chunk_number * CHUNK_SIZE:,} rows..."
            )

    customer_ids = set(reply_map.keys())

    print("\nPASS 1 complete.")
    print(f"AmazonHelp replies: {total_brand_replies:,}")
    print(f"Referenced customer tweets: {len(customer_ids):,}")

    # ------------------------------------------------------------------
    # PASS 2
    # Find the actual customer tweets referenced by AmazonHelp.
    # ------------------------------------------------------------------

    print("\nPASS 2: Extracting customer messages...")

    customer_rows = {}

    columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=columns,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        inbound = chunk[chunk["inbound"] == True]

        matches = inbound[
            inbound["tweet_id"].isin(customer_ids)
        ]

        for _, row in matches.iterrows():
            tweet_id = int(row["tweet_id"])

            customer_rows[tweet_id] = {
                "customer_tweet_id": tweet_id,
                "customer_author_id": str(row["author_id"]),
                "customer_created_at": str(row["created_at"]),
                "customer_text": str(row["text"]),
            }

        if chunk_number % 5 == 0:
            print(
                f"Processed {chunk_number * CHUNK_SIZE:,} rows..."
            )

    print("\nPASS 2 complete.")
    print(f"Customer tweets found: {len(customer_rows):,}")

    # ------------------------------------------------------------------
    # BUILD CUSTOMER → BRAND PAIRS
    # ------------------------------------------------------------------

    records = []

    for customer_tweet_id, replies in reply_map.items():

        customer = customer_rows.get(customer_tweet_id)

        if customer is None:
            continue

        for reply in replies:

            records.append(
                {
                    "customer_tweet_id": customer[
                        "customer_tweet_id"
                    ],
                    "customer_author_id": customer[
                        "customer_author_id"
                    ],
                    "customer_created_at": customer[
                        "customer_created_at"
                    ],
                    "customer_text": customer[
                        "customer_text"
                    ],
                    "brand": BRAND,
                    "brand_reply_tweet_id": reply[
                        "reply_tweet_id"
                    ],
                    "brand_reply_created_at": reply[
                        "reply_created_at"
                    ],
                    "brand_reply_text": reply[
                        "reply_text"
                    ],
                }
            )

    pairs = pd.DataFrame(records)

    # Remove empty messages.
    pairs = pairs[
        pairs["customer_text"].notna()
        & pairs["brand_reply_text"].notna()
    ]

    pairs["customer_text"] = (
        pairs["customer_text"]
        .astype(str)
        .str.strip()
    )

    pairs["brand_reply_text"] = (
        pairs["brand_reply_text"]
        .astype(str)
        .str.strip()
    )

    pairs = pairs[
        (pairs["customer_text"] != "")
        & (pairs["brand_reply_text"] != "")
    ]

    pairs.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

    print(f"Output: {OUTPUT_PATH}")
    print(f"Pairs: {len(pairs):,}")

    print("\nColumns:")
    for column in pairs.columns:
        print(f"  - {column}")

    print("\nSample:")
    print(
        pairs[
            [
                "customer_text",
                "brand_reply_text",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()