import os
from typing import Dict, List, Optional, Set

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalReplyRetriever:
    """
    Retrieves historically similar AmazonHelp customer-support
    examples using TF-IDF cosine similarity.

    Expected CSV columns:
        customer_text
        brand_reply_text

    The optional `exclude_texts` argument allows evaluation code to
    exclude test/golden examples from the retrieval candidate pool,
    preventing retrieval leakage.
    """

    def __init__(
        self,
        csv_path: str,
        max_features: int = 50000,
        ngram_range=(1, 2),
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Historical data file not found: {csv_path}"
            )

        self.csv_path = csv_path

        self.df = pd.read_csv(csv_path)

        required_columns = {
            "customer_text",
            "brand_reply_text",
        }

        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        # Keep only rows that contain both customer message and reply.
        self.df = self.df.dropna(
            subset=[
                "customer_text",
                "brand_reply_text",
            ]
        ).copy()

        self.df["customer_text"] = (
            self.df["customer_text"]
            .astype(str)
            .str.strip()
        )

        self.df["brand_reply_text"] = (
            self.df["brand_reply_text"]
            .astype(str)
            .str.strip()
        )

        # Remove empty rows.
        self.df = self.df[
            (self.df["customer_text"] != "")
            & (self.df["brand_reply_text"] != "")
        ].reset_index(drop=True)

        print(
            f"Loaded {len(self.df)} historical "
            "customer-reply examples."
        )

        # TF-IDF representation of customer messages.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["customer_text"]
        )

    def retrieve(
        self,
        customer_message: str,
        top_k: int = 5,
        min_score: float = 0.05,
        exclude_texts: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """
        Retrieve historically similar customer-support examples.

        Parameters
        ----------
        customer_message:
            Current customer message.

        top_k:
            Maximum number of historical examples to return.

        min_score:
            Minimum cosine similarity required.

        exclude_texts:
            Optional set of customer messages to exclude from retrieval.
            This is primarily used during golden/test evaluation to
            prevent exact test examples from being retrieved.

        Returns
        -------
        List[Dict]
            Each result contains:
                customer_text
                brand_reply_text
                similarity
        """

        if not isinstance(customer_message, str):
            raise TypeError(
                "customer_message must be a string."
            )

        customer_message = customer_message.strip()

        if not customer_message:
            return []

        # Normalize excluded messages.
        normalized_exclusions = {
            str(text).strip()
            for text in (exclude_texts or set())
            if str(text).strip()
        }

        # Convert query into TF-IDF vector.
        query_vector = self.vectorizer.transform(
            [customer_message]
        )

        # Calculate cosine similarity against all historical examples.
        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).flatten()

        # Highest similarity first.
        ranked_indices = scores.argsort()[::-1]

        results = []

        for idx in ranked_indices:
            row = self.df.iloc[idx]

            historical_customer_text = (
                str(row["customer_text"]).strip()
            )

            # ---------------------------------------------------------
            # Evaluation leakage protection
            # ---------------------------------------------------------
            # If this historical customer message belongs to the
            # golden/test set, skip it.
            if historical_customer_text in normalized_exclusions:
                continue

            score = float(scores[idx])

            # Because ranking is descending, once we go below
            # min_score we can stop.
            if score < min_score:
                break

            results.append(
                {
                    "customer_text": historical_customer_text,
                    "brand_reply_text": str(
                        row["brand_reply_text"]
                    ).strip(),
                    "similarity": score,
                }
            )

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    DATA_FILE = (
        "data/processed/"
        "amazonhelp_pairs.csv"
    )

    retriever = HistoricalReplyRetriever(
        DATA_FILE
    )

    test_message = (
        "My package was supposed to arrive "
        "yesterday but it still hasn't arrived."
    )

    results = retriever.retrieve(
        test_message,
        top_k=5,
    )

    print()
    print("=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70)

    if not results:
        print("No historical matches found.")
    else:
        for i, result in enumerate(
            results,
            start=1,
        ):
            print()
            print(f"Result {i}")
            print(
                f"Similarity: "
                f"{result['similarity']:.4f}"
            )
            print(
                f"Customer: "
                f"{result['customer_text']}"
            )
            print(
                f"AmazonHelp: "
                f"{result['brand_reply_text']}"
            )