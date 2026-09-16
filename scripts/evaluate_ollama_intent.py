import os
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Make project root importable
sys.path.insert(0, os.path.abspath("."))

from src.intent.ollama_classifier import OllamaIntentClassifier


DEV_FILE = "data/processed/dev_annotations.csv"
OUTPUT_FILE = "data/processed/ollama_intent_predictions.csv"


MESSAGE_COLUMN_CANDIDATES = [
    "text",
    "customer_message",
    "message",
    "tweet_text",
    "customer_text",
    "inbound_text",
    "tweet",
]

INTENT_COLUMN_CANDIDATES = [
    "intent",
    "label",
    "gold_intent",
    "verified_intent",
]


def find_column(df, candidates, column_type):
    """Find a matching column."""

    # Exact match
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # Case-insensitive match
    lower_map = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not find {column_type} column.\n\n"
        f"Available columns:\n{list(df.columns)}\n\n"
        f"Expected one of:\n{candidates}"
    )


def load_dev_data():
    """Load the development dataset."""

    if not os.path.exists(DEV_FILE):
        raise FileNotFoundError(
            f"Development file not found:\n{DEV_FILE}"
        )

    df = pd.read_csv(DEV_FILE)

    print()
    print("Columns detected:")
    print(list(df.columns))

    message_column = find_column(
        df,
        MESSAGE_COLUMN_CANDIDATES,
        "customer message",
    )

    intent_column = find_column(
        df,
        INTENT_COLUMN_CANDIDATES,
        "intent",
    )

    print()
    print(f"Message column : {message_column}")
    print(f"Intent column  : {intent_column}")

    df = df.rename(
        columns={
            message_column: "text",
            intent_column: "intent",
        }
    )

    df = df.dropna(
        subset=["text", "intent"]
    ).copy()

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"] != ""
    ].copy()

    if "row_id" not in df.columns:
        df.insert(
            0,
            "row_id",
            range(len(df)),
        )

    df = df.reset_index(drop=True)

    return df


def print_results(df):
    """Print final evaluation metrics."""

    successful = df[
        df["status"] == "success"
    ].copy()

    if successful.empty:
        print()
        print("=" * 70)
        print("No successful predictions.")
        print("=" * 70)
        return

    y_true = (
        successful["intent"]
        .astype(str)
        .str.strip()
    )

    y_pred = (
        successful["predicted_intent"]
        .astype(str)
        .str.strip()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print()
    print("=" * 70)
    print("OLLAMA INTENT EVALUATION")
    print("=" * 70)

    print(
        f"Examples evaluated : {len(successful)}"
    )

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Macro F1           : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1        : {weighted_f1:.4f}"
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    print(
        confusion_df.to_string()
    )

    # ------------------------------------------------------------
    # Incorrect predictions
    # ------------------------------------------------------------

    wrong = successful[
        successful["intent"]
        != successful["predicted_intent"]
    ]

    print()
    print("=" * 70)
    print(
        f"INCORRECT PREDICTIONS ({len(wrong)})"
    )
    print("=" * 70)

    if wrong.empty:

        print("No incorrect predictions.")

    else:

        for _, row in wrong.iterrows():

            print()
            print(
                f"Row ID      : {row['row_id']}"
            )

            print(
                f"Message     : {row['text']}"
            )

            print(
                f"True intent : {row['intent']}"
            )

            print(
                f"Prediction  : "
                f"{row['predicted_intent']}"
            )

            print(
                f"Confidence  : "
                f"{row['confidence']}"
            )

            print(
                f"Reason      : "
                f"{row['reason']}"
            )


def main():

    print("=" * 70)
    print("OLLAMA QWEN3 INTENT CLASSIFIER EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load development dataset
    # ------------------------------------------------------------

    df = load_dev_data()

    print()
    print(
        f"Development examples : {len(df)}"
    )

    print(
        f"Output file           : {OUTPUT_FILE}"
    )

    # ------------------------------------------------------------
    # Prepare result dataframe
    # ------------------------------------------------------------

    result_df = df.copy()

    result_df["predicted_intent"] = None
    result_df["confidence"] = float("nan")
    result_df["reason"] = None
    result_df["status"] = "pending"
    result_df["error"] = None

    # ------------------------------------------------------------
    # Load previous predictions
    # ------------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        try:
            previous = pd.read_csv(
                OUTPUT_FILE
            )

            if "row_id" in previous.columns:

                print()
                print(
                    f"Previous predictions found: "
                    f"{len(previous)}"
                )

                previous_columns = [
                    "row_id",
                    "predicted_intent",
                    "confidence",
                    "reason",
                    "status",
                    "error",
                ]

                previous_columns = [
                    col
                    for col in previous_columns
                    if col in previous.columns
                ]

                previous = previous[
                    previous_columns
                ]

                result_df = result_df.drop(
                    columns=[
                        "predicted_intent",
                        "confidence",
                        "reason",
                        "status",
                        "error",
                    ]
                )

                result_df = result_df.merge(
                    previous,
                    on="row_id",
                    how="left",
                )

                # Restore missing columns
                if "predicted_intent" not in result_df.columns:
                    result_df["predicted_intent"] = None

                if "confidence" not in result_df.columns:
                    result_df["confidence"] = float("nan")

                if "reason" not in result_df.columns:
                    result_df["reason"] = None

                if "status" not in result_df.columns:
                    result_df["status"] = "pending"

                if "error" not in result_df.columns:
                    result_df["error"] = None

                result_df["confidence"] = pd.to_numeric(
                    result_df["confidence"],
                    errors="coerce",
                )

                result_df["status"] = (
                    result_df["status"]
                    .fillna("pending")
                    .astype(str)
                )

        except Exception as exc:

            print(
                f"Warning: could not restore previous "
                f"predictions: {exc}"
            )

    # ------------------------------------------------------------
    # Find rows still needing classification
    # ------------------------------------------------------------

    pending_indices = list(
        result_df.index[
            result_df["status"] != "success"
        ]
    )

    successful_count = (
        len(result_df)
        - len(pending_indices)
    )

    print()
    print(
        f"Already successful : "
        f"{successful_count}"
    )

    print(
        f"Remaining          : "
        f"{len(pending_indices)}"
    )

    if not pending_indices:

        print()
        print(
            "All examples have already been evaluated."
        )

        print_results(
            result_df
        )

        return

    # ------------------------------------------------------------
    # Initialize Ollama
    # ------------------------------------------------------------

    print()
    print(
        "Initializing Ollama..."
    )

    classifier = (
        OllamaIntentClassifier(
            model="qwen3:8b"
        )
    )

    print(
        "Ollama classifier ready."
    )

    # ------------------------------------------------------------
    # Process examples
    # ------------------------------------------------------------

    for position, index in enumerate(
        pending_indices,
        start=1,
    ):

        row = result_df.loc[index]

        message = str(
            row["text"]
        ).strip()

        true_intent = str(
            row["intent"]
        ).strip()

        print()
        print("-" * 70)

        print(
            f"Processing "
            f"{position}/"
            f"{len(pending_indices)}"
        )

        print(
            f"Row ID      : "
            f"{row['row_id']}"
        )

        print(
            f"True intent : "
            f"{true_intent}"
        )

        print(
            f"Message     : "
            f"{message}"
        )

        try:

            result = classifier.classify(
                message
            )

            predicted_intent = (
                result["intent"]
            )

            confidence = float(
                result["confidence"]
            )

            reason = result["reason"]

            result_df.at[
                index,
                "predicted_intent",
            ] = predicted_intent

            result_df.at[
                index,
                "confidence",
            ] = confidence

            result_df.at[
                index,
                "reason",
            ] = reason

            result_df.at[
                index,
                "status",
            ] = "success"

            result_df.at[
                index,
                "error",
            ] = None

            prediction_status = (
                "CORRECT"
                if predicted_intent == true_intent
                else "WRONG"
            )

            print(
                f"Prediction  : "
                f"{predicted_intent}"
            )

            print(
                f"Confidence  : "
                f"{confidence:.2f}"
            )

            print(
                f"Result      : "
                f"{prediction_status}"
            )

        except Exception as exc:

            error_message = str(
                exc
            )

            result_df.at[
                index,
                "status",
            ] = "error"

            result_df.at[
                index,
                "error",
            ] = error_message

            print(
                f"ERROR       : "
                f"{error_message}"
            )

        # --------------------------------------------------------
        # Save after every example
        # --------------------------------------------------------

        os.makedirs(
            os.path.dirname(OUTPUT_FILE),
            exist_ok=True,
        )

        result_df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

    # ------------------------------------------------------------
    # Final results
    # ------------------------------------------------------------

    print_results(
        result_df
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Predictions saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()