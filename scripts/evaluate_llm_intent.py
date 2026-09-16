import os
import sys
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ================================================================
# Make project root importable
# ================================================================

sys.path.insert(0, os.path.abspath("."))

from src.intent.llm_classifier import GeminiIntentClassifier


# ================================================================
# Configuration
# ================================================================

DEV_FILE = "data/processed/dev_annotations.csv"
OUTPUT_FILE = "data/processed/llm_intent_predictions.csv"

# Keep below the reported 20-request quota.
# 4.5 sec ≈ 13 requests/minute.
REQUEST_DELAY_SECONDS = 4.5


# Possible customer-message column names
MESSAGE_COLUMN_CANDIDATES = [
    "text",
    "customer_message",
    "message",
    "tweet_text",
    "customer_text",
    "inbound_text",
    "tweet",
]

# Possible intent column names
INTENT_COLUMN_CANDIDATES = [
    "intent",
    "label",
    "gold_intent",
    "verified_intent",
]


# ================================================================
# Find a column
# ================================================================

def find_column(df, candidates, column_type):
    """
    Find a matching column using exact or case-insensitive matching.
    """

    # Exact match
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # Case-insensitive match
    lowercase_map = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().strip()

        if key in lowercase_map:
            return lowercase_map[key]

    raise ValueError(
        f"Could not find {column_type} column.\n\n"
        f"Available columns:\n{list(df.columns)}\n\n"
        f"Expected one of:\n{candidates}"
    )


# ================================================================
# Load development data
# ================================================================

def load_development_data():
    """
    Load the development dataset and normalize column names.
    """

    if not os.path.exists(DEV_FILE):
        raise FileNotFoundError(
            f"Development file not found:\n{DEV_FILE}"
        )

    df = pd.read_csv(DEV_FILE)

    print()
    print("Columns detected in development file:")
    print(list(df.columns))

    # Find columns
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

    # Normalize names internally
    df = df.rename(
        columns={
            message_column: "text",
            intent_column: "intent",
        }
    )

    # Remove missing values
    df = df.dropna(
        subset=["text", "intent"]
    ).copy()

    # Normalize strings
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

    # Remove empty messages
    df = df[
        df["text"] != ""
    ].copy()

    # Stable ID
    if "row_id" not in df.columns:
        df.insert(
            0,
            "row_id",
            range(len(df)),
        )

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ================================================================
# Load previous predictions
# ================================================================

def load_previous_predictions():
    """
    Load existing predictions if available.
    """

    if not os.path.exists(OUTPUT_FILE):
        return pd.DataFrame()

    try:
        previous = pd.read_csv(
            OUTPUT_FILE
        )

        if "row_id" not in previous.columns:
            print(
                "Previous prediction file has no row_id. "
                "Ignoring it."
            )
            return pd.DataFrame()

        return previous

    except Exception as exc:
        print(
            f"Warning: could not read previous predictions: {exc}"
        )

        return pd.DataFrame()


# ================================================================
# Prepare result dataframe
# ================================================================

def prepare_result_dataframe(df, previous):
    """
    Create a consistent result dataframe and safely restore
    previous predictions.
    """

    result_df = df.copy()

    # ------------------------------------------------------------
    # Initialize with correct data types.
    # ------------------------------------------------------------

    result_df["predicted_intent"] = pd.Series(
        [None] * len(result_df),
        dtype="object",
    )

    result_df["confidence"] = pd.Series(
        [float("nan")] * len(result_df),
        dtype="float64",
    )

    result_df["reason"] = pd.Series(
        [None] * len(result_df),
        dtype="object",
    )

    result_df["status"] = pd.Series(
        ["pending"] * len(result_df),
        dtype="object",
    )

    result_df["error"] = pd.Series(
        [None] * len(result_df),
        dtype="object",
    )

    # ------------------------------------------------------------
    # Restore previous predictions.
    # ------------------------------------------------------------

    if not previous.empty:

        previous_columns = [
            "row_id",
            "predicted_intent",
            "confidence",
            "reason",
            "status",
            "error",
        ]

        available_columns = [
            column
            for column in previous_columns
            if column in previous.columns
        ]

        previous_clean = previous[
            available_columns
        ].copy()

        # Make sure row_id types match.
        result_df["row_id"] = pd.to_numeric(
            result_df["row_id"],
            errors="coerce",
        )

        previous_clean["row_id"] = pd.to_numeric(
            previous_clean["row_id"],
            errors="coerce",
        )

        # Remove old result columns before merge.
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
            previous_clean,
            on="row_id",
            how="left",
        )

    # ------------------------------------------------------------
    # Ensure all result columns exist.
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # IMPORTANT:
    # Restore proper pandas dtypes after CSV loading.
    # ------------------------------------------------------------

    result_df["confidence"] = pd.to_numeric(
        result_df["confidence"],
        errors="coerce",
    )

    result_df["predicted_intent"] = (
        result_df["predicted_intent"]
        .astype("object")
    )

    result_df["reason"] = (
        result_df["reason"]
        .astype("object")
    )

    result_df["status"] = (
        result_df["status"]
        .fillna("pending")
        .astype(str)
    )

    result_df["error"] = (
        result_df["error"]
        .astype("object")
    )

    return result_df


# ================================================================
# Save predictions
# ================================================================

def save_predictions(df):
    """
    Save predictions after every API request.
    """

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ================================================================
# Check whether an error is a quota error
# ================================================================

def is_quota_error(error_message):
    """
    Detect Gemini quota/rate-limit errors.
    """

    text = error_message.lower()

    quota_signals = [
        "429",
        "too_many_requests",
        "quota exceeded",
        "rate limit",
        "resource_exhausted",
    ]

    return any(
        signal in text
        for signal in quota_signals
    )


# ================================================================
# Print results
# ================================================================

def print_results(results_df):

    successful = results_df[
        results_df["status"] == "success"
    ].copy()

    if successful.empty:

        print()
        print("=" * 70)
        print("NO SUCCESSFUL PREDICTIONS")
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

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Main metrics
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL GEMINI INTENT EVALUATION")
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

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

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
    ].copy()

    print()
    print("=" * 70)
    print(
        f"INCORRECT PREDICTIONS ({len(wrong)})"
    )
    print("=" * 70)

    if wrong.empty:

        print(
            "No incorrect predictions."
        )

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

    # ------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------

    errors = results_df[
        results_df["status"] == "error"
    ].copy()

    if not errors.empty:

        print()
        print("=" * 70)
        print(
            f"UNFINISHED / ERROR ROWS ({len(errors)})"
        )
        print("=" * 70)

        for _, row in errors.iterrows():

            print(
                f"Row {row['row_id']}: "
                f"{row['error']}"
            )


# ================================================================
# Main
# ================================================================

def main():

    print("=" * 70)
    print("GEMINI INTENT CLASSIFIER EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    df = load_development_data()

    print()
    print(
        f"Development examples : {len(df)}"
    )

    print(
        f"Prediction file      : {OUTPUT_FILE}"
    )

    print(
        f"Request delay        : "
        f"{REQUEST_DELAY_SECONDS} seconds"
    )

    # ------------------------------------------------------------
    # Load previous predictions
    # ------------------------------------------------------------

    previous = (
        load_previous_predictions()
    )

    if not previous.empty:

        print()
        print(
            f"Previous predictions found: "
            f"{len(previous)}"
        )

    # ------------------------------------------------------------
    # Prepare dataframe
    # ------------------------------------------------------------

    result_df = prepare_result_dataframe(
        df,
        previous,
    )

    # ------------------------------------------------------------
    # Determine pending rows.
    #
    # IMPORTANT:
    # Only SUCCESS rows count as completed.
    # Errors are retried on the next run.
    # ------------------------------------------------------------

    pending_mask = (
        result_df["status"] != "success"
    )

    pending_indices = list(
        result_df.index[pending_mask]
    )

    pending_count = len(
        pending_indices
    )

    completed_count = (
        len(result_df)
        - pending_count
    )

    print()
    print(
        f"Already successful : "
        f"{completed_count}"
    )

    print(
        f"Remaining to run   : "
        f"{pending_count}"
    )

    # ------------------------------------------------------------
    # Nothing to process
    # ------------------------------------------------------------

    if pending_count == 0:

        print()
        print(
            "All examples have successful predictions."
        )

        print_results(
            result_df
        )

        return

    # ------------------------------------------------------------
    # Initialize Gemini
    # ------------------------------------------------------------

    print()
    print(
        "Initializing Gemini classifier..."
    )

    classifier = (
        GeminiIntentClassifier()
    )

    print(
        "Gemini classifier ready."
    )

    # ------------------------------------------------------------
    # Process pending examples
    # ------------------------------------------------------------

    processed_this_run = 0

    for index in pending_indices:

        row = result_df.loc[index]

        row_id = row["row_id"]

        message = (
            str(row["text"])
            .strip()
        )

        true_intent = (
            str(row["intent"])
            .strip()
        )

        current_number = (
            completed_count
            + processed_this_run
            + 1
        )

        print()
        print("-" * 70)

        print(
            f"Processing "
            f"{current_number}/"
            f"{len(result_df)}"
        )

        print(
            f"Row ID      : {row_id}"
        )

        print(
            f"True intent : "
            f"{true_intent}"
        )

        print(
            f"Message     : "
            f"{message}"
        )

        # --------------------------------------------------------
        # Gemini request
        # --------------------------------------------------------

        try:

            result = (
                classifier.classify(
                    message
                )
            )

            predicted_intent = (
                result["intent"]
            )

            confidence = float(
                result["confidence"]
            )

            reason = (
                result["reason"]
            )

            # ----------------------------------------------------
            # Store successful prediction
            # ----------------------------------------------------

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

            # ----------------------------------------------------
            # Show result
            # ----------------------------------------------------

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
                f"Reason      : "
                f"{reason}"
            )

            print(
                f"Result      : "
                f"{prediction_status}"
            )

        # --------------------------------------------------------
        # Handle errors
        # --------------------------------------------------------

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

            # ----------------------------------------------------
            # Save immediately before stopping on quota.
            # ----------------------------------------------------

            save_predictions(
                result_df
            )

            if is_quota_error(
                error_message
            ):

                print()
                print("=" * 70)
                print("GEMINI QUOTA / RATE LIMIT REACHED")
                print("=" * 70)

                print(
                    "The run is stopping now."
                )

                print(
                    "Successful predictions have been saved."
                )

                print(
                    f"Saved to: {OUTPUT_FILE}"
                )

                print()
                print(
                    "Run the same command again after "
                    "the quota resets."
                )

                return

        # --------------------------------------------------------
        # Save after EVERY example
        # --------------------------------------------------------

        save_predictions(
            result_df
        )

        processed_this_run += 1

        # --------------------------------------------------------
        # Wait before next Gemini request
        # --------------------------------------------------------

        if (
            processed_this_run
            < pending_count
        ):

            print(
                f"Waiting "
                f"{REQUEST_DELAY_SECONDS} "
                f"seconds..."
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    # ------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------

    save_predictions(
        result_df
    )

    # ------------------------------------------------------------
    # Final evaluation
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


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    main()