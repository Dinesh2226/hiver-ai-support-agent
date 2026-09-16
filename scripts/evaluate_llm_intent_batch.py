import json
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

from dotenv import load_dotenv
from google import genai


# ================================================================
# Configuration
# ================================================================

DEV_FILE = "data/processed/dev_annotations.csv"
OUTPUT_FILE = "data/processed/llm_intent_batch_predictions.csv"

MODEL_NAME = "gemini-3.6-flash"

# 10 examples per Gemini request
BATCH_SIZE = 10

# Delay between batches
REQUEST_DELAY_SECONDS = 5.0


INTENTS = [
    "ORDER_DELIVERY",
    "RETURN_REFUND",
    "ACCOUNT_ACCESS",
    "PAYMENT_BILLING",
    "TECHNICAL_SUPPORT",
    "PRODUCT_CONTENT",
    "PRICING_PROMOTIONS",
    "PRIME_MEMBERSHIP",
    "COMPLAINT_FEEDBACK",
    "OTHER",
]


INTENT_DEFINITIONS = """
ORDER_DELIVERY:
Orders, shipping, tracking, delivery dates, delays, missing packages,
or delivered-but-not-received packages.

RETURN_REFUND:
Returns, refunds, damaged/wrong items, replacements, or return pickup.

ACCOUNT_ACCESS:
Login, password, locked account, or account access problems.

PAYMENT_BILLING:
Payment failures, charges, billing, unexpected charges, or payment-account issues.

TECHNICAL_SUPPORT:
Alexa, Echo, Fire TV, Prime Video, apps, website errors, checkout problems,
or technical/device problems.

PRODUCT_CONTENT:
Product availability, catalog/content availability, release information,
or product information.

PRICING_PROMOTIONS:
Prices, discounts, promotions, offers, coupons, deals, or cashback clearly
related to an offer.

PRIME_MEMBERSHIP:
Amazon Prime membership, trial, subscription, renewal, cancellation,
membership fees, or Prime benefits.

COMPLAINT_FEEDBACK:
General complaints or service criticism when there is no more specific
operational support issue.

OTHER:
Praise, thanks, vague requests, insufficient context, or anything that
does not fit the taxonomy.
"""


# ================================================================
# Possible column names
# ================================================================

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


# ================================================================
# Find column
# ================================================================

def find_column(df, candidates, column_type):

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lowercase_map = {
        str(col).lower().strip(): col
        for col in df.columns
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


# ================================================================
# Load previous batch predictions
# ================================================================

def load_previous_predictions():

    if not os.path.exists(OUTPUT_FILE):
        return pd.DataFrame()

    try:
        previous = pd.read_csv(
            OUTPUT_FILE
        )

        if "row_id" not in previous.columns:
            return pd.DataFrame()

        return previous

    except Exception as exc:

        print(
            f"Warning: could not load previous predictions: {exc}"
        )

        return pd.DataFrame()


# ================================================================
# Prepare result dataframe
# ================================================================

def prepare_result_dataframe(
    df,
    previous,
):

    result_df = df.copy()

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
    # Restore previous results
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
            col
            for col in previous_columns
            if col in previous.columns
        ]

        previous_clean = previous[
            available_columns
        ].copy()

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

    # Make sure columns exist
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

    # Fix dtypes after reading CSV
    result_df["confidence"] = pd.to_numeric(
        result_df["confidence"],
        errors="coerce",
    )

    result_df["status"] = (
        result_df["status"]
        .fillna("pending")
        .astype(str)
    )

    result_df["predicted_intent"] = (
        result_df["predicted_intent"]
        .astype("object")
    )

    result_df["reason"] = (
        result_df["reason"]
        .astype("object")
    )

    result_df["error"] = (
        result_df["error"]
        .astype("object")
    )

    return result_df


# ================================================================
# Save
# ================================================================

def save_predictions(df):

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ================================================================
# Parse Gemini JSON
# ================================================================

def parse_json_response(raw):

    raw = raw.strip()

    # Remove Markdown fences
    if raw.startswith("```"):

        lines = raw.splitlines()

        if (
            lines
            and lines[0].strip().startswith("```")
        ):
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        raw = "\n".join(lines).strip()

    # Direct JSON
    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        pass

    # Extract JSON array
    start = raw.find("[")
    end = raw.rfind("]")

    if start != -1 and end != -1 and end > start:

        json_text = raw[
            start:end + 1
        ]

        try:
            return json.loads(
                json_text
            )
        except json.JSONDecodeError:
            pass

    # Extract JSON object
    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end != -1 and end > start:

        json_text = raw[
            start:end + 1
        ]

        try:
            return json.loads(
                json_text
            )
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Gemini returned invalid JSON.\n"
        f"Raw response:\n{raw}"
    )


# ================================================================
# Validate batch response
# ================================================================

def validate_batch_response(
    parsed,
    expected_count,
):

    if not isinstance(parsed, list):

        # Sometimes model may return:
        # {"results": [...]}
        if isinstance(parsed, dict):

            if isinstance(
                parsed.get("results"),
                list,
            ):
                parsed = parsed["results"]

            else:
                raise ValueError(
                    "Gemini returned an object instead of "
                    "a result list."
                )

        else:

            raise ValueError(
                "Gemini batch response is not a list."
            )

    if len(parsed) != expected_count:

        raise ValueError(
            f"Gemini returned {len(parsed)} results, "
            f"but expected {expected_count}."
        )

    clean_results = []

    for item in parsed:

        if not isinstance(item, dict):

            raise ValueError(
                "Each batch result must be a JSON object."
            )

        intent = item.get(
            "intent"
        )

        if intent not in INTENTS:

            raise ValueError(
                f"Invalid intent returned: {intent}"
            )

        try:
            confidence = float(
                item.get(
                    "confidence",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        reason = item.get(
            "reason",
            "",
        )

        if reason is None:
            reason = ""

        clean_results.append(
            {
                "intent": intent,
                "confidence": confidence,
                "reason": str(reason).strip(),
            }
        )

    return clean_results


# ================================================================
# Gemini batch classification
# ================================================================

def classify_batch(
    client,
    batch_df,
):

    examples_text = []

    for number, (_, row) in enumerate(
        batch_df.iterrows(),
        start=1,
    ):

        message = str(
            row["text"]
        ).strip()

        examples_text.append(
            f"""
EXAMPLE {number}
CUSTOMER MESSAGE:
{message}
"""
        )

    examples_block = "\n".join(
        examples_text
    )

    prompt = f"""
You are an Amazon customer-support intent classifier.

Classify EACH customer message into exactly ONE of these intents:

{", ".join(INTENTS)}

INTENT DEFINITIONS:

{INTENT_DEFINITIONS}

Rules:

1. Classify the customer's PRIMARY support problem.
2. Ignore emotional tone when a specific operational problem exists.
3. Do not infer unsupported information.
4. Use OTHER when the message is vague, praise/thanks, or lacks enough
   information to fit another intent.
5. Delivery/order/tracking/delayed/missing-package issues are
   ORDER_DELIVERY.
6. Returns/refunds/damaged/wrong-item/replacement issues are
   RETURN_REFUND.
7. Login/password/account-access problems are ACCOUNT_ACCESS.
8. Charges/payment/billing problems are PAYMENT_BILLING.
9. Alexa, Echo, Fire TV, Prime Video, app, website, or device problems
   are TECHNICAL_SUPPORT.
10. Product/content availability and product information are
    PRODUCT_CONTENT.
11. Prices, discounts, deals, coupons, promotions, and promotional
    cashback are PRICING_PROMOTIONS.
12. Amazon Prime membership/trial/subscription/renewal/fees/benefits
    are PRIME_MEMBERSHIP.
13. General customer-service criticism without a more specific
    operational issue is COMPLAINT_FEEDBACK.

Return EXACTLY one JSON array with one object for each example,
in the SAME ORDER as the examples.

DO NOT return Markdown.
DO NOT return ```json.
DO NOT add commentary.

Each object must have exactly:

{{
  "intent": "ONE_INTENT",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

CUSTOMER EXAMPLES:

{examples_block}
"""

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=prompt,
    )

    raw = interaction.output_text

    if not raw:
        raise ValueError(
            "Gemini returned empty output."
        )

    parsed = parse_json_response(
        raw
    )

    return validate_batch_response(
        parsed,
        len(batch_df),
    )


# ================================================================
# Quota detection
# ================================================================

def is_quota_error(error_message):

    text = error_message.lower()

    signals = [
        "429",
        "too_many_requests",
        "quota exceeded",
        "rate limit",
        "resource_exhausted",
    ]

    return any(
        signal in text
        for signal in signals
    )


# ================================================================
# Final metrics
# ================================================================

def print_results(
    results_df,
):

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
    print("FINAL GEMINI BATCH INTENT EVALUATION")
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


# ================================================================
# Main
# ================================================================

def main():

    print("=" * 70)
    print("GEMINI BATCH INTENT EVALUATION")
    print("=" * 70)

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY was not found in .env"
        )

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    df = load_development_data()

    print()
    print(
        f"Development examples : {len(df)}"
    )

    print(
        f"Batch size           : {BATCH_SIZE}"
    )

    print(
        f"Expected API calls   : "
        f"{(len(df) + BATCH_SIZE - 1) // BATCH_SIZE}"
    )

    print(
        f"Output file          : {OUTPUT_FILE}"
    )

    # ------------------------------------------------------------
    # Load previous results
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

    result_df = prepare_result_dataframe(
        df,
        previous,
    )

    # ------------------------------------------------------------
    # Only rows without successful predictions need processing
    # ------------------------------------------------------------

    pending_indices = list(
        result_df.index[
            result_df["status"] != "success"
        ]
    )

    print()
    print(
        f"Already successful : "
        f"{len(result_df) - len(pending_indices)}"
    )

    print(
        f"Remaining examples : "
        f"{len(pending_indices)}"
    )

    if not pending_indices:

        print()
        print(
            "All examples already evaluated."
        )

        print_results(
            result_df
        )

        return

    # ------------------------------------------------------------
    # Gemini client
    # ------------------------------------------------------------

    client = genai.Client(
        api_key=api_key
    )

    # ------------------------------------------------------------
    # Create batches
    # ------------------------------------------------------------

    batches = []

    for start in range(
        0,
        len(pending_indices),
        BATCH_SIZE,
    ):

        batch_indices = (
            pending_indices[
                start:start + BATCH_SIZE
            ]
        )

        batches.append(
            batch_indices
        )

    print()
    print(
        f"Batches remaining: {len(batches)}"
    )

    # ------------------------------------------------------------
    # Process batches
    # ------------------------------------------------------------

    for batch_number, batch_indices in enumerate(
        batches,
        start=1,
    ):

        batch_df = result_df.loc[
            batch_indices
        ].copy()

        print()
        print("=" * 70)
        print(
            f"BATCH {batch_number}/"
            f"{len(batches)}"
        )
        print("=" * 70)

        print(
            f"Rows: "
            f"{batch_indices[0]} - "
            f"{batch_indices[-1]}"
        )

        try:

            predictions = classify_batch(
                client,
                batch_df,
            )

            # ----------------------------------------------------
            # Store predictions
            # ----------------------------------------------------

            for index, prediction in zip(
                batch_indices,
                predictions,
            ):

                result_df.at[
                    index,
                    "predicted_intent",
                ] = prediction[
                    "intent"
                ]

                result_df.at[
                    index,
                    "confidence",
                ] = float(
                    prediction[
                        "confidence"
                    ]
                )

                result_df.at[
                    index,
                    "reason",
                ] = prediction[
                    "reason"
                ]

                result_df.at[
                    index,
                    "status",
                ] = "success"

                result_df.at[
                    index,
                    "error",
                ] = None

                true_intent = str(
                    result_df.at[
                        index,
                        "intent"
                    ]
                ).strip()

                predicted_intent = (
                    prediction["intent"]
                )

                status = (
                    "CORRECT"
                    if true_intent
                    == predicted_intent
                    else "WRONG"
                )

                print(
                    f"Row {result_df.at[index, 'row_id']}: "
                    f"{predicted_intent} "
                    f"({status})"
                )

            # ----------------------------------------------------
            # Save after every successful batch
            # ----------------------------------------------------

            save_predictions(
                result_df
            )

            print()
            print(
                f"Batch {batch_number} saved."
            )

        except Exception as exc:

            error_message = str(
                exc
            )

            print()
            print(
                f"BATCH ERROR: "
                f"{error_message}"
            )

            # Mark this batch as error
            for index in batch_indices:

                result_df.at[
                    index,
                    "status",
                ] = "error"

                result_df.at[
                    index,
                    "error",
                ] = error_message

            save_predictions(
                result_df
            )

            # ----------------------------------------------------
            # Stop on quota
            # ----------------------------------------------------

            if is_quota_error(
                error_message
            ):

                print()
                print("=" * 70)
                print("GEMINI QUOTA REACHED")
                print("=" * 70)

                print(
                    "Stopping the evaluation."
                )

                print(
                    f"Progress saved to:\n"
                    f"{OUTPUT_FILE}"
                )

                return

            # Non-quota error:
            # stop rather than corrupt evaluation alignment.
            print()
            print(
                "Stopping because the batch response "
                "could not be validated."
            )

            return

        # --------------------------------------------------------
        # Delay before next API request
        # --------------------------------------------------------

        if batch_number < len(batches):

            print()
            print(
                f"Waiting "
                f"{REQUEST_DELAY_SECONDS} seconds "
                f"before next batch..."
            )

            time.sleep(
                REQUEST_DELAY_SECONDS
            )

    # ------------------------------------------------------------
    # Final save and metrics
    # ------------------------------------------------------------

    save_predictions(
        result_df
    )

    print_results(
        result_df
    )

    print()
    print("=" * 70)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Predictions saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()