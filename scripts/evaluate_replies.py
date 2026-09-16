import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from src.retrieval.historical_retriever import HistoricalReplyRetriever
from src.generation.ollama_reply_generator import OllamaReplyGenerator


GOLDEN_FILE = "data/golden/golden_set.csv"
PREDICTIONS_FILE = "data/golden/golden_predictions.csv"

OUTPUT_FILE = "data/golden/reply_evaluation.csv"
SAMPLE_FILE = "data/golden/reply_evaluation_sample.csv"

REPLY_SAMPLE_SIZE = 30
RANDOM_SEED = 42


def normalize_bool(value):
    text = str(value).strip().lower()

    if text in {"true", "yes", "1", "y"}:
        return True

    if text in {"false", "no", "0", "n"}:
        return False

    return None


def load_data():
    if not os.path.exists(GOLDEN_FILE):
        raise FileNotFoundError(
            f"Golden file not found: {GOLDEN_FILE}"
        )

    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(
            f"Golden predictions not found: {PREDICTIONS_FILE}"
        )

    golden = pd.read_csv(GOLDEN_FILE)
    predictions = pd.read_csv(PREDICTIONS_FILE)

    if "customer_text" not in golden.columns:

        if "text" in golden.columns:
            golden = golden.rename(
                columns={"text": "customer_text"}
            )
        else:
            raise ValueError(
                "Golden set must contain customer_text or text."
            )

    # Create the same deterministic row IDs used by
    # evaluate_golden.py when the golden file itself has no row_id.
    if "row_id" not in golden.columns:
        golden.insert(
            0,
            "row_id",
            range(len(golden)),
        )

    required_prediction_columns = {
        "row_id",
        "predicted_intent",
        "predicted_escalation",
        "top_similarity",
        "status",
    }

    missing = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing:
        raise ValueError(
            "golden_predictions.csv is missing columns: "
            f"{sorted(missing)}"
        )

    golden["row_id"] = pd.to_numeric(
        golden["row_id"],
        errors="coerce",
    )

    predictions["row_id"] = pd.to_numeric(
        predictions["row_id"],
        errors="coerce",
    )

    golden["customer_text"] = (
        golden["customer_text"]
        .astype(str)
        .str.strip()
    )

    if "intent" in golden.columns:
        golden["intent"] = (
            golden["intent"]
            .astype(str)
            .str.strip()
        )

    return golden, predictions


def select_stratified_sample(
    golden,
    predictions,
    sample_size=30,
):
    prediction_columns = [
        "row_id",
        "predicted_intent",
        "predicted_escalation",
        "top_similarity",
        "status",
    ]

    prediction_subset = predictions[
        prediction_columns
    ].copy()

    df = golden.merge(
        prediction_subset,
        on="row_id",
        how="inner",
    )

    df = df[
        df["status"].astype(str).str.lower() == "success"
    ].copy()

    if df.empty:
        raise ValueError(
            "No successful rows found in golden_predictions.csv."
        )

    if len(df) <= sample_size:
        return df.reset_index(drop=True)

    def retrieval_bucket(score):
        try:
            score = float(score)
        except Exception:
            score = 0.0

        if score < 0.20:
            return "low"
        elif score < 0.40:
            return "medium"
        else:
            return "high"

    df["retrieval_bucket"] = (
        df["top_similarity"]
        .fillna(0.0)
        .apply(retrieval_bucket)
    )

    df["escalation_bucket"] = (
        df["predicted_escalation"]
        .apply(normalize_bool)
        .map(
            {
                True: "escalate",
                False: "auto_handle",
            }
        )
        .fillna("unknown")
    )

    selected_indices = set()

    # ------------------------------------------------------------
    # 1. Cover every intent
    # ------------------------------------------------------------

    if "intent" in df.columns:

        intents = sorted(
            df["intent"]
            .dropna()
            .astype(str)
            .unique()
        )

        for i, intent in enumerate(intents):

            candidates = df[
                (df["intent"].astype(str) == intent)
                & (~df.index.isin(selected_indices))
            ]

            if candidates.empty:
                continue

            chosen = candidates.sample(
                n=1,
                random_state=RANDOM_SEED + i,
            )

            selected_indices.add(
                chosen.index[0]
            )

            if len(selected_indices) >= sample_size:
                break

    # ------------------------------------------------------------
    # 2. Cover retrieval quality
    # ------------------------------------------------------------

    for i, bucket in enumerate(
        ["low", "medium", "high"]
    ):

        if len(selected_indices) >= sample_size:
            break

        candidates = df[
            (df["retrieval_bucket"] == bucket)
            & (~df.index.isin(selected_indices))
        ]

        if candidates.empty:
            continue

        take = min(
            5,
            len(candidates),
            sample_size - len(selected_indices),
        )

        if take <= 0:
            continue

        chosen = candidates.sample(
            n=take,
            random_state=RANDOM_SEED + 100 + i,
        )

        selected_indices.update(
            chosen.index.tolist()
        )

    # ------------------------------------------------------------
    # 3. Cover escalation states
    # ------------------------------------------------------------

    for i, bucket in enumerate(
        ["escalate", "auto_handle"]
    ):

        if len(selected_indices) >= sample_size:
            break

        candidates = df[
            (df["escalation_bucket"] == bucket)
            & (~df.index.isin(selected_indices))
        ]

        if candidates.empty:
            continue

        take = min(
            4,
            len(candidates),
            sample_size - len(selected_indices),
        )

        if take <= 0:
            continue

        chosen = candidates.sample(
            n=take,
            random_state=RANDOM_SEED + 200 + i,
        )

        selected_indices.update(
            chosen.index.tolist()
        )

    # ------------------------------------------------------------
    # 4. Fill remaining slots
    # ------------------------------------------------------------

    if len(selected_indices) < sample_size:

        remaining = df[
            ~df.index.isin(selected_indices)
        ]

        take = min(
            sample_size - len(selected_indices),
            len(remaining),
        )

        if take > 0:

            chosen = remaining.sample(
                n=take,
                random_state=RANDOM_SEED,
            )

            selected_indices.update(
                chosen.index.tolist()
            )

    result = df.loc[
        sorted(selected_indices)
    ].copy()

    if len(result) > sample_size:

        result = result.sample(
            n=sample_size,
            random_state=RANDOM_SEED,
        )

    return result.reset_index(drop=True)


def extract_reply_text(result):
    """
    OllamaReplyGenerator.generate() returns a dictionary.
    Extract the generated reply robustly.
    """

    if isinstance(result, str):
        return result.strip()

    if not isinstance(result, dict):
        return str(result).strip()

    for key in [
        "reply",
        "draft_reply",
        "generated_reply",
        "text",
        "content",
    ]:
        value = result.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def extract_evidence_used(result):
    if not isinstance(result, dict):
        return ""

    for key in [
        "evidence_used",
        "evidence_indices",
        "sources",
    ]:
        if key in result:
            return str(result[key])

    return ""


def extract_grounding_confidence(result):
    if not isinstance(result, dict):
        return ""

    for key in [
        "grounding_confidence",
        "grounding_score",
        "confidence",
    ]:
        if key in result:
            return result[key]

    return ""


def main():

    print("=" * 70)
    print("REPLY QUALITY EVALUATION")
    print("=" * 70)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    golden, predictions = load_data()

    print(
        f"Golden examples: {len(golden)}"
    )

    # ------------------------------------------------------------
    # Select 30 examples
    # ------------------------------------------------------------

    sample = select_stratified_sample(
        golden=golden,
        predictions=predictions,
        sample_size=REPLY_SAMPLE_SIZE,
    )

    print(
        f"Selected reply-evaluation examples: "
        f"{len(sample)}"
    )

    sample.to_csv(
        SAMPLE_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # Exclude all golden texts from retrieval
    # ------------------------------------------------------------

    golden_texts = set(
        golden["customer_text"]
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        existing = pd.read_csv(
            OUTPUT_FILE
        )

        if "row_id" in existing.columns:

            existing_columns = [
                "row_id",
                "generated_reply",
                "evidence_used",
                "generator_grounding_confidence",
                "generation_status",
                "generation_error",
            ]

            existing_columns = [
                col
                for col in existing_columns
                if col in existing.columns
            ]

            existing = existing[
                existing_columns
            ].copy()

            sample = sample.merge(
                existing,
                on="row_id",
                how="left",
                suffixes=("", "_old"),
            )

            for column in [
                "generated_reply",
                "evidence_used",
                "generator_grounding_confidence",
                "generation_status",
                "generation_error",
            ]:

                old_column = f"{column}_old"

                if old_column in sample.columns:

                    if column not in sample.columns:
                        sample[column] = sample[old_column]
                    else:
                        sample[column] = sample[column].fillna(
                            sample[old_column]
                        )

                    sample.drop(
                        columns=[old_column],
                        inplace=True,
                    )

    # ------------------------------------------------------------
    # Output columns
    # ------------------------------------------------------------

    defaults = {
        "generated_reply": None,
        "evidence_used": None,
        "generator_grounding_confidence": None,
        "generation_status": "pending",
        "generation_error": None,
    }

    for column, default_value in defaults.items():

        if column not in sample.columns:
            sample[column] = default_value

    sample["generation_status"] = (
        sample["generation_status"]
        .fillna("pending")
    )

    # ------------------------------------------------------------
    # Initialize components
    # ------------------------------------------------------------

    print("\nLoading reply generator...")

    generator = OllamaReplyGenerator(
        model="qwen3:8b"
    )

    print(
        "Loading historical retriever..."
    )

    retriever = HistoricalReplyRetriever(
        "data/processed/amazonhelp_pairs.csv"
    )

    print(
        "Reply evaluation components ready."
    )

    # ------------------------------------------------------------
    # Pending
    # ------------------------------------------------------------

    pending = list(
        sample.index[
            sample["generation_status"] != "success"
        ]
    )

    print(
        f"\nAlready completed: "
        f"{len(sample) - len(pending)}"
    )

    print(
        f"Remaining: "
        f"{len(pending)}"
    )

    # ------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------

    for position, index in enumerate(
        pending,
        start=1,
    ):

        row = sample.loc[index]

        message = str(
            row["customer_text"]
        ).strip()

        intent = str(
            row["predicted_intent"]
        ).strip()

        print()
        print("-" * 70)

        print(
            f"Reply {position}/{len(pending)}"
        )

        print(
            f"Row ID: {row['row_id']}"
        )

        print(
            f"Intent: {intent}"
        )

        print(
            f"Retrieval similarity: "
            f"{float(row['top_similarity']):.3f}"
        )

        print(
            f"Message: {message}"
        )

        try:

            # ----------------------------------------------------
            # Retrieval
            # ----------------------------------------------------

            historical_examples = (
                retriever.retrieve(
                    message,
                    top_k=5,
                    exclude_texts=golden_texts,
                )
            )

            # ----------------------------------------------------
            # Generate
            # ----------------------------------------------------

            result = generator.generate(
                customer_message=message,
                intent=intent,
                historical_examples=historical_examples,
            )

            reply_text = extract_reply_text(
                result
            )

            evidence_used = (
                extract_evidence_used(
                    result
                )
            )

            generator_grounding_confidence = (
                extract_grounding_confidence(
                    result
                )
            )

            if not reply_text:
                raise ValueError(
                    "Reply generator returned an empty reply."
                )

            # ----------------------------------------------------
            # Save
            # ----------------------------------------------------

            sample.at[
                index,
                "generated_reply",
            ] = reply_text

            sample.at[
                index,
                "evidence_used",
            ] = evidence_used

            sample.at[
                index,
                "generator_grounding_confidence",
            ] = generator_grounding_confidence

            sample.at[
                index,
                "generation_status",
            ] = "success"

            sample.at[
                index,
                "generation_error",
            ] = None

            print(
                f"Reply: {reply_text}"
            )

            if evidence_used:
                print(
                    f"Evidence used: "
                    f"{evidence_used}"
                )

            if generator_grounding_confidence != "":
                print(
                    f"Generator grounding confidence: "
                    f"{generator_grounding_confidence}"
                )

        except Exception as exc:

            error = str(exc)

            sample.at[
                index,
                "generation_status",
            ] = "error"

            sample.at[
                index,
                "generation_error",
            ] = error

            print(
                f"ERROR: {error}"
            )

        # Save after every row.
        sample.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        time.sleep(0.5)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    successful = sample[
        sample["generation_status"] == "success"
    ].copy()

    failed = sample[
        sample["generation_status"] != "success"
    ].copy()

    print()
    print("=" * 70)
    print("REPLY GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Selected: {len(sample)}"
    )

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    if not successful.empty:

        print()
        print(
            successful[
                [
                    "row_id",
                    "intent",
                    "predicted_escalation",
                    "top_similarity",
                    "generated_reply",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print(
        f"\nSample saved to:\n{SAMPLE_FILE}"
    )


if __name__ == "__main__":
    main()