import os
import sys

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.abspath("."))

from src.intent.ollama_classifier import OllamaIntentClassifier
from src.retrieval.historical_retriever import HistoricalReplyRetriever
from src.escalation.escalation_engine import decide_escalation


GOLDEN_FILE = "data/golden/golden_set.csv"
OUTPUT_FILE = "data/golden/golden_predictions.csv"


def normalize_bool(value):
    text = str(value).strip().lower()

    if text in {"yes", "true", "1", "y"}:
        return True

    if text in {"no", "false", "0", "n"}:
        return False

    return None


def load_golden():
    if not os.path.exists(GOLDEN_FILE):
        raise FileNotFoundError(
            f"Golden set not found:\n{GOLDEN_FILE}"
        )

    df = pd.read_csv(GOLDEN_FILE)

    if "customer_text" not in df.columns:
        if "text" in df.columns:
            df = df.rename(
                columns={"text": "customer_text"}
            )
        else:
            raise ValueError(
                "Golden set must contain customer_text or text."
            )

    required = {"customer_text", "intent"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=["customer_text", "intent"]
    ).copy()

    df["customer_text"] = (
        df["customer_text"]
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["customer_text"] != ""
    ].copy()

    if "row_id" not in df.columns:
        df.insert(
            0,
            "row_id",
            range(len(df)),
        )

    return df.reset_index(drop=True)


def save(df):
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def main():

    print("=" * 70)
    print("FAST GOLDEN SET EVALUATION")
    print("=" * 70)

    df = load_golden()

    print(
        f"\nGolden examples: {len(df)}"
    )

    # ------------------------------------------------------------
    # GOLDEN TEXT EXCLUSIONS
    # ------------------------------------------------------------
    # Prevent the golden/test examples from being retrieved as
    # historical evidence during evaluation.
    golden_texts = set(
        df["customer_text"]
        .astype(str)
        .str.strip()
    )

    print(
        f"Golden messages excluded from retrieval: "
        f"{len(golden_texts)}"
    )

    # ------------------------------------------------------------
    # Resume previous run
    # ------------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        previous = pd.read_csv(
            OUTPUT_FILE
        )

        if "row_id" in previous.columns:

            print(
                f"Previous predictions found: "
                f"{len(previous)}"
            )

            keep = [
                "row_id",
                "predicted_intent",
                "intent_confidence",
                "predicted_escalation",
                "escalation_reason_pred",
                "top_similarity",
                "status",
                "error",
            ]

            keep = [
                c for c in keep
                if c in previous.columns
            ]

            previous = previous[keep]

            df = df.merge(
                previous,
                on="row_id",
                how="left",
            )

    # ------------------------------------------------------------
    # Initialize columns
    # ------------------------------------------------------------

    defaults = {
        "predicted_intent": None,
        "intent_confidence": float("nan"),
        "predicted_escalation": None,
        "escalation_reason_pred": None,
        "top_similarity": float("nan"),
        "status": "pending",
        "error": None,
    }

    for column, value in defaults.items():

        if column not in df.columns:
            df[column] = value

    # ------------------------------------------------------------
    # Gold escalation
    # ------------------------------------------------------------

    if "escalation" in df.columns:

        df["gold_escalation_bool"] = (
            df["escalation"]
            .apply(normalize_bool)
        )

    else:

        df["gold_escalation_bool"] = None

    # ------------------------------------------------------------
    # Initialize components
    # ------------------------------------------------------------

    print("\nLoading Ollama classifier...")

    classifier = OllamaIntentClassifier(
        model="qwen3:8b"
    )

    print("Loading historical retriever...")

    retriever = HistoricalReplyRetriever(
        "data/processed/amazonhelp_pairs.csv"
    )

    print("Fast evaluation components ready.")

    # ------------------------------------------------------------
    # Pending rows
    # ------------------------------------------------------------

    pending = list(
        df.index[
            df["status"] != "success"
        ]
    )

    print(
        f"\nAlready completed: "
        f"{len(df) - len(pending)}"
    )

    print(
        f"Remaining: "
        f"{len(pending)}"
    )

    # ------------------------------------------------------------
    # Process
    # ------------------------------------------------------------

    for position, index in enumerate(
        pending,
        start=1,
    ):

        row = df.loc[index]

        message = str(
            row["customer_text"]
        ).strip()

        print()
        print("-" * 70)
        print(
            f"Processing {position}/{len(pending)}"
        )
        print(
            f"Row ID: {row['row_id']}"
        )
        print(
            f"Message: {message}"
        )

        try:

            # ----------------------------------------------------
            # 1. Intent
            # ----------------------------------------------------

            intent_result = classifier.classify(
                message
            )

            predicted_intent = (
                intent_result["intent"]
            )

            confidence = float(
                intent_result["confidence"]
            )

            # ----------------------------------------------------
            # 2. Retrieval
            # ----------------------------------------------------
            # IMPORTANT:
            # Exclude every golden customer message from the
            # historical retrieval candidate pool.
            # This prevents exact test leakage.
            # ----------------------------------------------------

            historical_examples = (
                retriever.retrieve(
                    message,
                    top_k=5,
                    exclude_texts=golden_texts,
                )
            )

            if historical_examples:
                top_similarity = float(
                    historical_examples[0].get(
                        "similarity",
                        0.0,
                    )
                )
            else:
                top_similarity = 0.0

            # ----------------------------------------------------
            # 3. Escalation
            # ----------------------------------------------------

            escalation_result = decide_escalation(
                customer_message=message,
                intent=predicted_intent,
                intent_confidence=confidence,
                retrieval_results=historical_examples,
            )

            # ----------------------------------------------------
            # Save
            # ----------------------------------------------------

            df.at[
                index,
                "predicted_intent",
            ] = predicted_intent

            df.at[
                index,
                "intent_confidence",
            ] = confidence

            df.at[
                index,
                "predicted_escalation",
            ] = escalation_result[
                "escalate"
            ]

            df.at[
                index,
                "escalation_reason_pred",
            ] = escalation_result[
                "reason"
            ]

            df.at[
                index,
                "top_similarity",
            ] = top_similarity

            df.at[
                index,
                "status",
            ] = "success"

            df.at[
                index,
                "error",
            ] = None

            intent_status = (
                "CORRECT"
                if predicted_intent
                == row["intent"]
                else "WRONG"
            )

            print(
                f"Predicted intent: "
                f"{predicted_intent}"
            )

            print(
                f"Gold intent: "
                f"{row['intent']}"
            )

            print(
                f"Intent result: "
                f"{intent_status}"
            )

            print(
                f"Escalation: "
                f"{escalation_result['escalate']}"
            )

            print(
                f"Top similarity: "
                f"{top_similarity:.3f}"
            )

        except Exception as exc:

            error = str(exc)

            df.at[
                index,
                "status",
            ] = "error"

            df.at[
                index,
                "error",
            ] = error

            print(
                f"ERROR: {error}"
            )

        # Save after EVERY row.
        save(df)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    successful = df[
        df["status"] == "success"
    ].copy()

    print()
    print("=" * 70)
    print("INTENT RESULTS")
    print("=" * 70)

    if successful.empty:
        print("No successful examples.")
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

    print(
        f"Examples evaluated: "
        f"{len(successful)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy_score(y_true, y_pred):.4f}"
    )

    print(
        f"Macro Precision: "
        f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )

    print(
        f"Macro Recall: "
        f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )

    print(
        f"Macro F1: "
        f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    )

    print(
        f"Weighted F1: "
        f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}"
    )

    print()

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

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        ).to_string()
    )

    # ------------------------------------------------------------
    # Retrieval statistics
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    retrieval_scores = pd.to_numeric(
        successful["top_similarity"],
        errors="coerce",
    ).dropna()

    if not retrieval_scores.empty:

        print(
            f"Examples with retrieval results: "
            f"{len(retrieval_scores)}"
        )

        print(
            f"Mean top similarity: "
            f"{retrieval_scores.mean():.4f}"
        )

        print(
            f"Median top similarity: "
            f"{retrieval_scores.median():.4f}"
        )

        print(
            f"Minimum top similarity: "
            f"{retrieval_scores.min():.4f}"
        )

        print(
            f"Maximum top similarity: "
            f"{retrieval_scores.max():.4f}"
        )

        print(
            f"Exact 1.000 matches remaining: "
            f"{(retrieval_scores == 1.0).sum()}"
        )

        print(
            f"Similarity >= 0.50: "
            f"{(retrieval_scores >= 0.50).sum()}"
        )

        print(
            f"Similarity >= 0.70: "
            f"{(retrieval_scores >= 0.70).sum()}"
        )

        print(
            f"Similarity < 0.20: "
            f"{(retrieval_scores < 0.20).sum()}"
        )

    else:

        print("No retrieval scores available.")

    # ------------------------------------------------------------
    # Escalation metrics
    # ------------------------------------------------------------

    esc = successful[
        successful["gold_escalation_bool"].notna()
    ].copy()

    if not esc.empty:

        y_true_esc = esc[
            "gold_escalation_bool"
        ].astype(bool)

        y_pred_esc = (
            esc["predicted_escalation"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
        )

        valid = y_pred_esc.notna()

        y_true_esc = y_true_esc[valid]
        y_pred_esc = y_pred_esc[valid]

        print()
        print("=" * 70)
        print("ESCALATION RESULTS")
        print("=" * 70)

        print(
            f"Examples evaluated: "
            f"{len(y_true_esc)}"
        )

        print(
            f"Accuracy: "
            f"{accuracy_score(y_true_esc, y_pred_esc):.4f}"
        )

        print(
            f"Precision: "
            f"{precision_score(y_true_esc, y_pred_esc, zero_division=0):.4f}"
        )

        print(
            f"Recall: "
            f"{recall_score(y_true_esc, y_pred_esc, zero_division=0):.4f}"
        )

        print(
            f"F1: "
            f"{f1_score(y_true_esc, y_pred_esc, zero_division=0):.4f}"
        )

    print()
    print("=" * 70)
    print("FAST GOLDEN EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()