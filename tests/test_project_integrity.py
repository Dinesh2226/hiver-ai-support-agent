from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE / "data" / "processed" / "train_annotations.csv"
GOLDEN_FILE = BASE / "data" / "golden" / "golden_set.csv"
GOLDEN_PRED_FILE = BASE / "data" / "golden" / "golden_predictions.csv"
REPLY_EVAL_FILE = BASE / "data" / "golden" / "reply_evaluation.csv"
JUDGE_FILE = BASE / "data" / "golden" / "reply_judgments.csv"
BASELINE_FILE = BASE / "data" / "golden" / "baseline_results.txt"
FAILURE_FILE = BASE / "data" / "golden" / "failure_analysis.md"
DECISION_FILE = BASE / "data" / "golden" / "decision_log.md"


VALID_INTENTS = {
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
}


def test_required_artifacts_exist():
    required = [
        TRAIN_FILE,
        GOLDEN_FILE,
        GOLDEN_PRED_FILE,
        REPLY_EVAL_FILE,
        JUDGE_FILE,
        BASELINE_FILE,
        FAILURE_FILE,
        DECISION_FILE,
    ]

    missing = [str(path) for path in required if not path.exists()]

    assert not missing, f"Missing required artifacts: {missing}"


def test_training_and_golden_sizes():
    train = pd.read_csv(TRAIN_FILE)
    golden = pd.read_csv(GOLDEN_FILE)

    assert len(train) == 700
    assert len(golden) == 200


def test_golden_contains_all_intents():
    golden = pd.read_csv(GOLDEN_FILE)

    assert "intent" in golden.columns

    intents = set(golden["intent"].dropna().astype(str))

    assert intents == VALID_INTENTS


def test_golden_predictions_complete():
    predictions = pd.read_csv(GOLDEN_PRED_FILE)

    assert len(predictions) == 200
    assert "row_id" in predictions.columns
    assert "predicted_intent" in predictions.columns

    assert predictions["row_id"].nunique() == 200

    predicted = set(
        predictions["predicted_intent"]
        .dropna()
        .astype(str)
    )

    assert predicted.issubset(VALID_INTENTS)


def test_reply_evaluation_complete():
    replies = pd.read_csv(REPLY_EVAL_FILE)

    assert len(replies) == 30
    assert "generated_reply" in replies.columns

    successful = replies[
        replies["generation_status"].astype(str).str.lower() == "success"
    ]

    assert len(successful) == 30


def test_reply_judgments_complete():
    judgments = pd.read_csv(JUDGE_FILE)

    assert len(judgments) == 30
    assert "overall_score" in judgments.columns
    assert "critical_error" in judgments.columns

    scores = pd.to_numeric(
        judgments["overall_score"],
        errors="coerce",
    )

    assert scores.notna().all()
    assert scores.between(1, 5).all()


def test_baseline_results_contain_expected_models():
    text = BASELINE_FILE.read_text(encoding="utf-8")

    assert "TRIVIAL BASELINE" in text
    assert "SIMPLE BASELINE" in text
    assert "Accuracy: 0.3700" in text
    assert "Accuracy: 0.5200" in text


def test_failure_analysis_contains_required_sections():
    text = FAILURE_FILE.read_text(encoding="utf-8")

    assert "# Failure Analysis" in text
    assert "## Top 5 Failure Modes" in text
    assert "## What is misleading about my headline number?" in text
    assert "## Next-Week Plan" in text


def test_decision_log_contains_15_decisions():
    text = DECISION_FILE.read_text(encoding="utf-8")

    for number in range(1, 16):
        assert f"| {number} |" in text