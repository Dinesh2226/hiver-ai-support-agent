from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

reply_file = BASE / "data" / "golden" / "reply_evaluation.csv"
judge_file = BASE / "data" / "golden" / "reply_judgments.csv"
out_file = BASE / "data" / "golden" / "human_review.csv"

replies = pd.read_csv(reply_file)
judgments = pd.read_csv(judge_file)

# Merge the automated judge's results so we can compare them later.
df = replies.merge(
    judgments[
        [
            "row_id",
            "overall_score",
            "critical_error",
        ]
    ],
    on="row_id",
    how="left",
)

# Fixed, stratified sample of 12 examples covering:
# - simple/ambiguous messages
# - multiple intents
# - good and poor generated replies
# - different escalation situations
preferred_ids = [
    2,
    15,
    18,
    22,
    25,
    33,
    78,
    89,
    123,
    126,
    140,
    196,
]

review = df[df["row_id"].isin(preferred_ids)].copy()

review = review[
    [
        "row_id",
        "customer_text",
        "intent",
        "predicted_intent",
        "predicted_escalation",
        "top_similarity",
        "generator_grounding_confidence",
        "generated_reply",
        "evidence_used",
        "overall_score",
        "critical_error",
    ]
]

# Human-entered fields.
review["human_groundedness"] = ""
review["human_correctness_safety"] = ""
review["human_helpfulness"] = ""
review["human_tone"] = ""
review["human_evidence_adherence"] = ""
review["human_overall_score"] = ""
review["human_critical_error"] = ""
review["human_reason"] = ""

review.to_csv(out_file, index=False)

print("=" * 70)
print("HUMAN REVIEW SHEET CREATED")
print("=" * 70)
print(f"Rows: {len(review)}")
print(f"Saved: {out_file}")
print()
print("Human rating fields:")
print("  groundedness            -> 1-5")
print("  correctness_safety      -> 1-5")
print("  helpfulness             -> 1-5")
print("  tone                    -> 1-5")
print("  evidence_adherence      -> 1-5")
print("  overall_score           -> 1-5")
print("  critical_error          -> True/False")
print("  reason                  -> short explanation")
print()
print("IMPORTANT:")
print("Do not use the LLM judge's overall_score or critical_error")
print("while performing your human rating.")