from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data" / "golden" / "decision_log.md"

decisions = [
    (
        1,
        "Brand selection",
        "Selected AmazonHelp because it has a large number of inbound/outbound support interactions in the Kaggle dataset.",
        "Provides enough historical support examples for retrieval and evaluation."
    ),
    (
        2,
        "Intent taxonomy",
        "Defined 10 operational support intents plus OTHER.",
        "Keeps the classifier small enough to build and evaluate within the assignment timeframe."
    ),
    (
        3,
        "Golden set",
        "Created a 200-example golden set and treated it as the held-out benchmark.",
        "Provides a fixed evaluation set separate from the training annotations."
    ),
    (
        4,
        "Training data",
        "Used 700 annotated examples for model development.",
        "Provides enough labeled data for a simple baseline while remaining practical to annotate."
    ),
    (
        5,
        "Trivial baseline",
        "Implemented a majority-class classifier.",
        "Establishes the minimum useful benchmark and exposes class imbalance."
    ),
    (
        6,
        "Simple baseline",
        "Implemented TF-IDF with Logistic Regression.",
        "Provides a lightweight lexical baseline that is fast and reproducible."
    ),
    (
        7,
        "LLM classifier",
        "Used Qwen3:8b locally through Ollama.",
        "Avoids dependence on cloud quota and makes the core demo reproducible locally."
    ),
    (
        8,
        "Retrieval",
        "Used TF-IDF cosine similarity over historical AmazonHelp customer/reply pairs.",
        "Provides simple, explainable grounding without requiring native embedding infrastructure."
    ),
    (
        9,
        "Retrieval leakage",
        "Excluded golden customer messages from retrieval during golden evaluation.",
        "Prevents direct retrieval of the evaluation input itself."
    ),
    (
        10,
        "Escalation policy",
        "Used conservative deterministic escalation rules for low confidence, weak evidence, payment/account risk, and high-risk language.",
        "Support automation should prefer human review when the cost of an incorrect automated response is high."
    ),
    (
        11,
        "Reply generation",
        "Generated replies with Qwen3:8b using the customer message, intent, and historical evidence.",
        "Combines intent routing with historical resolution patterns."
    ),
    (
        12,
        "Evidence safety",
        "Removed historical URLs, usernames, order numbers, and other identifiers from generated replies.",
        "Historical examples should guide response style/content without leaking customer-specific information."
    ),
    (
        13,
        "LLM judge",
        "Evaluated 30 stratified replies using an explicit 1–5 rubric.",
        "Measures groundedness, correctness/safety, helpfulness, tone, and evidence adherence."
    ),
    (
        14,
        "Human review",
        "Reviewed 12 of the same replies and compared human-reviewed scores with the automated judge.",
        "Checks whether the automated judge is directionally aligned with human assessment."
    ),
    (
        15,
        "Cloud LLM decision",
        "Did not make Gemini a development dependency because project request quota was exhausted during testing.",
        "Local inference keeps the core workflow runnable without external API quota."
    ),
]

lines = [
    "# Decision Log",
    "",
    "| # | Decision | What we did | Why |",
    "|---:|---|---|---|",
]

for number, decision, what, why in decisions:
    lines.append(
        f"| {number} | {decision} | {what} | {why} |"
    )

OUT.write_text("\n".join(lines), encoding="utf-8")

print("=" * 70)
print("DECISION LOG CREATED")
print("=" * 70)
print(f"Decisions: {len(decisions)}")
print(f"Saved: {OUT}")