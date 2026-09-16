from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

GOLDEN_FILE = BASE / "data" / "golden" / "golden_predictions.csv"
JUDGE_FILE = BASE / "data" / "golden" / "reply_judgments.csv"
OUT_FILE = BASE / "data" / "golden" / "failure_analysis.md"


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find any of {candidates}\n"
        f"Available columns: {df.columns.tolist()}"
    )


def main():
    golden = pd.read_csv(GOLDEN_FILE)
    judge = pd.read_csv(JUDGE_FILE)

    # ------------------------------------------------------------
    # Identify golden columns
    # ------------------------------------------------------------
    gold_intent_col = find_column(
        golden,
        ["intent", "gold_intent", "true_intent"],
    )

    pred_intent_col = find_column(
        golden,
        ["predicted_intent", "pred_intent", "prediction"],
    )

    message_col = find_column(
        golden,
        ["customer_text", "customer_message", "text"],
    )

    # ------------------------------------------------------------
    # Intent errors
    # ------------------------------------------------------------
    golden["correct"] = (
        golden[gold_intent_col].astype(str).str.strip()
        == golden[pred_intent_col].astype(str).str.strip()
    )

    errors = golden[~golden["correct"]].copy()

    confusion = (
        errors.groupby(
            [gold_intent_col, pred_intent_col]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    # ------------------------------------------------------------
    # Judge columns
    # ------------------------------------------------------------
    judge_overall_col = find_column(
        judge,
        ["overall_score"],
    )

    judge_critical_col = find_column(
        judge,
        ["critical_error"],
    )

    judge_reason_col = find_column(
        judge,
        ["judge_reason", "reason"],
    )

    # ------------------------------------------------------------
    # Reply quality statistics
    # ------------------------------------------------------------
    judge["overall_numeric"] = pd.to_numeric(
        judge[judge_overall_col],
        errors="coerce",
    )

    judge_critical = (
        judge[judge_critical_col]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    mean_reply_score = judge["overall_numeric"].mean()
    critical_count = int(judge_critical.sum())

    # ------------------------------------------------------------
    # Build Markdown
    # ------------------------------------------------------------
    lines = []

    lines.append("# Failure Analysis")
    lines.append("")
    lines.append(
        "Analysis based on the 200-example golden intent benchmark "
        "and 30-example reply-quality judge sample."
    )
    lines.append("")

    # ------------------------------------------------------------
    # Overall intent results
    # ------------------------------------------------------------
    lines.append("## Intent Classification")
    lines.append("")
    lines.append(
        f"- Golden examples: **{len(golden)}**"
    )
    lines.append(
        f"- Incorrect intent predictions: **{len(errors)}**"
    )
    lines.append("")

    # ------------------------------------------------------------
    # Confusion pairs
    # ------------------------------------------------------------
    lines.append("### Largest confusion pairs")
    lines.append("")
    lines.append(
        "| Gold | Predicted | Count |"
    )
    lines.append(
        "|---|---|---:|"
    )

    for _, row in confusion.head(10).iterrows():
        lines.append(
            f"| `{row[gold_intent_col]}` | "
            f"`{row[pred_intent_col]}` | "
            f"{int(row['count'])} |"
        )

    lines.append("")

    # ------------------------------------------------------------
    # Representative examples
    # ------------------------------------------------------------
    lines.append("### Representative classification errors")
    lines.append("")

    for _, row in errors.head(12).iterrows():
        message = (
            str(row[message_col])
            .replace("\n", " ")
            .strip()
        )

        if len(message) > 250:
            message = message[:247] + "..."

        lines.append(
            f"- **Row {row['row_id']}**: "
            f"gold=`{row[gold_intent_col]}`, "
            f"predicted=`{row[pred_intent_col]}`"
        )

        lines.append(
            f"  - Message: {message}"
        )

    lines.append("")

    # ------------------------------------------------------------
    # Top 5 failure modes
    # ------------------------------------------------------------
    lines.append("## Top 5 Failure Modes")
    lines.append("")

    failure_modes = [
        (
            "1. Operational delivery issue vs complaint language",
            "Customers often combine a concrete delivery problem "
            "with strong frustration or criticism. The model can "
            "overweight complaint-style wording.",
            "Prioritize the underlying operational problem over tone "
            "and add counterexamples containing strong emotional language."
        ),
        (
            "2. Complaint/feedback vs OTHER",
            "Short conversational messages can lack enough context "
            "to distinguish dissatisfaction from general commentary.",
            "Add boundary examples and use clarification-first behavior "
            "when intent evidence is weak."
        ),
        (
            "3. Rare-intent undercoverage",
            "Rare intents have limited training and evaluation support, "
            "making their decision boundaries less stable.",
            "Collect more targeted examples for rare intents and use "
            "class-aware sampling."
        ),
        (
            "4. Multilingual and context-poor messages",
            "Short Japanese, French, German, and other multilingual "
            "messages provide weaker lexical evidence for the current "
            "TF-IDF/retrieval pipeline.",
            "Add multilingual examples and evaluate language-specific "
            "performance."
        ),
        (
            "5. Reply generation grounding failures",
            "Generated replies can become generic or introduce facts "
            "that are not supported by the customer message or evidence.",
            "Use evidence-first prompting, prohibit unsupported claims, "
            "and ask a targeted clarification question when evidence "
            "is insufficient."
        ),
    ]

    for title, cause, mitigation in failure_modes:
        lines.append(f"### {title}")
        lines.append(
            f"**Root cause:** {cause}"
        )
        lines.append(
            f"**Mitigation:** {mitigation}"
        )
        lines.append("")

    # ------------------------------------------------------------
    # Reply judge
    # ------------------------------------------------------------
    lines.append("## Reply Generation Evaluation")
    lines.append("")
    lines.append(
        f"- Replies judged: **{len(judge)}**"
    )
    lines.append(
        f"- Mean overall score: **{mean_reply_score:.2f}/5**"
    )
    lines.append(
        f"- Critical errors: **{critical_count}/{len(judge)}**"
    )
    lines.append("")

    lines.append("### Lowest-scoring reply examples")
    lines.append("")

    low = judge.sort_values(
        "overall_numeric",
        ascending=True,
    )

    for _, row in low.head(10).iterrows():
        reason = (
            str(row[judge_reason_col])
            .replace("\n", " ")
            .strip()
        )

        if len(reason) > 240:
            reason = reason[:237] + "..."

        lines.append(
            f"- **Row {row['row_id']}** — "
            f"score={row[judge_overall_col]}/5"
        )
        lines.append(
            f"  - {reason}"
        )

    lines.append("")

    # ------------------------------------------------------------
    # Misleading headline number
    # ------------------------------------------------------------
    lines.append(
        "## What is misleading about my headline number?"
    )
    lines.append("")
    lines.append(
        "Intent accuracy alone does not represent end-to-end support "
        "quality. The 200-example golden set is relatively small and "
        "imbalanced, while several rare intents have very low support."
    )
    lines.append("")
    lines.append(
        "The reply-generation score is also diagnostic rather than "
        "ground truth because the judge is itself an automated LLM. "
        "Human-reviewed ratings should therefore be used to validate "
        "the judge rather than treating its absolute score as a definitive "
        "customer-quality measurement."
    )
    lines.append("")

    # ------------------------------------------------------------
    # Next week
    # ------------------------------------------------------------
    lines.append("## Next-Week Plan")
    lines.append("")
    lines.append(
        "1. Add targeted boundary examples for the highest-confusion "
        "intent pairs."
    )
    lines.append(
        "2. Expand rare-intent and multilingual annotation."
    )
    lines.append(
        "3. Add deterministic precedence rules for high-signal "
        "operational issues."
    )
    lines.append(
        "4. Improve multilingual retrieval once a supported runtime "
        "is available."
    )
    lines.append(
        "5. Add an evidence/unsupported-claim validator after generation."
    )
    lines.append(
        "6. Expand human reply evaluation and periodically re-check "
        "LLM-judge agreement."
    )

    OUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------
    print("=" * 80)
    print("FAILURE ANALYSIS CREATED")
    print("=" * 80)

    print(f"Golden examples: {len(golden)}")
    print(f"Intent errors:   {len(errors)}")
    print()

    print("Top confusion pairs:")
    print(confusion.head(10).to_string(index=False))

    print()
    print(f"Reply judge mean: {mean_reply_score:.2f}/5")
    print(
        f"Critical errors: {critical_count}/{len(judge)}"
    )

    print()
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()