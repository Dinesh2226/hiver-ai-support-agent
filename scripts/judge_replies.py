import os
import sys
import json
import time
import re

import pandas as pd
import requests

sys.path.insert(0, os.path.abspath("."))


INPUT_FILE = "data/golden/reply_evaluation.csv"
OUTPUT_FILE = "data/golden/reply_judgments.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"


RUBRIC = {
    "groundedness": (
        "Is the reply supported by the retrieved historical evidence "
        "and the customer's message? Penalize unsupported claims."
    ),
    "correctness_safety": (
        "Is the reply factually safe and does it avoid inventing "
        "account status, refunds, delivery facts, policies, URLs, "
        "or actions that were not established?"
    ),
    "helpfulness": (
        "Does the reply give a useful next step or ask for the "
        "right information needed to help the customer?"
    ),
    "tone": (
        "Is the reply professional, concise, empathetic, and "
        "appropriate for customer support?"
    ),
    "evidence_adherence": (
        "Does the reply actually use the available evidence appropriately "
        "without copying unsupported details, personal identifiers, "
        "or historical URLs?"
    ),
}


def extract_json(text):
    """
    Extract the first JSON object from the model response.
    """

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # JSON inside markdown code fence
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.DOTALL,
    )

    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # First {...} block
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(
        "Could not parse JSON from judge response."
    )


def judge_reply(
    customer_message,
    intent,
    escalation,
    similarity,
    generated_reply,
    evidence_used,
):
    rubric_text = "\n".join(
        f"- {name}: {description}"
        for name, description in RUBRIC.items()
    )

    prompt = f"""
You are evaluating an AI customer-support reply.

Do NOT reward a reply simply because it sounds polite.
Judge whether it is actually appropriate for the customer's message
and supported by the available evidence.

Customer message:
{customer_message}

Predicted intent:
{intent}

Predicted escalation:
{escalation}

Top retrieval similarity:
{similarity}

Evidence used by generator:
{evidence_used}

Generated reply:
{generated_reply}

Evaluation rubric:

{rubric_text}

Score each criterion from 1 to 5:

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Important rules:

- Do not assume facts that are not present.
- Do not reward fabricated policies or guarantees.
- Do not reward invented delivery/refund/account outcomes.
- Penalize unsupported claims.
- Penalize unnecessary personal information.
- Penalize historical URLs or unsupported links.
- A concise request for clarification can score well when the
  original customer message lacks enough information.
- Consider the customer's actual operational issue, not just tone.

Return ONLY valid JSON in exactly this format:

{{
  "groundedness": 1,
  "correctness_safety": 1,
  "helpfulness": 1,
  "tone": 1,
  "evidence_adherence": 1,
  "overall_score": 1,
  "critical_error": false,
  "reason": "brief explanation"
}}
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluation judge. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    message = data.get("message", {})

    content = message.get(
        "content",
        "",
    )

    if not content:
        raise ValueError(
            "Ollama judge returned empty content."
        )

    return extract_json(content)


def clamp_score(value):
    try:
        value = int(float(value))
    except Exception:
        return 0

    return max(1, min(5, value))


def normalize_judgment(result):

    groundedness = clamp_score(
        result.get("groundedness", 0)
    )

    correctness_safety = clamp_score(
        result.get("correctness_safety", 0)
    )

    helpfulness = clamp_score(
        result.get("helpfulness", 0)
    )

    tone = clamp_score(
        result.get("tone", 0)
    )

    evidence_adherence = clamp_score(
        result.get("evidence_adherence", 0)
    )

    overall_score = result.get(
        "overall_score",
        round(
            (
                groundedness
                + correctness_safety
                + helpfulness
                + tone
                + evidence_adherence
            )
            / 5,
            2,
        ),
    )

    try:
        overall_score = float(
            overall_score
        )
    except Exception:
        overall_score = 0.0

    overall_score = max(
        1.0,
        min(5.0, overall_score),
    )

    critical_error = result.get(
        "critical_error",
        False,
    )

    if isinstance(
        critical_error,
        str,
    ):
        critical_error = (
            critical_error
            .strip()
            .lower()
            in {
                "true",
                "yes",
                "1",
            }
        )

    return {
        "groundedness": groundedness,
        "correctness_safety": correctness_safety,
        "helpfulness": helpfulness,
        "tone": tone,
        "evidence_adherence": evidence_adherence,
        "overall_score": overall_score,
        "critical_error": bool(
            critical_error
        ),
        "reason": str(
            result.get(
                "reason",
                "",
            )
        ).strip(),
    }


def main():

    print("=" * 70)
    print("LLM-AS-JUDGE: REPLY QUALITY")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "row_id",
        "customer_text",
        "predicted_intent",
        "predicted_escalation",
        "top_similarity",
        "generated_reply",
        "evidence_used",
        "generation_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df[
        df["generation_status"]
        .astype(str)
        .str.lower()
        .eq("success")
    ].copy()

    print(
        f"Successful replies to judge: {len(df)}"
    )

    if df.empty:
        print("No successful replies available.")
        return

    # ------------------------------------------------------------
    # Resume existing judgments
    # ------------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        previous = pd.read_csv(
            OUTPUT_FILE
        )

        if "row_id" in previous.columns:

            previous_columns = [
                "row_id",
                "groundedness",
                "correctness_safety",
                "helpfulness",
                "tone",
                "evidence_adherence",
                "overall_score",
                "critical_error",
                "judge_reason",
                "judge_status",
                "judge_error",
            ]

            previous_columns = [
                c
                for c in previous_columns
                if c in previous.columns
            ]

            previous = previous[
                previous_columns
            ]

            df = df.merge(
                previous,
                on="row_id",
                how="left",
                suffixes=("", "_old"),
            )

            for column in [
                "groundedness",
                "correctness_safety",
                "helpfulness",
                "tone",
                "evidence_adherence",
                "overall_score",
                "critical_error",
                "judge_reason",
                "judge_status",
                "judge_error",
            ]:

                old_column = f"{column}_old"

                if old_column in df.columns:

                    if column not in df.columns:
                        df[column] = df[
                            old_column
                        ]
                    else:
                        df[column] = df[
                            column
                        ].fillna(
                            df[
                                old_column
                            ]
                        )

                    df.drop(
                        columns=[
                            old_column
                        ],
                        inplace=True,
                    )

    # ------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------

    defaults = {
        "groundedness": None,
        "correctness_safety": None,
        "helpfulness": None,
        "tone": None,
        "evidence_adherence": None,
        "overall_score": None,
        "critical_error": None,
        "judge_reason": None,
        "judge_status": "pending",
        "judge_error": None,
    }

    for column, value in defaults.items():

        if column not in df.columns:
            df[column] = value

    df["judge_status"] = (
        df["judge_status"]
        .fillna("pending")
    )

    pending = list(
        df.index[
            df["judge_status"] != "success"
        ]
    )

    print(
        f"Already judged: "
        f"{len(df) - len(pending)}"
    )

    print(
        f"Remaining: "
        f"{len(pending)}"
    )

    # ------------------------------------------------------------
    # Judge each reply
    # ------------------------------------------------------------

    for position, index in enumerate(
        pending,
        start=1,
    ):

        row = df.loc[index]

        print()
        print("-" * 70)

        print(
            f"Judging {position}/{len(pending)}"
        )

        print(
            f"Row ID: {row['row_id']}"
        )

        print(
            f"Intent: "
            f"{row['predicted_intent']}"
        )

        print(
            f"Message: "
            f"{row['customer_text']}"
        )

        print(
            f"Reply: "
            f"{row['generated_reply']}"
        )

        try:

            result = judge_reply(
                customer_message=str(
                    row["customer_text"]
                ),
                intent=str(
                    row["predicted_intent"]
                ),
                escalation=str(
                    row["predicted_escalation"]
                ),
                similarity=float(
                    row["top_similarity"]
                ),
                generated_reply=str(
                    row["generated_reply"]
                ),
                evidence_used=str(
                    row["evidence_used"]
                ),
            )

            judgment = normalize_judgment(
                result
            )

            df.at[
                index,
                "groundedness",
            ] = judgment[
                "groundedness"
            ]

            df.at[
                index,
                "correctness_safety",
            ] = judgment[
                "correctness_safety"
            ]

            df.at[
                index,
                "helpfulness",
            ] = judgment[
                "helpfulness"
            ]

            df.at[
                index,
                "tone",
            ] = judgment[
                "tone"
            ]

            df.at[
                index,
                "evidence_adherence",
            ] = judgment[
                "evidence_adherence"
            ]

            df.at[
                index,
                "overall_score",
            ] = judgment[
                "overall_score"
            ]

            df.at[
                index,
                "critical_error",
            ] = judgment[
                "critical_error"
            ]

            df.at[
                index,
                "judge_reason",
            ] = judgment[
                "reason"
            ]

            df.at[
                index,
                "judge_status",
            ] = "success"

            df.at[
                index,
                "judge_error",
            ] = None

            print(
                "Scores:"
            )

            print(
                f"  Groundedness: "
                f"{judgment['groundedness']}/5"
            )

            print(
                f"  Correctness/Safety: "
                f"{judgment['correctness_safety']}/5"
            )

            print(
                f"  Helpfulness: "
                f"{judgment['helpfulness']}/5"
            )

            print(
                f"  Tone: "
                f"{judgment['tone']}/5"
            )

            print(
                f"  Evidence adherence: "
                f"{judgment['evidence_adherence']}/5"
            )

            print(
                f"  Overall: "
                f"{judgment['overall_score']:.2f}/5"
            )

            print(
                f"  Critical error: "
                f"{judgment['critical_error']}"
            )

        except Exception as exc:

            error = str(exc)

            df.at[
                index,
                "judge_status",
            ] = "error"

            df.at[
                index,
                "judge_error",
            ] = error

            print(
                f"ERROR: {error}"
            )

        df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        time.sleep(0.5)

    # ------------------------------------------------------------
    # Final metrics
    # ------------------------------------------------------------

    judged = df[
        df["judge_status"] == "success"
    ].copy()

    print()
    print("=" * 70)
    print("LLM-AS-JUDGE RESULTS")
    print("=" * 70)

    print(
        f"Replies judged: {len(judged)}"
    )

    if judged.empty:
        print(
            "No successful judgments."
        )
        return

    score_columns = [
        "groundedness",
        "correctness_safety",
        "helpfulness",
        "tone",
        "evidence_adherence",
        "overall_score",
    ]

    for column in score_columns:

        values = pd.to_numeric(
            judged[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        print(
            f"{column}: "
            f"{values.mean():.2f}"
        )

    critical_errors = (
        judged["critical_error"]
        .astype(str)
        .str.lower()
        .isin(
            {
                "true",
                "yes",
                "1",
            }
        )
        .sum()
    )

    print(
        f"Critical errors: "
        f"{critical_errors}"
    )

    print()
    print(
        "Score distribution:"
    )

    print(
        judged[
            [
                "row_id",
                "overall_score",
                "critical_error",
                "judge_reason",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()