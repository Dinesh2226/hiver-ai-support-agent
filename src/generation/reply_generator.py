import json
import os
from typing import List, Dict

from dotenv import load_dotenv
from google import genai


class GroundedReplyGenerator:
    """
    Generates customer-support replies using:
    1. The customer's message
    2. The predicted intent
    3. Historical AmazonHelp replies retrieved from the dataset

    The model is instructed not to invent order-specific information,
    refunds, dates, tracking details, or policies that are not supported
    by the evidence.
    """

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY was not found in .env"
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = "gemini-3.6-flash"

    def generate(
        self,
        customer_message: str,
        intent: str,
        historical_examples: List[Dict],
    ) -> dict:
        """
        Generate a grounded support reply.

        Returns:
        {
            "reply": "...",
            "evidence_used": [...],
            "grounding_confidence": 0.0
        }
        """

        if not customer_message.strip():
            raise ValueError(
                "customer_message cannot be empty."
            )

        if not historical_examples:
            historical_examples = []

        evidence_blocks = []

        for i, example in enumerate(
            historical_examples,
            start=1,
        ):
            evidence_blocks.append(
                f"""
EVIDENCE {i}
Historical customer:
{example.get("customer_text", "")}

Historical AmazonHelp reply:
{example.get("brand_reply_text", "")}

Similarity score:
{float(example.get("similarity", 0.0)):.4f}
"""
            )

        evidence_text = "\n".join(
            evidence_blocks
        )

        if not evidence_text:
            evidence_text = (
                "No sufficiently similar historical "
                "examples were retrieved."
            )

        prompt = f"""
You are an AI customer-support drafting assistant for AmazonHelp.

Your task is to draft ONE concise, professional support reply.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{intent}

HISTORICAL AMAZONHELP EVIDENCE:
{evidence_text}

GROUNDING RULES:

1. Use the historical AmazonHelp replies as evidence for the style,
   troubleshooting steps, and types of responses historically used.

2. Do not copy a historical reply blindly.

3. Do not invent:
   - order status
   - tracking information
   - delivery dates
   - refund amounts
   - account information
   - customer-specific facts
   - policies that are not supported by the evidence

4. Do not claim that you checked an order, account, tracking number,
   payment, or internal system.

5. When the evidence suggests asking the customer for information,
   ask only for information that is actually useful for resolving
   the issue.

6. Keep the response concise and natural, similar to a real customer
   support interaction.

7. Acknowledge the customer's problem when appropriate.

8. If the historical evidence is insufficient, give a cautious reply
   that asks for the necessary information or directs the customer
   to the relevant support/tracking path without making unsupported
   claims.

9. Do not mention that you are an AI.

10. Do not mention this dataset or the historical examples.

Return JSON only in exactly this structure:

{{
  "reply": "customer-facing response",
  "evidence_used": [1, 2],
  "grounding_confidence": 0.0
}}

grounding_confidence must be between 0.0 and 1.0.

The evidence_used array should contain the evidence numbers that
actually influenced the drafted response.
"""

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
        )

        raw = interaction.output_text

        if not raw:
            raise ValueError(
                "Gemini returned empty output."
            )

        raw = raw.strip()

        # ---------------------------------------------------------
        # Remove Markdown code fences
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Parse JSON
        # ---------------------------------------------------------

        try:
            result = json.loads(raw)

        except json.JSONDecodeError:

            start = raw.find("{")
            end = raw.rfind("}")

            if start == -1 or end == -1:
                raise ValueError(
                    "Gemini returned invalid JSON:\n"
                    f"{raw}"
                )

            try:
                result = json.loads(
                    raw[start:end + 1]
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Gemini returned invalid JSON:\n"
                    f"{raw}"
                ) from exc

        # ---------------------------------------------------------
        # Validate result
        # ---------------------------------------------------------

        if not isinstance(result, dict):
            raise ValueError(
                "Gemini response must be a JSON object."
            )

        reply = result.get(
            "reply",
            "",
        )

        if not isinstance(reply, str):
            reply = str(reply)

        reply = reply.strip()

        if not reply:
            raise ValueError(
                "Gemini returned an empty reply."
            )

        evidence_used = result.get(
            "evidence_used",
            [],
        )

        if not isinstance(
            evidence_used,
            list,
        ):
            evidence_used = []

        # Keep only valid integer evidence numbers.
        valid_evidence = []

        for item in evidence_used:

            try:
                number = int(item)

                if 1 <= number <= len(
                    historical_examples
                ):
                    valid_evidence.append(
                        number
                    )

            except (
                TypeError,
                ValueError,
            ):
                continue

        try:
            grounding_confidence = float(
                result.get(
                    "grounding_confidence",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            grounding_confidence = 0.0

        grounding_confidence = max(
            0.0,
            min(
                1.0,
                grounding_confidence,
            ),
        )

        return {
            "reply": reply,
            "evidence_used": valid_evidence,
            "grounding_confidence": grounding_confidence,
        }


if __name__ == "__main__":

    # IMPORTANT:
    # This test uses the retrieval module first.
    from src.retrieval.historical_retriever import (
        HistoricalReplyRetriever,
    )

    DATA_FILE = (
        "data/processed/"
        "amazonhelp_pairs.csv"
    )

    customer_message = (
        "My package was supposed to arrive "
        "yesterday but it still hasn't arrived."
    )

    intent = "ORDER_DELIVERY"

    print(
        "Loading historical retriever..."
    )

    retriever = HistoricalReplyRetriever(
        DATA_FILE
    )

    evidence = retriever.retrieve(
        customer_message,
        top_k=5,
    )

    print(
        f"Retrieved {len(evidence)} historical examples."
    )

    print(
        "Calling Gemini for grounded reply..."
    )

    generator = GroundedReplyGenerator()

    result = generator.generate(
        customer_message=customer_message,
        intent=intent,
        historical_examples=evidence,
    )

    print()
    print("=" * 70)
    print("GROUNDED REPLY TEST")
    print("=" * 70)

    print()
    print(
        f"Customer:\n{customer_message}"
    )

    print()
    print(
        f"Intent:\n{intent}"
    )

    print()
    print(
        f"Reply:\n{result['reply']}"
    )

    print()
    print(
        f"Evidence used: "
        f"{result['evidence_used']}"
    )

    print(
        f"Grounding confidence: "
        f"{result['grounding_confidence']:.2f}"
    )