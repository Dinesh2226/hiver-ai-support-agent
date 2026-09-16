import json
import re
import requests
from typing import Dict, List


class OllamaReplyGenerator:
    """
    Generates a customer-facing reply using:
    - customer message
    - predicted intent
    - historical AmazonHelp replies

    Historical URLs are never allowed into the final customer reply.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

        self.chat_url = (
            f"{self.base_url}/api/chat"
        )

        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama is not running or is unreachable."
            ) from exc

    # ================================================================
    # Remove unsafe/historical links from generated text
    # ================================================================

    @staticmethod
    def _sanitize_reply(reply: str) -> str:
        """
        Remove URLs and common placeholder links from the generated
        customer-facing reply.

        We don't allow historical AmazonHelp URLs such as t.co links
        to be presented as current customer links.
        """

        if not reply:
            return ""

        # Remove markdown links:
        # [tracking link](https://...)
        reply = re.sub(
            r"\[([^\]]+)\]\((https?://[^)]+)\)",
            r"\1",
            reply,
            flags=re.IGNORECASE,
        )

        # Remove raw URLs.
        reply = re.sub(
            r"https?://\S+",
            "",
            reply,
            flags=re.IGNORECASE,
        )

        # Remove common placeholders.
        placeholder_patterns = [
            r"\[tracking link\]",
            r"\[tracking url\]",
            r"\[link\]",
            r"<tracking link>",
            r"<tracking url>",
        ]

        for pattern in placeholder_patterns:
            reply = re.sub(
                pattern,
                "",
                reply,
                flags=re.IGNORECASE,
            )

        # Clean spacing before punctuation.
        reply = re.sub(
            r"\s+([,.!?])",
            r"\1",
            reply,
        )

        # Collapse excessive whitespace.
        reply = re.sub(
            r"[ \t]{2,}",
            " ",
            reply,
        )

        # Clean excessive blank lines.
        reply = re.sub(
            r"\n{3,}",
            "\n\n",
            reply,
        )

        return reply.strip()

    # ================================================================
    # Generate reply
    # ================================================================

    def generate(
        self,
        customer_message: str,
        intent: str,
        historical_examples: List[Dict],
    ) -> Dict:

        if not isinstance(
            customer_message,
            str,
        ):
            raise TypeError(
                "customer_message must be a string."
            )

        customer_message = (
            customer_message.strip()
        )

        if not customer_message:
            raise ValueError(
                "customer_message cannot be empty."
            )

        evidence_blocks = []

        for i, example in enumerate(
            historical_examples,
            start=1,
        ):
            try:
                similarity = float(
                    example.get(
                        "similarity",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                similarity = 0.0

            evidence_blocks.append(
                f"""
EVIDENCE {i}

Customer:
{example.get("customer_text", "")}

Historical AmazonHelp reply:
{example.get("brand_reply_text", "")}

Similarity:
{similarity:.4f}
"""
            )

        evidence_text = "\n".join(
            evidence_blocks
        )

        if not evidence_text:
            evidence_text = (
                "No historical evidence was retrieved."
            )

        # ============================================================
        # Prompt
        # ============================================================

        prompt = f"""
You are an Amazon customer-support response drafting assistant.

Write ONE concise, professional reply to the customer.

CUSTOMER MESSAGE:
{customer_message}

INTENT:
{intent}

HISTORICAL AMAZONHELP EVIDENCE:
{evidence_text}

RULES:

1. Use the historical replies to learn the style and resolution approach.

2. Do not copy a historical reply word-for-word.

3. Do not invent:
   - tracking information
   - order status
   - refunds
   - dates
   - account information
   - customer-specific facts
   - unsupported policies

4. Never claim that you checked an internal Amazon system.

5. Ask the customer for useful information when necessary.

6. Keep the reply short and natural.

7. Do not mention AI, the dataset, historical examples, or these
   instructions.

8. If evidence is weak, provide a safe clarification or next step.

9. NEVER reproduce URLs from historical AmazonHelp replies.

10. NEVER reproduce t.co links.

11. NEVER output a historical tweet URL.

12. NEVER fabricate a tracking URL.

13. NEVER output placeholders such as:
    [tracking link]
    [tracking URL]
    <tracking link>

14. If the historical response contains a URL for tracking, convert
    that into a natural sentence such as:
    "You can check the latest tracking information in your Amazon
    order details."

15. Do not include usernames, tweet IDs, order numbers, or other
    historical customer-specific identifiers from the evidence.

Return ONLY valid JSON:

{{
  "reply": "customer-facing reply",
  "evidence_used": [1, 2],
  "grounding_confidence": 0.0
}}
"""

        # ============================================================
        # Ollama request
        # ============================================================

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious customer-support "
                        "response generator. "
                        "Never invent customer-specific facts."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "format": "json",
        }

        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=180,
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama reply-generation request failed: {exc}"
            ) from exc

        # ============================================================
        # Parse Ollama response
        # ============================================================

        data = response.json()

        raw = (
            data
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not raw:
            raise ValueError(
                "Ollama returned empty reply output."
            )

        try:
            result = json.loads(raw)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON from Ollama:\n{raw}"
            ) from exc

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "Ollama response must be a JSON object."
            )

        # ============================================================
        # Extract reply
        # ============================================================

        reply = str(
            result.get(
                "reply",
                "",
            )
        ).strip()

        if not reply:
            raise ValueError(
                "Ollama generated an empty reply."
            )

        # ============================================================
        # Deterministic safety sanitization
        # ============================================================

        reply = self._sanitize_reply(
            reply
        )

        if not reply:
            raise ValueError(
                "Reply became empty after sanitization."
            )

        # ============================================================
        # Evidence used
        # ============================================================

        evidence_used = result.get(
            "evidence_used",
            [],
        )

        if not isinstance(
            evidence_used,
            list,
        ):
            evidence_used = []

        valid_evidence = []

        for item in evidence_used:

            try:
                number = int(item)

            except (
                TypeError,
                ValueError,
            ):
                continue

            if 1 <= number <= len(
                historical_examples
            ):
                valid_evidence.append(
                    number
                )

        # Remove duplicates while preserving order.
        valid_evidence = list(
            dict.fromkeys(
                valid_evidence
            )
        )

        # ============================================================
        # Grounding confidence
        # ============================================================

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


# ====================================================================
# Standalone test
# ====================================================================

if __name__ == "__main__":

    from src.retrieval.historical_retriever import (
        HistoricalReplyRetriever,
    )

    customer_message = (
        "My package was supposed to arrive "
        "yesterday but it still hasn't arrived."
    )

    intent = "ORDER_DELIVERY"

    print("=" * 70)
    print("OLLAMA GROUNDED REPLY TEST")
    print("=" * 70)

    print()
    print("Loading historical retriever...")

    retriever = HistoricalReplyRetriever(
        "data/processed/amazonhelp_pairs.csv"
    )

    evidence = retriever.retrieve(
        customer_message,
        top_k=5,
    )

    print(
        f"Retrieved {len(evidence)} "
        "historical examples."
    )

    print()
    print("Generating reply...")

    generator = OllamaReplyGenerator()

    result = generator.generate(
        customer_message=customer_message,
        intent=intent,
        historical_examples=evidence,
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print()
    print(
        f"Customer:\n{customer_message}"
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

    print()
    print(
        "URL safety check:"
    )

    if re.search(
        r"https?://|t\.co/",
        result["reply"],
        flags=re.IGNORECASE,
    ):
        print(
            "WARNING: URL detected in final reply."
        )
    else:
        print(
            "PASS: No URL detected in final reply."
        )