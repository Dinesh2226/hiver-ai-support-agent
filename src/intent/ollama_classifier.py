import json
import time
from typing import Dict

import requests


# =================================================================
# Intent taxonomy
# =================================================================

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
Orders, shipping, tracking, delivery dates, delivery delays,
missing packages, or delivered-but-not-received packages.

RETURN_REFUND:
Returns, refunds, damaged items, wrong items, replacements,
or return pickup.

ACCOUNT_ACCESS:
Login, password, locked account, account recovery,
or account access problems.

PAYMENT_BILLING:
Payment failures, charges, duplicate charges, billing,
unexpected charges, or payment-account issues.

TECHNICAL_SUPPORT:
Alexa, Echo, Fire TV, Prime Video, apps, website errors,
checkout problems, or technical/device problems.

PRODUCT_CONTENT:
Product availability, catalog/content availability,
regional content availability, release information,
or product information.

PRICING_PROMOTIONS:
Prices, discounts, promotions, offers, coupons, deals,
or promotional cashback.

PRIME_MEMBERSHIP:
Any question, complaint, cancellation, renewal, fee,
trial, subscription, enrollment, or benefit concerning
Amazon Prime membership.

COMPLAINT_FEEDBACK:
General complaints or service criticism when there is
NO more specific operational support issue.

OTHER:
Praise, thanks, greetings, vague messages, insufficient
context, or messages that do not fit the other intents.
"""


class OllamaIntentClassifier:
    """
    Local Amazon customer-support intent classifier
    using Ollama + Qwen3.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        max_retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

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
                f"Could not connect to Ollama at "
                f"{self.base_url}. "
                f"Make sure Ollama is running."
            ) from exc

    def _build_prompt(
        self,
        customer_message: str,
    ) -> str:

        return f"""
You are a highly precise Amazon customer-support intent classifier.

Classify ONE customer message into exactly ONE intent.

AVAILABLE INTENTS:
{", ".join(INTENTS)}

DEFINITIONS:
{INTENT_DEFINITIONS}

IMPORTANT PRIORITIES:

1. A specific operational issue beats emotional tone.
2. Prime membership questions MUST be PRIME_MEMBERSHIP.
3. Praise, thanks, greetings, links-only, or vague non-actionable
   messages should be OTHER.
4. General dissatisfaction with no specific operational problem
   should be COMPLAINT_FEEDBACK.
5. Do not infer facts that are not present.

Examples:

Customer: "Why did I become an Amazon Prime member?"
Intent: PRIME_MEMBERSHIP

Customer: "I want to cancel my Prime membership."
Intent: PRIME_MEMBERSHIP

Customer: "Where is my package?"
Intent: ORDER_DELIVERY

Customer: "My package says delivered but I never received it."
Intent: ORDER_DELIVERY

Customer: "How do I return this broken item?"
Intent: RETURN_REFUND

Customer: "I cannot log into my account."
Intent: ACCOUNT_ACCESS

Customer: "Why was my card charged twice?"
Intent: PAYMENT_BILLING

Customer: "My Fire TV is not working."
Intent: TECHNICAL_SUPPORT

Customer: "What discounts are available?"
Intent: PRICING_PROMOTIONS

Customer: "Is this movie available in India?"
Intent: PRODUCT_CONTENT

Customer: "Your customer service is terrible."
Intent: COMPLAINT_FEEDBACK

Customer: "Thanks Amazon, great delivery!"
Intent: OTHER

Customer: "@AmazonHelp"
Intent: OTHER

Customer: "@AmazonHelp https://example.com"
Intent: OTHER

Return ONLY valid JSON.

Required format:

{{
  "intent": "ONE_INTENT",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

CUSTOMER MESSAGE:
{customer_message}
"""

    def classify(
        self,
        customer_message: str,
    ) -> Dict:

        if not isinstance(
            customer_message,
            str,
        ):
            raise TypeError(
                "customer_message must be a string."
            )

        customer_message = customer_message.strip()

        if not customer_message:
            return {
                "intent": "OTHER",
                "confidence": 1.0,
                "reason": "The message is empty.",
            }

        prompt = self._build_prompt(
            customer_message
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise Amazon customer "
                        "support intent classifier."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,

            # Keep structured JSON output.
            "format": "json",

            # Qwen3: disable thinking for fast classification.
            "think": False,

            # Keep the response short.
            "options": {
                "temperature": 0,
            },
        }

        last_error = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):

            try:
                response = requests.post(
                    self.chat_url,
                    json=payload,
                    timeout=self.timeout,
                )

                response.raise_for_status()

                data = response.json()

                break

            except requests.Timeout as exc:

                last_error = (
                    f"Timeout after {self.timeout} seconds."
                )

                print(
                    f"[Ollama] Timeout "
                    f"{attempt}/{self.max_retries}"
                )

                if attempt < self.max_retries:
                    time.sleep(1.5)

            except requests.ConnectionError as exc:

                last_error = (
                    f"Connection error: {exc}"
                )

                print(
                    f"[Ollama] Connection error "
                    f"{attempt}/{self.max_retries}"
                )

                if attempt < self.max_retries:
                    time.sleep(1.5)

            except requests.HTTPError as exc:

                status_code = (
                    exc.response.status_code
                    if exc.response is not None
                    else None
                )

                last_error = (
                    f"Ollama HTTP {status_code}: {exc}"
                )

                print(
                    f"[Ollama] HTTP {status_code} "
                    f"{attempt}/{self.max_retries}"
                )

                if (
                    status_code in {
                        500,
                        502,
                        503,
                        504,
                    }
                    and attempt < self.max_retries
                ):
                    time.sleep(2)

                else:
                    break

            except requests.RequestException as exc:

                last_error = (
                    f"Ollama request failed: {exc}"
                )

                print(
                    f"[Ollama] Request error "
                    f"{attempt}/{self.max_retries}"
                )

                if attempt < self.max_retries:
                    time.sleep(1.5)

        else:
            data = None

        if "data" not in locals() or data is None:
            raise RuntimeError(
                last_error
                or "Ollama request failed."
            )

        # ---------------------------------------------------------
        # Extract content
        # ---------------------------------------------------------

        message_data = data.get(
            "message",
            {},
        )

        raw = str(
            message_data.get(
                "content",
                "",
            )
            or ""
        ).strip()

        # ---------------------------------------------------------
        # Qwen may put output in 'thinking' if thinking is enabled.
        # We explicitly disabled thinking above, but keep this
        # fallback for robustness.
        # ---------------------------------------------------------

        if not raw:

            thinking = str(
                message_data.get(
                    "thinking",
                    "",
                )
                or ""
            ).strip()

            # Try to extract JSON from thinking only as fallback.
            if thinking:
                start = thinking.find("{")
                end = thinking.rfind("}")

                if (
                    start != -1
                    and end != -1
                    and end > start
                ):
                    raw = thinking[
                        start:end + 1
                    ].strip()

        if not raw:
            raise ValueError(
                "Ollama returned empty content.\n"
                f"Full response: {data}"
            )

        # ---------------------------------------------------------
        # Parse JSON
        # ---------------------------------------------------------

        try:
            result = json.loads(raw)

        except json.JSONDecodeError as exc:

            # Try extracting JSON object.
            start = raw.find("{")
            end = raw.rfind("}")

            if (
                start != -1
                and end != -1
                and end > start
            ):

                try:
                    result = json.loads(
                        raw[start:end + 1]
                    )

                except json.JSONDecodeError as inner_exc:
                    raise ValueError(
                        "Ollama returned invalid JSON:\n"
                        f"{raw}"
                    ) from inner_exc

            else:
                raise ValueError(
                    "Ollama returned invalid JSON:\n"
                    f"{raw}"
                ) from exc

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "Ollama response must be a JSON object."
            )

        # ---------------------------------------------------------
        # Validate intent
        # ---------------------------------------------------------

        intent = result.get(
            "intent"
        )

        if intent not in INTENTS:
            raise ValueError(
                f"Invalid intent returned by Ollama: "
                f"{intent}"
            )

        # ---------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------

        try:
            confidence = float(
                result.get(
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

        # ---------------------------------------------------------
        # Reason
        # ---------------------------------------------------------

        reason = str(
            result.get(
                "reason",
                "",
            )
        ).strip()

        return {
            "intent": intent,
            "confidence": confidence,
            "reason": reason,
        }


# =================================================================
# Standalone test
# =================================================================

if __name__ == "__main__":

    classifier = OllamaIntentClassifier(
        model="qwen3:8b"
    )

    test_messages = [
        "Where is my package? It was supposed to arrive yesterday.",
        "I cannot log into my Amazon account.",
        "Why was my card charged twice?",
        "My Fire TV is not working.",
        "How do I return this damaged item?",
        "Why did I become an Amazon Prime member?",
        "What discounts are available today?",
        "Is this movie available in India?",
        "Your customer service is terrible.",
        "Thanks Amazon, great delivery!",
    ]

    expected = [
        "ORDER_DELIVERY",
        "ACCOUNT_ACCESS",
        "PAYMENT_BILLING",
        "TECHNICAL_SUPPORT",
        "RETURN_REFUND",
        "PRIME_MEMBERSHIP",
        "PRICING_PROMOTIONS",
        "PRODUCT_CONTENT",
        "COMPLAINT_FEEDBACK",
        "OTHER",
    ]

    print("=" * 70)
    print("OLLAMA INTENT CLASSIFIER TEST")
    print("=" * 70)

    correct = 0

    for i, message in enumerate(
        test_messages
    ):

        try:
            result = classifier.classify(
                message
            )

            is_correct = (
                result["intent"]
                == expected[i]
            )

            if is_correct:
                correct += 1

            status = (
                "CORRECT"
                if is_correct
                else "WRONG"
            )

            print("-" * 70)
            print(
                f"Message:    {message}"
            )
            print(
                f"Expected:   {expected[i]}"
            )
            print(
                f"Intent:     {result['intent']}"
            )
            print(
                f"Confidence: "
                f"{result['confidence']:.2f}"
            )
            print(
                f"Reason:     "
                f"{result['reason']}"
            )
            print(
                f"Result:     {status}"
            )

        except Exception as exc:

            print("-" * 70)
            print(
                f"Message: {message}"
            )
            print(
                f"ERROR: {exc}"
            )

    print("=" * 70)
    print(
        f"Test accuracy: "
        f"{correct}/{len(test_messages)} "
        f"({correct / len(test_messages):.1%})"
    )
    print("=" * 70)