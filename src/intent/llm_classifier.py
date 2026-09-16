import json
import os

from dotenv import load_dotenv
from google import genai


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
Orders, shipping, tracking, delivery dates, delays, missing packages,
or delivered-but-not-received packages.

RETURN_REFUND:
Returns, refunds, damaged/wrong items, replacements, or return pickup.

ACCOUNT_ACCESS:
Login, password, locked account, or account access problems.

PAYMENT_BILLING:
Payment failures, charges, billing, unexpected charges, or payment-account issues.

TECHNICAL_SUPPORT:
Alexa, Echo, Fire TV, Prime Video, apps, website errors, checkout problems,
or technical/device problems.

PRODUCT_CONTENT:
Product availability, catalog/content availability, release information,
or product information.

PRICING_PROMOTIONS:
Prices, discounts, promotions, offers, coupons, deals, or cashback clearly
related to an offer.

PRIME_MEMBERSHIP:
Amazon Prime membership, trial, subscription, renewal, cancellation,
membership fees, or Prime benefits.

COMPLAINT_FEEDBACK:
General complaints or service criticism when there is no more specific
operational support issue.

OTHER:
Praise, thanks, vague requests, insufficient context, or anything that
does not fit the taxonomy.
"""


class GeminiIntentClassifier:
    """
    Gemini-based customer support intent classifier.

    Returns:
    {
        "intent": "...",
        "confidence": 0.0,
        "reason": "..."
    }
    """

    def __init__(self):
        # Load variables from .env
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY was not found in .env"
            )

        # Create Gemini client
        self.client = genai.Client(api_key=api_key)

        # Keep model name in one place so it is easy to change later.
        self.model = "gemini-3.6-flash"

    def classify(self, customer_message: str) -> dict:
        """
        Classify one customer message into exactly one intent.

        Parameters
        ----------
        customer_message : str
            The customer's incoming support message.

        Returns
        -------
        dict
            {
                "intent": str,
                "confidence": float,
                "reason": str
            }
        """

        if not isinstance(customer_message, str):
            raise TypeError(
                "customer_message must be a string"
            )

        customer_message = customer_message.strip()

        if not customer_message:
            return {
                "intent": "OTHER",
                "confidence": 1.0,
                "reason": "The customer message is empty."
            }

        prompt = f"""
You are an Amazon customer-support intent classifier.

Your task is to classify the CUSTOMER MESSAGE into exactly ONE of
the following intents:

{", ".join(INTENTS)}

INTENT DEFINITIONS:

{INTENT_DEFINITIONS}

CLASSIFICATION RULES:

1. Classify the customer's PRIMARY support problem.
2. If a specific operational problem exists, choose that intent even
   when the customer also expresses frustration or anger.
3. Do not infer information that is not present in the message.
4. Use OTHER when the message is vague, only expresses thanks/praise,
   or does not contain enough information to fit another intent.
5. A delivery complaint about a package should be ORDER_DELIVERY.
6. A return/refund/damaged/wrong-item issue should be RETURN_REFUND.
7. A payment charge or billing issue should be PAYMENT_BILLING.
8. A login/password/account-access issue should be ACCOUNT_ACCESS.
9. Technical/device/app/site problems should be TECHNICAL_SUPPORT.
10. Product/content availability or product information should be
    PRODUCT_CONTENT.
11. Discounts, offers, coupons, deals, promotional cashback, or pricing
    questions should be PRICING_PROMOTIONS.
12. Amazon Prime membership questions should be PRIME_MEMBERSHIP.
13. General service criticism without a specific operational issue should
    be COMPLAINT_FEEDBACK.

Return JSON only.

Do NOT use Markdown.
Do NOT wrap the JSON in ``` or ```json.
Do NOT add any text before or after the JSON.

Return exactly this structure:

{{
  "intent": "ONE_INTENT",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

The confidence must be a number between 0.0 and 1.0.

CUSTOMER MESSAGE:
{customer_message}
"""

        # Call Gemini
        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
        )

        # Extract text safely
        raw = getattr(interaction, "output_text", None)

        if not raw:
            raise ValueError(
                "Gemini returned no output text."
            )

        raw = raw.strip()

        # ---------------------------------------------------------
        # Remove Markdown code fences if Gemini still adds them.
        # Example:
        #
        # ```json
        # {"intent": "..."}
        # ```
        # ---------------------------------------------------------
        if raw.startswith("```"):
            lines = raw.splitlines()

            # Remove first fence line
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            # Remove final fence line
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            raw = "\n".join(lines).strip()

        # ---------------------------------------------------------
        # First attempt: parse the complete response as JSON.
        # ---------------------------------------------------------
        try:
            result = json.loads(raw)

        except json.JSONDecodeError:
            # -----------------------------------------------------
            # Fallback: Gemini may have returned some extra text
            # around the JSON object. Extract the outermost object.
            # -----------------------------------------------------
            start = raw.find("{")
            end = raw.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise ValueError(
                    "Gemini returned invalid JSON.\n"
                    f"Raw response:\n{raw}"
                )

            json_text = raw[start:end + 1]

            try:
                result = json.loads(json_text)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Gemini returned invalid JSON.\n"
                    f"Raw response:\n{raw}"
                ) from exc

        # ---------------------------------------------------------
        # Ensure the parsed result is a dictionary.
        # ---------------------------------------------------------
        if not isinstance(result, dict):
            raise ValueError(
                "Gemini JSON response must be an object."
            )

        # ---------------------------------------------------------
        # Validate intent.
        # ---------------------------------------------------------
        intent = result.get("intent")

        if intent not in INTENTS:
            raise ValueError(
                f"Invalid intent returned by Gemini: {intent}\n"
                f"Raw response:\n{raw}"
            )

        # ---------------------------------------------------------
        # Normalize confidence.
        # ---------------------------------------------------------
        confidence_value = result.get("confidence", 0.0)

        try:
            confidence = float(confidence_value)

        except (TypeError, ValueError):
            confidence = 0.0

        # Keep confidence inside valid range.
        confidence = max(0.0, min(1.0, confidence))

        # ---------------------------------------------------------
        # Normalize reason.
        # ---------------------------------------------------------
        reason = result.get("reason", "")

        if reason is None:
            reason = ""

        reason = str(reason).strip()

        # ---------------------------------------------------------
        # Final clean result.
        # ---------------------------------------------------------
        return {
            "intent": intent,
            "confidence": confidence,
            "reason": reason,
        }


if __name__ == "__main__":
    # Small standalone test.
    classifier = GeminiIntentClassifier()

    test_message = (
        "Where is my package? It was supposed to arrive yesterday."
    )

    result = classifier.classify(test_message)

    print(json.dumps(result, indent=2))