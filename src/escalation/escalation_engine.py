from typing import Dict, List


HIGH_RISK_INTENTS = {
    "PAYMENT_BILLING",
    "ACCOUNT_ACCESS",
}


def decide_escalation(
    customer_message: str,
    intent: str,
    intent_confidence: float,
    retrieval_results: List[Dict],
) -> Dict:
    """
    Conservative escalation policy.

    Escalates when:
    - intent confidence is low
    - evidence is weak
    - message contains high-risk signals
    - message is too ambiguous to safely automate
    """

    message = customer_message.lower().strip()

    reasons = []
    escalation = False

    # ------------------------------------------------------------
    # 1. Confidence gate
    # ------------------------------------------------------------

    if intent_confidence < 0.70:
        escalation = True
        reasons.append(
            "Low intent-classification confidence."
        )

    # ------------------------------------------------------------
    # 2. Retrieval evidence gate
    # ------------------------------------------------------------

    top_similarity = 0.0

    if retrieval_results:
        top_similarity = float(
            retrieval_results[0].get(
                "similarity",
                0.0,
            )
        )

    if top_similarity < 0.20:
        escalation = True
        reasons.append(
            "No sufficiently similar historical support example."
        )

    # ------------------------------------------------------------
    # 3. Sensitive/risky intents
    # ------------------------------------------------------------

    if intent in HIGH_RISK_INTENTS:
        escalation = True
        reasons.append(
            f"{intent} is routed conservatively to human review."
        )

    # ------------------------------------------------------------
    # 4. Explicit high-risk language
    # ------------------------------------------------------------

    high_risk_terms = [
        "fraud",
        "scam",
        "stolen",
        "hack",
        "hacked",
        "unauthorized",
        "lawsuit",
        "lawyer",
        "police",
        "prosecutor",
        "legal action",
        "identity theft",
        "chargeback",
    ]

    matched_risk_terms = [
        term
        for term in high_risk_terms
        if term in message
    ]

    if matched_risk_terms:
        escalation = True
        reasons.append(
            "High-risk language detected: "
            + ", ".join(matched_risk_terms)
        )

    # ------------------------------------------------------------
    # 5. Severe customer-impact signals
    # ------------------------------------------------------------

    severe_terms = [
        "missing",
        "not received",
        "stolen",
        "wrong item",
        "empty package",
        "charged twice",
        "charged without",
        "account locked",
    ]

    matched_severe_terms = [
        term
        for term in severe_terms
        if term in message
    ]

    if matched_severe_terms:
        escalation = True
        reasons.append(
            "Potentially high-impact issue detected: "
            + ", ".join(matched_severe_terms)
        )

    # ------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------

    if not reasons:
        reasons.append(
            "Intent confidence and historical evidence "
            "are sufficient for automated handling."
        )

    return {
        "escalate": escalation,
        "reason": " ".join(reasons),
        "top_similarity": top_similarity,
        "matched_risk_terms": matched_risk_terms,
        "matched_severe_terms": matched_severe_terms,
    }


if __name__ == "__main__":

    test_cases = [
        {
            "message": (
                "Where is my package? "
                "It was supposed to arrive yesterday."
            ),
            "intent": "ORDER_DELIVERY",
            "confidence": 0.95,
            "retrieval": [
                {"similarity": 0.59},
            ],
        },
        {
            "message": (
                "My card was charged twice."
            ),
            "intent": "PAYMENT_BILLING",
            "confidence": 0.95,
            "retrieval": [
                {"similarity": 0.70},
            ],
        },
        {
            "message": (
                "I think someone stole my account."
            ),
            "intent": "ACCOUNT_ACCESS",
            "confidence": 0.80,
            "retrieval": [
                {"similarity": 0.40},
            ],
        },
        {
            "message": (
                "Thanks Amazon, great service!"
            ),
            "intent": "OTHER",
            "confidence": 0.98,
            "retrieval": [
                {"similarity": 0.50},
            ],
        },
    ]

    print("=" * 70)
    print("ESCALATION ENGINE TEST")
    print("=" * 70)

    for case in test_cases:

        result = decide_escalation(
            customer_message=case["message"],
            intent=case["intent"],
            intent_confidence=case["confidence"],
            retrieval_results=case["retrieval"],
        )

        print()
        print(
            f"Message: {case['message']}"
        )

        print(
            f"Escalate: {result['escalate']}"
        )

        print(
            f"Reason: {result['reason']}"
        )

        print(
            f"Top similarity: "
            f"{result['top_similarity']:.4f}"
        )