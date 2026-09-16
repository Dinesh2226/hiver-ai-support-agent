import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.intent.ollama_classifier import OllamaIntentClassifier
from src.retrieval.historical_retriever import HistoricalReplyRetriever
from src.escalation.escalation_engine import decide_escalation
from src.generation.ollama_reply_generator import OllamaReplyGenerator


class SupportAgent:
    """
    End-to-end AI customer-support agent.

    Pipeline:
        Customer message
            -> Intent classification
            -> Historical retrieval
            -> Escalation decision
            -> Grounded reply generation
            -> Deterministic grounding score
    """

    def __init__(
        self,
        historical_data_path="data/processed/amazonhelp_pairs.csv",
    ):
        print("Initializing Support Agent...")

        # ---------------------------------------------------------
        # Intent classifier
        # ---------------------------------------------------------

        self.intent_classifier = OllamaIntentClassifier(
            model="qwen3:8b"
        )

        # ---------------------------------------------------------
        # Historical retrieval
        # ---------------------------------------------------------

        self.retriever = HistoricalReplyRetriever(
            historical_data_path
        )

        # ---------------------------------------------------------
        # Reply generator
        # ---------------------------------------------------------

        self.reply_generator = OllamaReplyGenerator(
            model="qwen3:8b"
        )

        print("Support Agent ready.")

    # =============================================================
    # Grounding score
    # =============================================================

    @staticmethod
    def calculate_grounding_score(
        top_similarity: float,
        evidence_used_count: int,
    ) -> float:
        """
        Calculate a deterministic grounding score.

        This is NOT a calibrated probability.

        It combines:
        - retrieval similarity
        - amount of retrieved evidence actually used

        Similarity is normalized against 0.60 because our current
        retrieval examples frequently reach approximately that level
        for strong matches.
        """

        if evidence_used_count <= 0:
            return 0.0

        # Normalize top similarity to [0, 1].
        similarity_score = min(
            max(top_similarity / 0.60, 0.0),
            1.0,
        )

        # More evidence used gives additional support.
        evidence_score = min(
            evidence_used_count / 3.0,
            1.0,
        )

        score = (
            0.70 * similarity_score
            + 0.30 * evidence_score
        )

        return round(
            min(max(score, 0.0), 1.0),
            2,
        )

    # =============================================================
    # Handle one customer message
    # =============================================================

    def handle(
        self,
        customer_message: str,
    ) -> dict:
        """
        Process one incoming customer message.
        """

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
                "Customer message cannot be empty."
            )

        # =========================================================
        # 1. Intent classification
        # =========================================================

        intent_result = (
            self.intent_classifier.classify(
                customer_message
            )
        )

        intent = intent_result["intent"]

        intent_confidence = float(
            intent_result["confidence"]
        )

        intent_reason = (
            intent_result.get(
                "reason",
                "",
            )
        )

        # =========================================================
        # 2. Historical retrieval
        # =========================================================

        historical_examples = (
            self.retriever.retrieve(
                customer_message,
                top_k=5,
            )
        )

        if historical_examples:
            top_similarity = float(
                historical_examples[0].get(
                    "similarity",
                    0.0,
                )
            )
        else:
            top_similarity = 0.0

        # =========================================================
        # 3. Escalation decision
        # =========================================================

        escalation_result = decide_escalation(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_confidence,
            retrieval_results=historical_examples,
        )

        # =========================================================
        # 4. Grounded reply generation
        # =========================================================

        reply_result = (
            self.reply_generator.generate(
                customer_message=customer_message,
                intent=intent,
                historical_examples=historical_examples,
            )
        )

        evidence_used = reply_result.get(
            "evidence_used",
            [],
        )

        if not isinstance(
            evidence_used,
            list,
        ):
            evidence_used = []

        # =========================================================
        # 5. Deterministic grounding score
        # =========================================================

        grounding_score = (
            self.calculate_grounding_score(
                top_similarity=top_similarity,
                evidence_used_count=len(
                    evidence_used
                ),
            )
        )

        # =========================================================
        # 6. Unified response
        # =========================================================

        return {
            "customer_message": customer_message,

            # Intent
            "intent": intent,
            "intent_confidence": intent_confidence,
            "intent_reason": intent_reason,

            # Reply
            "reply": reply_result["reply"],

            # Evidence
            "evidence_used": evidence_used,
            "grounding_confidence": grounding_score,

            # Explain that this is a heuristic, not a probability.
            "grounding_score_type": (
                "heuristic_retrieval_evidence_score"
            ),

            # Escalation
            "escalate": escalation_result[
                "escalate"
            ],
            "escalation_reason": escalation_result[
                "reason"
            ],

            # Retrieval
            "top_similarity": top_similarity,
            "historical_examples": historical_examples,
        }


# =================================================================
# Standalone test
# =================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("END-TO-END SUPPORT AGENT TEST")
    print("=" * 70)

    agent = SupportAgent()

    test_messages = [
        (
            "My package was supposed to arrive yesterday "
            "but it still hasn't arrived."
        ),
        "Why was my card charged twice?",
        "I cannot log into my Amazon account.",
        "Your customer service is terrible.",
    ]

    for message in test_messages:

        print()
        print("=" * 70)
        print("CUSTOMER")
        print("=" * 70)

        print(message)

        try:

            result = agent.handle(
                message
            )

        except Exception as exc:

            print()
            print(
                f"ERROR: {exc}"
            )

            continue

        print()
        print("=" * 70)
        print("AGENT RESULT")
        print("=" * 70)

        print(
            f"Intent: "
            f"{result['intent']}"
        )

        print(
            f"Intent confidence: "
            f"{result['intent_confidence']:.2f}"
        )

        print(
            f"Intent reason: "
            f"{result['intent_reason']}"
        )

        print(
            f"\nReply:\n"
            f"{result['reply']}"
        )

        print(
            f"\nEscalate: "
            f"{result['escalate']}"
        )

        print(
            f"Escalation reason:\n"
            f"{result['escalation_reason']}"
        )

        print(
            f"\nTop historical similarity: "
            f"{result['top_similarity']:.4f}"
        )

        print(
            f"Evidence used: "
            f"{result['evidence_used']}"
        )

        print(
            f"Grounding score: "
            f"{result['grounding_confidence']:.2f}"
        )

        print(
            f"Grounding score type: "
            f"{result['grounding_score_type']}"
        )