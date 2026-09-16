import os
import re
import sys

import streamlit as st

sys.path.insert(0, os.path.abspath("."))

from src.support_agent import SupportAgent


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Amazon Support AI Agent",
    page_icon="🤖",
    layout="wide",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def redact_urls(text: str) -> str:
    """
    Redact URLs from historical evidence before displaying them.
    This prevents old tweet/tracking URLs from appearing as
    actionable customer links in the UI.
    """

    if not text:
        return ""

    text = re.sub(
        r"https?://\S+",
        "[historical link redacted]",
        str(text),
        flags=re.IGNORECASE,
    )

    return text


# ------------------------------------------------------------------
# Initialize agent once
# ------------------------------------------------------------------

@st.cache_resource
def load_agent():
    return SupportAgent(
        historical_data_path=(
            "data/processed/amazonhelp_pairs.csv"
        )
    )


agent = load_agent()


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.title("🤖 AI Customer Support Agent")

st.caption(
    "AmazonHelp historical-support grounded assistant"
)


# ------------------------------------------------------------------
# Customer message
# ------------------------------------------------------------------

customer_message = st.text_area(
    "Customer message",
    height=140,
    placeholder=(
        "Example: My package was supposed to arrive yesterday "
        "but it still hasn't arrived."
    ),
)


# ------------------------------------------------------------------
# Run agent
# ------------------------------------------------------------------

if st.button(
    "Analyze & Draft Reply",
    type="primary",
    use_container_width=True,
):

    if not customer_message.strip():

        st.warning(
            "Please enter a customer message."
        )

    else:

        with st.spinner(
            "Analyzing customer message..."
        ):

            try:
                result = agent.handle(
                    customer_message
                )

            except Exception as exc:

                st.error(
                    f"Agent error: {exc}"
                )

                st.stop()

        # ----------------------------------------------------------
        # Top-level metrics
        # ----------------------------------------------------------

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Intent",
                result["intent"],
            )

        with col2:
            st.metric(
                "Intent confidence",
                f"{result['intent_confidence']:.0%}",
            )

        with col3:

            if result["escalate"]:
                st.metric(
                    "Decision",
                    "Human escalation",
                )
            else:
                st.metric(
                    "Decision",
                    "Auto-handle",
                )

        with col4:
            st.metric(
                "Top similarity",
                f"{result['top_similarity']:.2f}",
            )

        with col5:
            st.metric(
                "Grounding score",
                f"{result['grounding_confidence']:.0%}",
            )

        st.divider()

        # ----------------------------------------------------------
        # Draft reply
        # ----------------------------------------------------------

        st.subheader("Draft reply")

        st.info(
            result["reply"]
        )

        st.caption(
            "The reply is generated from historically similar "
            "AmazonHelp responses and sanitized to remove "
            "historical URLs."
        )

        # ----------------------------------------------------------
        # Escalation
        # ----------------------------------------------------------

        st.subheader(
            "Escalation decision"
        )

        if result["escalate"]:

            st.error(
                "Human review required"
            )

        else:

            st.success(
                "Safe for automated handling"
            )

        st.write(
            result["escalation_reason"]
        )

        # ----------------------------------------------------------
        # Intent explanation
        # ----------------------------------------------------------

        with st.expander(
            "Intent reasoning"
        ):

            st.write(
                result["intent_reason"]
            )

        # ----------------------------------------------------------
        # Grounding explanation
        # ----------------------------------------------------------

        with st.expander(
            "Grounding score explanation"
        ):

            st.write(
                "Grounding score is a heuristic based on "
                "historical retrieval similarity and the amount "
                "of retrieved evidence used by the reply generator."
            )

            st.write(
                "It is not a calibrated probability."
            )

            st.write(
                f"Top retrieval similarity: "
                f"{result['top_similarity']:.3f}"
            )

            st.write(
                f"Evidence used: "
                f"{len(result['evidence_used'])}"
            )

        # ----------------------------------------------------------
        # Evidence
        # ----------------------------------------------------------

        st.subheader(
            "Historical evidence"
        )

        evidence_used = set(
            result["evidence_used"]
        )

        for i, example in enumerate(
            result["historical_examples"],
            start=1,
        ):

            similarity = float(
                example.get(
                    "similarity",
                    0.0,
                )
            )

            if i in evidence_used:

                st.markdown(
                    f"**Evidence {i} "
                    f"• similarity {similarity:.3f}**"
                )

            else:

                st.markdown(
                    f"**Retrieved example {i} "
                    f"• similarity {similarity:.3f}**"
                )

            st.write(
                "Customer:"
            )

            st.caption(
                redact_urls(
                    example.get(
                        "customer_text",
                        "",
                    )
                )
            )

            st.write(
                "Historical AmazonHelp reply:"
            )

            st.caption(
                redact_urls(
                    example.get(
                        "brand_reply_text",
                        "",
                    )
                )
            )

            st.divider()

        # ----------------------------------------------------------
        # Technical details
        # ----------------------------------------------------------

        with st.expander(
            "Agent details"
        ):

            st.json(
                {
                    "intent": result["intent"],
                    "intent_confidence": result[
                        "intent_confidence"
                    ],
                    "intent_reason": result[
                        "intent_reason"
                    ],
                    "top_similarity": result[
                        "top_similarity"
                    ],
                    "evidence_used": result[
                        "evidence_used"
                    ],
                    "grounding_confidence": result[
                        "grounding_confidence"
                    ],
                    "grounding_score_type": result.get(
                        "grounding_score_type",
                        "heuristic_retrieval_evidence_score",
                    ),
                    "escalate": result[
                        "escalate"
                    ],
                    "escalation_reason": result[
                        "escalation_reason"
                    ],
                }
            )