from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(
    "data/processed/correction_set.csv"
)

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


def load_data():
    df = pd.read_csv(DATA_PATH, dtype=str).fillna("")

    for column in [
        "intent",
        "verified_intent",
        "verification_notes",
    ]:
        if column in df.columns:
            df[column] = df[column].astype(str)

    return df


def save_data(df):
    df.to_csv(
        DATA_PATH,
        index=False,
        encoding="utf-8",
    )


def main():
    st.set_page_config(
        page_title="Hiver Label Correction",
        page_icon="🔎",
        layout="wide",
    )

    st.title("Hiver — Training Label Correction")

    if not DATA_PATH.exists():
        st.error(f"File not found: {DATA_PATH}")
        return

    df = load_data()

    if "current_index" not in st.session_state:
        verified = (
            df["verified_intent"]
            .fillna("")
            .str.strip()
            != ""
        )

        remaining = df.index[~verified]

        st.session_state.current_index = (
            int(remaining[0]) if len(remaining) else 0
        )

    index = st.session_state.current_index

    index = max(0, min(index, len(df) - 1))
    row = df.iloc[index]

    verified_count = (
        df["verified_intent"]
        .fillna("")
        .str.strip()
        != ""
    ).sum()

    st.progress(
        verified_count / len(df)
        if len(df)
        else 0
    )

    st.write(
        f"**Verified: {verified_count}/{len(df)}**"
    )

    st.divider()

    st.caption(
        f"Example {index + 1} of {len(df)}"
    )

    st.subheader("Customer message")

    st.info(
        str(row["customer_text"])
    )

    st.subheader("Historical AmazonHelp reply")

    st.write(
        str(row["brand_reply_text"])
    )

    st.subheader("Current machine label")

    st.code(
        str(row["intent"])
    )

    current = str(
        row["verified_intent"]
    ).strip()

    if current not in INTENTS:
        current = str(row["intent"]).strip()

    if current not in INTENTS:
        current = "OTHER"

    selected = st.selectbox(
        "Correct intent",
        INTENTS,
        index=INTENTS.index(current),
    )

    notes = st.text_input(
        "Verification note (optional)",
        value=str(
            row["verification_notes"]
        ),
    )

    col1, col2 = st.columns(2)

    with col1:
        save_next = st.button(
            "Confirm / Correct & Next",
            type="primary",
            use_container_width=True,
        )

    with col2:
        skip = st.button(
            "Skip",
            use_container_width=True,
        )

    if save_next:
        df.loc[index, "verified_intent"] = selected
        df.loc[index, "verification_notes"] = notes

        save_data(df)

        next_indices = df.index[
            df.index > index
        ]

        if len(next_indices):
            st.session_state.current_index = int(
                next_indices[0]
            )
        else:
            st.session_state.current_index = 0

        st.rerun()

    if skip:
        next_indices = df.index[
            df.index > index
        ]

        if len(next_indices):
            st.session_state.current_index = int(
                next_indices[0]
            )
        else:
            st.session_state.current_index = 0

        st.rerun()


if __name__ == "__main__":
    main()