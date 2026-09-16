from pathlib import Path

import pandas as pd
import streamlit as st


DATASETS = {
    "Training": Path("data/processed/train_annotations.csv"),
    "Development": Path("data/processed/dev_annotations.csv"),
    "Golden": Path("data/golden/golden_set.csv"),
}

INTENTS = [
    "ORDER_DELIVERY",
    "RETURN_REFUND",
    "ACCOUNT_ACCESS",
    "PAYMENT_BILLING",
    "TECHNICAL_SUPPORT",
    "PRODUCT_CONTENT",
    "PRICING_PROMOTIONS",
    "COMPLAINT_FEEDBACK",
    "OTHER",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    # Convert empty/NaN annotation fields to empty strings.
    for column in ["intent", "intent_notes"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    return df

def save_data(df: pd.DataFrame, path: Path) -> None:
    annotation_columns = [
        "intent",
        "intent_notes",
        "escalation",
        "escalation_reason",
        "ideal_reply_points",
        "difficulty",
    ]

    for column in annotation_columns:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    df.to_csv(
        path,
        index=False,
        encoding="utf-8"
    )


def initialize_state(df: pd.DataFrame, dataset_name: str):
    key = f"index_{dataset_name}"

    if key not in st.session_state:
        labeled = (
            df["intent"].fillna("").astype(str).str.strip() != ""
        )
        first_unlabeled = labeled[labeled == False].index

        st.session_state[key] = (
            int(first_unlabeled[0])
            if len(first_unlabeled) > 0
            else 0
        )


def main():
    st.set_page_config(
        page_title="Hiver Intent Annotator",
        page_icon="💬",
        layout="wide",
    )

    st.title("Hiver — Intent Annotation Tool")

    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["Training", "Development", "Golden"],
    )

    path = DATASETS[dataset_name]

    if not path.exists():
        st.error(f"File not found: {path}")
        return

    df = load_data(path)

    initialize_state(df, dataset_name)

    index_key = f"index_{dataset_name}"

    if len(df) == 0:
        st.warning("Dataset is empty.")
        return

    current_index = st.session_state[index_key]

    # Keep index inside valid range.
    current_index = max(0, min(current_index, len(df) - 1))
    st.session_state[index_key] = current_index

    row = df.iloc[current_index]

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    labeled_mask = (
        df["intent"]
        .fillna("")
        .astype(str)
        .str.strip() != ""
    )

    labeled_count = int(labeled_mask.sum())
    remaining_count = len(df) - labeled_count

    st.progress(
        labeled_count / len(df)
        if len(df)
        else 0
    )

    st.write(
        f"**{dataset_name}:** "
        f"{labeled_count}/{len(df)} labeled "
        f"({remaining_count} remaining)"
    )

    st.divider()

    # ------------------------------------------------------------------
    # Example
    # ------------------------------------------------------------------

    st.caption(
        f"Example {current_index + 1} of {len(df)}"
    )

    st.subheader("Customer message")

    customer_text = str(
        row.get("customer_text", "")
    )

    st.info(customer_text)

    st.subheader("Historical AmazonHelp reply")

    reply_text = str(
        row.get("brand_reply_text", "")
    )

    st.write(reply_text)

    # ------------------------------------------------------------------
    # Existing label
    # ------------------------------------------------------------------

    existing_intent = str(
        row.get("intent", "")
    ).strip()

    if existing_intent not in INTENTS:
        existing_intent = INTENTS[0]

    selected_intent = st.selectbox(
        "Intent",
        INTENTS,
        index=INTENTS.index(existing_intent),
    )

    existing_notes = row.get("intent_notes", "")
    if pd.isna(existing_notes):
        existing_notes = ""

    notes = st.text_input(
        "Annotation notes (optional)",
        value=str(existing_notes),
    )

    # ------------------------------------------------------------------
    # Intent descriptions
    # ------------------------------------------------------------------

    descriptions = {
        "ORDER_DELIVERY":
            "Order status, delivery, tracking, delays, missing package.",

        "RETURN_REFUND":
            "Returns, refunds, damaged/wrong item, return pickup.",

        "ACCOUNT_ACCESS":
            "Login, password, locked account, account access.",

        "PAYMENT_BILLING":
            "Payment failure, billing, unexpected charge, payments account.",

        "TECHNICAL_SUPPORT":
            "Alexa, Fire TV, Prime Video, app, website or device problems.",

        "PRODUCT_CONTENT":
            "Product availability, catalog/content availability, product info.",

        "PRICING_PROMOTIONS":
            "Discounts, promotions, offers, pricing.",

        "COMPLAINT_FEEDBACK":
            "General complaint or feedback without a more specific issue.",

        "OTHER":
            "Does not fit the defined intents or is too ambiguous.",
    }

    st.caption(
        f"**{selected_intent}:** "
        f"{descriptions[selected_intent]}"
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        save_next = st.button(
            "Save & Next",
            type="primary",
            use_container_width=True,
        )

    with col2:
        skip = st.button(
            "Skip",
            use_container_width=True,
        )

    with col3:
        save_only = st.button(
            "Save",
            use_container_width=True,
        )

    if save_only:
        df.loc[current_index, "intent"] = selected_intent
        df.loc[current_index, "intent_notes"] = notes

        save_data(df, path)

        st.success("Saved.")
        st.rerun()

    if save_next:
        df.loc[current_index, "intent"] = selected_intent
        df.loc[current_index, "intent_notes"] = notes

        save_data(df, path)

        next_indices = df.index[df.index > current_index]

        if len(next_indices) > 0:
            st.session_state[index_key] = int(next_indices[0])
        else:
            st.session_state[index_key] = 0

        st.rerun()

    if skip:
        next_indices = df.index[df.index > current_index]

        if len(next_indices) > 0:
            st.session_state[index_key] = int(next_indices[0])
        else:
            st.session_state[index_key] = 0

        st.rerun()

    st.divider()

    st.warning(
        "For the Golden dataset, do not use this tool until the "
        "final system is frozen. Golden labels must remain independent "
        "of model training and tuning."
    )


if __name__ == "__main__":
    main()