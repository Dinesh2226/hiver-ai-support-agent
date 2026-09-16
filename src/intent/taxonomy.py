INTENTS = {
    "ORDER_DELIVERY": {
        "description": (
            "Questions or problems involving an existing order, "
            "delivery, shipping, tracking, delays, or missing packages."
        )
    },

    "RETURN_REFUND": {
        "description": (
            "Returns, refunds, damaged or wrong items, "
            "or return-pickup problems."
        )
    },

    "ACCOUNT_ACCESS": {
        "description": (
            "Login, password, locked-account, or account-access issues."
        )
    },

    "PAYMENT_BILLING": {
        "description": (
            "Payment failures, billing questions, unexpected charges, "
            "or Amazon Payments issues."
        )
    },

    "TECHNICAL_SUPPORT": {
        "description": (
            "Technical problems involving Amazon apps, devices, "
            "Alexa, Fire TV, Prime Video, or website functionality."
        )
    },

    "PRODUCT_CONTENT": {
        "description": (
            "Product availability, catalog/content availability, "
            "or product information questions."
        )
    },

    "PRICING_PROMOTIONS": {
        "description": (
            "Discounts, promotions, offers, or pricing questions."
        )
    },

    "PRIME_MEMBERSHIP": {
        "description": (
            "Amazon Prime membership, trials, subscriptions, "
            "renewals, cancellations, membership fees, or Prime benefits."
        )
    },

    "COMPLAINT_FEEDBACK": {
        "description": (
            "General dissatisfaction, service complaints, or feedback "
            "without a more specific operational intent."
        )
    },

    "OTHER": {
        "description": (
            "Messages that do not fit another defined support intent."
        )
    },
}

INTENT_NAMES = list(INTENTS.keys())