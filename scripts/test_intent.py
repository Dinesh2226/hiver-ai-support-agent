from pathlib import Path

import joblib


MODEL_PATH = Path(
    "data/processed/intent_model/intent_classifier.joblib"
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    test_messages = [
        "Where is my package?",
        "My order has not arrived yet.",
        "I cannot sign into my Amazon account.",
        "Why was I charged twice?",
        "My Fire TV is not working.",
        "How do I return this damaged item?",
        "Why did I become an Amazon Prime member?",
        "What discounts are available?",
        "Is this movie available in my country?",
        "Your customer service is terrible.",
    ]

    print("=" * 80)
    print("INTENT PREDICTION TEST")
    print("=" * 80)

    for message in test_messages:
        probabilities = model.predict_proba([message])[0]
        classes = model.classes_

        best_index = probabilities.argmax()

        intent = classes[best_index]
        confidence = probabilities[best_index]

        print(f"\nMessage: {message}")
        print(f"Intent: {intent}")
        print(f"Confidence: {confidence:.3f}")


if __name__ == "__main__":
    main()