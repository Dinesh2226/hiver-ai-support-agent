from src.intent.llm_classifier import GeminiIntentClassifier


def main():
    classifier = GeminiIntentClassifier()

    messages = [
        "Where is my package? It was supposed to arrive yesterday.",
        "I cannot log into my Amazon account.",
        "Why was my card charged twice?",
        "My Fire TV is not working.",
        "How do I return this damaged item?",
        "Why did I become an Amazon Prime member?",
        "What discounts are available today?",
        "Is this movie available in India?",
        "Your customer service is terrible.",
        "Thanks Amazon, great delivery!"
    ]

    for message in messages:
        result = classifier.classify(message)

        print("=" * 70)
        print(f"Message:    {message}")
        print(f"Intent:     {result['intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reason:     {result['reason']}")


if __name__ == "__main__":
    main()