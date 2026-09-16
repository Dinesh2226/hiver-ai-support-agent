import os

from dotenv import load_dotenv
from google import genai


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in .env"
        )

    client = genai.Client(
        api_key=api_key
    )

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input="Reply with exactly: GEMINI TEST OK"
    )

    print("=" * 60)
    print("GEMINI API TEST")
    print("=" * 60)
    print(interaction.output_text)


if __name__ == "__main__":
    main()