from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/twcs.csv")


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    print(f"Dataset: {DATA_PATH}")
    print(f"File size: {DATA_PATH.stat().st_size / (1024 ** 2):.2f} MB")

    # Read only a small sample first.
    df = pd.read_csv(DATA_PATH, nrows=10)

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nShape of inspected sample:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nData types:")
    print(df.dtypes)


if __name__ == "__main__":
    main()