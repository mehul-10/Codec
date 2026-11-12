
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

RAW_DATA_PATH = DATA_DIR / "weatherAUS.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_weather.csv"


def load_weather_data():
    """Load the original weather dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


def preprocess_weather_data(df):
    """Clean data and prepare features for machine learning."""

    df = df.copy()

    # Remove records without a prediction target
    df = df.dropna(subset=["RainTomorrow"])

    # Normalize target labels
    df["RainTomorrow"] = df["RainTomorrow"].astype(str).str.strip()

    # Extract date-based features
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # Remove original date column
    df = df.drop(columns=["Date"])

    # Remove invalid target labels if any
    df = df[df["RainTomorrow"].isin(["Yes", "No"])]

    # Reset index
    df = df.reset_index(drop=True)

    return df


def save_processed_data(df):
    """Save the processed dataset."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nProcessed dataset saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":

    print("Loading weather dataset...")

    weather_data = load_weather_data()

    print("Original dataset shape:", weather_data.shape)

    print("\nPreprocessing data...")

    processed_data = preprocess_weather_data(weather_data)

    print("\nProcessed dataset shape:", processed_data.shape)

    print("\nTarget distribution:")
    print(processed_data["RainTomorrow"].value_counts())

    print("\nMissing values (top 10):")
    print(
        processed_data.isnull()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    save_processed_data(processed_data)

    print("\nData preprocessing completed successfully!")
