
import joblib
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_weather_model.pkl"


def load_model():
    """Load the trained machine learning model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run src.train_model first."
        )

    return joblib.load(MODEL_PATH)


def predict_rain(weather_data):
    """
    Predict whether it will rain tomorrow.

    weather_data must contain the weather input features
    used during model training.
    """

    model = load_model()

    # Get original feature names from the trained pipeline
    feature_names = model.named_steps[
        "preprocessor"
    ].feature_names_in_

    # Build input DataFrame in the correct column order
    input_df = pd.DataFrame([weather_data])

    # Add any omitted columns as missing values.
    # The trained pipeline will impute them.
    input_df = input_df.reindex(columns=feature_names)

    # Generate prediction and probability
    prediction = int(model.predict(input_df)[0])

    probabilities = model.predict_proba(input_df)[0]

    rain_probability = float(probabilities[1]) * 100
    no_rain_probability = float(probabilities[0]) * 100

    result = {
        "prediction": "Rain" if prediction == 1 else "No Rain",
        "rain_probability": round(rain_probability, 2),
        "no_rain_probability": round(no_rain_probability, 2)
    }

    return result


if __name__ == "__main__":

    # Example weather conditions for testing
    sample_weather = {
        "Location": "Albury",
        "MinTemp": 15.0,
        "MaxTemp": 25.0,
        "Rainfall": 2.0,
        "Evaporation": 5.0,
        "Sunshine": 7.0,
        "WindGustDir": "W",
        "WindGustSpeed": 35.0,
        "WindDir9am": "NW",
        "WindDir3pm": "W",
        "WindSpeed9am": 15.0,
        "WindSpeed3pm": 20.0,
        "Humidity9am": 70.0,
        "Humidity3pm": 65.0,
        "Pressure9am": 1015.0,
        "Pressure3pm": 1012.0,
        "Cloud9am": 6.0,
        "Cloud3pm": 7.0,
        "Temp9am": 18.0,
        "Temp3pm": 23.0,
        "RainToday": "Yes",
        "Year": 2016,
        "Month": 6,
        "Day": 15
    }

    print("Testing Rain Prediction Model...")

    result = predict_rain(sample_weather)

    print("\n========== PREDICTION RESULT ==========")
    print("Weather Prediction:", result["prediction"])
    print("Rain Probability:", result["rain_probability"], "%")
    print("No Rain Probability:", result["no_rain_probability"], "%")
    print("\nPrediction completed successfully!")
