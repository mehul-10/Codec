
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_weather.csv"
ASSETS_DIR = BASE_DIR / "assets"

ASSETS_DIR.mkdir(exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run data preprocessing first."
        )

    return pd.read_csv(DATA_PATH)


def analyze_data(df):

    print("\n========== DATASET OVERVIEW ==========")
    print("Dataset shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)

    print("\n========== STATISTICAL SUMMARY ==========")
    print(df.describe().round(2))

    print("\n========== RAIN TOMORROW DISTRIBUTION ==========")
    print(df["RainTomorrow"].value_counts())

    print("\nRain probability:")
    print(
        (df["RainTomorrow"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\n========== MONTHLY RAIN PROBABILITY ==========")

    monthly_rain = (
        df.groupby("Month")["RainTomorrow"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .round(2)
    )

    print(monthly_rain)

    print("\n========== TOP 10 LOCATIONS BY RAINFALL ==========")

    location_rain = (
        df.groupby("Location")["RainTomorrow"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
        .head(10)
        .round(2)
    )

    print(location_rain)


def create_visualizations(df):

    # 1. Rain distribution
    rain_counts = (
        df["RainTomorrow"]
        .value_counts()
        .rename_axis("RainTomorrow")
        .reset_index(name="Count")
    )

    fig1 = px.pie(
        rain_counts,
        names="RainTomorrow",
        values="Count",
        title="Rain Tomorrow Distribution",
        hole=0.45,
        color="RainTomorrow",
        color_discrete_map={
            "Yes": "#38bdf8",
            "No": "#6366f1"
        }
    )

    fig1.write_html(ASSETS_DIR / "rain_distribution.html")

    # 2. Monthly rain probability
    monthly = (
        df.groupby("Month")["RainTomorrow"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="RainProbability")
    )

    fig2 = px.bar(
        monthly,
        x="Month",
        y="RainProbability",
        title="Monthly Rain Probability",
        labels={
            "Month": "Month",
            "RainProbability": "Rain Probability (%)"
        },
        color="RainProbability",
        color_continuous_scale="Blues"
    )

    fig2.update_xaxes(dtick=1)
    fig2.write_html(ASSETS_DIR / "monthly_rain.html")

    # 3. Temperature trends
    temperature = (
        df.groupby("Month")[["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]]
        .mean()
        .reset_index()
        .melt(
            id_vars="Month",
            var_name="TemperatureType",
            value_name="Temperature"
        )
    )

    fig3 = px.line(
        temperature,
        x="Month",
        y="Temperature",
        color="TemperatureType",
        markers=True,
        title="Average Monthly Temperature Trends"
    )

    fig3.update_xaxes(dtick=1)
    fig3.write_html(ASSETS_DIR / "temperature_trends.html")

    # 4. Numerical feature correlation
    numeric_df = df.select_dtypes(include="number")

    correlation = numeric_df.corr()

    fig4 = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Weather Feature Correlation Heatmap"
    )

    fig4.write_html(ASSETS_DIR / "correlation_heatmap.html")

    print("\nAll EDA visualizations saved successfully!")


if __name__ == "__main__":

    print("Loading processed weather data...")

    weather_df = load_data()

    analyze_data(weather_df)

    create_visualizations(weather_df)

    print("\nEDA completed successfully!")
