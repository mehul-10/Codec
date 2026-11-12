
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json

from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="WeatherIQ | AI Rain Prediction",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_weather_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed_weather.csv"
INFO_PATH = BASE_DIR / "models" / "model_info.json"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #080d18;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #101827;
        border-right: 1px solid #263449;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    .metric-card {
        background: linear-gradient(135deg, #172338, #101827);
        border: 1px solid #293a52;
        border-radius: 15px;
        padding: 20px;
        min-height: 115px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 8px;
    }

    .result-card {
        background: linear-gradient(135deg, #142b43, #111b2b);
        border: 1px solid #31506d;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
    }

    .result-label {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    .result-value {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 10px 0;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0284c7, #2563eb);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        min-height: 48px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #0369a1, #1d4ed8);
        border: none;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px 0 10px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found.")

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_weather_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Processed dataset not found.")

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_model_info():
    if INFO_PATH.exists():
        with open(INFO_PATH, "r") as file:
            return json.load(file)

    return {"best_model": "Unknown"}


try:
    model = load_model()
    weather_df = load_weather_data()
    model_info = load_model_info()

except Exception as error:
    st.error(f"Unable to load project resources: {error}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🌦️ WeatherIQ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Weather Analysis & Rain Prediction</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("## 🌤️ WeatherIQ")
st.sidebar.caption("AI Weather Intelligence Platform")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Rain Prediction",
        "Weather Analytics",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 🤖 Model Information")
st.sidebar.write(
    f"**Selected Model:** {model_info['best_model']}"
)
st.sidebar.write("**Prediction Target:** Rain Tomorrow")
st.sidebar.write("**Dataset:** WeatherAUS")


# --------------------------------------------------
# RAIN PREDICTION PAGE
# --------------------------------------------------

if page == "Rain Prediction":

    st.subheader("🌧️ Predict Tomorrow's Rainfall")

    st.write(
        "Enter the current weather conditions to estimate "
        "the likelihood of rainfall tomorrow."
    )

    locations = sorted(
        weather_df["Location"].dropna().unique().tolist()
    )

    with st.form("weather_prediction_form"):

        st.markdown("### 📍 Location & Conditions")

        col1, col2 = st.columns(2)

        with col1:
            location = st.selectbox(
                "Weather Location",
                locations,
                index=locations.index("Albury")
                if "Albury" in locations else 0
            )

            rain_today = st.selectbox(
                "Is it raining today?",
                ["No", "Yes"]
            )

            min_temp = st.number_input(
                "Minimum Temperature (°C)",
                -10.0, 50.0, 15.0
            )

            max_temp = st.number_input(
                "Maximum Temperature (°C)",
                -10.0, 55.0, 25.0
            )

            rainfall = st.number_input(
                "Rainfall (mm)",
                0.0, 500.0, 2.0
            )

            evaporation = st.number_input(
                "Evaporation (mm)",
                0.0, 100.0, 5.0
            )

            sunshine = st.number_input(
                "Sunshine (hours)",
                0.0, 15.0, 7.0
            )

            wind_gust_dir = st.selectbox(
                "Wind Gust Direction",
                ["N", "NE", "E", "SE", "S", "SW", "W", "NW",
                 "NNE", "ENE", "ESE", "SSE", "SSW", "WSW",
                 "WNW", "NNW"]
            )

            wind_gust_speed = st.number_input(
                "Wind Gust Speed (km/h)",
                0.0, 200.0, 35.0
            )

        with col2:
            wind_dir_9am = st.selectbox(
                "Wind Direction at 9 AM",
                ["N", "NE", "E", "SE", "S", "SW", "W", "NW",
                 "NNE", "ENE", "ESE", "SSE", "SSW", "WSW",
                 "WNW", "NNW"]
            )

            wind_dir_3pm = st.selectbox(
                "Wind Direction at 3 PM",
                ["N", "NE", "E", "SE", "S", "SW", "W", "NW",
                 "NNE", "ENE", "ESE", "SSE", "SSW", "WSW",
                 "WNW", "NNW"]
            )

            wind_speed_9am = st.number_input(
                "Wind Speed at 9 AM (km/h)",
                0.0, 150.0, 15.0
            )

            wind_speed_3pm = st.number_input(
                "Wind Speed at 3 PM (km/h)",
                0.0, 150.0, 20.0
            )

            humidity_9am = st.slider(
                "Humidity at 9 AM (%)", 0, 100, 70
            )

            humidity_3pm = st.slider(
                "Humidity at 3 PM (%)", 0, 100, 65
            )

            pressure_9am = st.number_input(
                "Pressure at 9 AM (hPa)",
                900.0, 1100.0, 1015.0
            )

            pressure_3pm = st.number_input(
                "Pressure at 3 PM (hPa)",
                900.0, 1100.0, 1012.0
            )

            cloud_9am = st.slider(
                "Cloud at 9 AM (oktas)", 0, 9, 6
            )

            cloud_3pm = st.slider(
                "Cloud at 3 PM (oktas)", 0, 9, 7
            )

            temp_9am = st.number_input(
                "Temperature at 9 AM (°C)",
                -10.0, 50.0, 18.0
            )

            temp_3pm = st.number_input(
                "Temperature at 3 PM (°C)",
                -10.0, 55.0, 23.0
            )

        st.markdown("### 📅 Date Information")

        date_col1, date_col2, date_col3 = st.columns(3)

        with date_col1:
            year = st.number_input(
                "Year", 2008, 2035, 2016
            )

        with date_col2:
            month = st.selectbox(
                "Month", list(range(1, 13)), index=5
            )

        with date_col3:
            day = st.selectbox(
                "Day", list(range(1, 32)), index=14
            )

        predict_button = st.form_submit_button(
            "🌧️ Predict Tomorrow's Weather",
            use_container_width=True,
            type="primary"
        )

    if predict_button:

        # Match the training feature names
        weather_input = {
            "Location": location,
            "MinTemp": min_temp,
            "MaxTemp": max_temp,
            "Rainfall": rainfall,
            "Evaporation": evaporation,
            "Sunshine": sunshine,
            "WindGustDir": wind_gust_dir,
            "WindGustSpeed": wind_gust_speed,
            "WindDir9am": wind_dir_9am,
            "WindDir3pm": wind_dir_3pm,
            "WindSpeed9am": wind_speed_9am,
            "WindSpeed3pm": wind_speed_3pm,
            "Humidity9am": humidity_9am,
            "Humidity3pm": humidity_3pm,
            "Pressure9am": pressure_9am,
            "Pressure3pm": pressure_3pm,
            "Cloud9am": cloud_9am,
            "Cloud3pm": cloud_3pm,
            "Temp9am": temp_9am,
            "Temp3pm": temp_3pm,
            "RainToday": rain_today,
            "Year": year,
            "Month": month,
            "Day": day
        }

        try:
            input_df = pd.DataFrame([weather_input])

            probabilities = model.predict_proba(input_df)[0]
            prediction = int(model.predict(input_df)[0])

            rain_probability = float(probabilities[1]) * 100
            no_rain_probability = float(probabilities[0]) * 100

            st.session_state["prediction_result"] = {
                "prediction": prediction,
                "rain_probability": rain_probability,
                "no_rain_probability": no_rain_probability
            }

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if "prediction_result" in st.session_state:

        result = st.session_state["prediction_result"]

        st.divider()
        st.subheader("🔮 Prediction Results")

        is_rain = result["prediction"] == 1

        result_label = "RAIN EXPECTED 🌧️" if is_rain else "NO RAIN EXPECTED ☀️"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">AI WEATHER PREDICTION</div>
                <div class="result-value">{result_label}</div>
                <div class="result-label">
                    Model: {model_info['best_model']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "🌧️ Rain Probability",
                f"{result['rain_probability']:.2f}%"
            )

        with c2:
            st.metric(
                "☀️ No Rain Probability",
                f"{result['no_rain_probability']:.2f}%"
            )

        probability_chart = go.Figure(
            go.Bar(
                x=["Rain", "No Rain"],
                y=[
                    result["rain_probability"],
                    result["no_rain_probability"]
                ],
                marker_color=["#38bdf8", "#6366f1"],
                text=[
                    f"{result['rain_probability']:.2f}%",
                    f"{result['no_rain_probability']:.2f}%"
                ],
                textposition="auto"
            )
        )

        probability_chart.update_layout(
            title="Prediction Probability",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#101827"
        )

        st.plotly_chart(
            probability_chart,
            use_container_width=True
        )

        st.caption(
            "This is a machine-learning estimate based on supplied "
            "weather conditions, not an official weather forecast."
        )


# --------------------------------------------------
# WEATHER ANALYTICS PAGE
# --------------------------------------------------

elif page == "Weather Analytics":

    st.subheader("📊 Historical Weather Analytics")

    total_records = len(weather_df)

    rain_count = (
        weather_df["RainTomorrow"] == "Yes"
    ).sum()

    rain_percentage = (
        rain_count / total_records * 100
    )

    avg_max_temp = weather_df["MaxTemp"].mean()

    avg_rainfall = weather_df["Rainfall"].mean()

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, "Weather Records", f"{total_records:,}"),
        (col2, "Rain Probability", f"{rain_percentage:.1f}%"),
        (col3, "Average Max Temp", f"{avg_max_temp:.1f} °C"),
        (col4, "Average Rainfall", f"{avg_rainfall:.2f} mm")
    ]

    for column, label, value in metrics:
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        rain_counts = (
            weather_df["RainTomorrow"]
            .value_counts()
            .rename_axis("Weather")
            .reset_index(name="Count")
        )

        fig = px.pie(
            rain_counts,
            names="Weather",
            values="Count",
            hole=0.5,
            title="Rain Tomorrow Distribution",
            color="Weather",
            color_discrete_map={
                "Yes": "#38bdf8",
                "No": "#6366f1"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#101827"
        )

        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:

        monthly = (
            weather_df.groupby("Month")["RainTomorrow"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="RainProbability")
        )

        fig = px.bar(
            monthly,
            x="Month",
            y="RainProbability",
            title="Monthly Rain Probability",
            color="RainProbability",
            color_continuous_scale="Blues"
        )

        fig.update_xaxes(dtick=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101827",
            plot_bgcolor="#101827"
        )

        st.plotly_chart(fig, use_container_width=True)

    temperature = (
        weather_df.groupby("Month")[
            ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]
        ]
        .mean()
        .reset_index()
        .melt(
            id_vars="Month",
            var_name="Temperature",
            value_name="Value"
        )
    )

    fig = px.line(
        temperature,
        x="Month",
        y="Value",
        color="Temperature",
        markers=True,
        title="Monthly Temperature Trends"
    )

    fig.update_xaxes(dtick=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101827",
        plot_bgcolor="#101827"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📍 Location-wise Rain Probability")

    location_data = (
        weather_df.groupby("Location")["RainTomorrow"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="RainProbability")
        .sort_values("RainProbability", ascending=False)
        .head(15)
    )

    fig = px.bar(
        location_data,
        x="RainProbability",
        y="Location",
        orientation="h",
        title="Top 15 Locations by Rain Probability",
        color="RainProbability",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101827",
        plot_bgcolor="#101827",
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About Project":

    st.subheader("ℹ️ About WeatherIQ")

    st.markdown("""
    ### AI-Powered Weather Analysis and Rain Prediction

    WeatherIQ is a machine-learning application that analyzes
    historical weather observations and estimates the likelihood
    of rainfall tomorrow.

    **Machine Learning Models**
    - Logistic Regression
    - Random Forest Classifier

    **Selected Model**
    - Random Forest Classifier
    - Selection metric: F1 Score

    **Technologies**
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Plotly
    - Joblib

    **Dataset**
    - Australian Weather Dataset
    - 145,460 original records
    - 23 original columns
    - 142,193 labeled records after preprocessing

    **Prediction Target**
    - RainTomorrow: Yes / No

    **Important**
    This application is an educational machine-learning project.
    Its predictions are estimates and should not replace official
    meteorological forecasts.
    """)

    st.info(
        "Developed as part of the Codec Technologies AI Internship."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    WeatherIQ | AI-Powered Weather Intelligence<br>
    Built with Python, Machine Learning & Streamlit
</div>
""", unsafe_allow_html=True)
