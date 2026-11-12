import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="WeatherIQ | AI Weather Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_weather_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed_weather.csv"
INFO_PATH = BASE_DIR / "models" / "model_info.json"


# ==================================================
# SESSION STATE
# ==================================================

if "view" not in st.session_state:
    st.session_state.view = "home"

if "dashboard_page" not in st.session_state:
    st.session_state.dashboard_page = "Rain Prediction"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


def navigate(view: str, page: str | None = None):
    st.session_state.view = view
    if page:
        st.session_state.dashboard_page = page


# ==================================================
# BRIGHT, RESPONSIVE DESIGN
# ==================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172554;
        --muted: #64748b;
        --blue: #2563eb;
        --cyan: #0891b2;
        --line: #e5edf7;
        --surface: #ffffff;
        --soft: #f5f9ff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(186, 230, 253, .34), transparent 25rem),
            radial-gradient(circle at 94% 12%, rgba(221, 214, 254, .30), transparent 26rem),
            #f8fbff;
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(248, 251, 255, .86);
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    h1, h2, h3 {
        color: var(--ink) !important;
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.035em;
    }

    p, label, .stMarkdown, .stCaption {
        color: #475569;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #172554;
    }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 13px;
        color: white;
        background: linear-gradient(135deg, #0ea5e9, #4f46e5);
        box-shadow: 0 8px 18px rgba(37, 99, 235, .20);
        font-size: 1.2rem;
    }
    
    .hero-shell {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin: 0 auto;
}

.hero-shell h1,
.hero-shell p {
    text-align: center !important;
    margin-left: auto;
    margin-right: auto;
}

.hero-shell h1 {
    max-width: 900px;
}

.hero-shell p {
    max-width: 750px;
}

    .stButton > button,
.stButton > button p,
.stButton > button span {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 700 !important;
}

     .js-plotly-plot .plotly text {
    fill: #000000 !important;
}

.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text,
.js-plotly-plot .legend text,
.js-plotly-plot .gtitle {
    fill: #000000 !important;
}

   fig.update_layout(
    font=dict(
        family="Arial",
        color="#000000",
        size=13
    ),
    title_font=dict(color="#000000"),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF"
)

fig.update_xaxes(
    tickfont=dict(color="#000000"),
    title_font=dict(color="#000000")
)

fig.update_yaxes(
    tickfont=dict(color="#000000"),
    title_font=dict(color="#000000")
)   

    

    .hero-shell {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(191, 219, 254, .75);
        border-radius: 30px;
        padding: clamp(2rem, 6vw, 5.5rem) clamp(1.25rem, 5vw, 4.5rem);
        background:
            radial-gradient(circle at 12% 12%, rgba(186, 230, 253, .75), transparent 22rem),
            radial-gradient(circle at 92% 18%, rgba(221, 214, 254, .75), transparent 25rem),
            linear-gradient(135deg, #ffffff 0%, #f3f8ff 55%, #f8f5ff 100%);
        box-shadow: 0 24px 70px rgba(30, 64, 175, .08);
        text-align: center;
    }

    .hero-shell:before, .hero-shell:after {
        content: "";
        position: absolute;
        border-radius: 999px;
        pointer-events: none;
        filter: blur(1px);
    }

    .hero-shell:before {
        width: 230px;
        height: 230px;
        right: -85px;
        top: -100px;
        background: rgba(125, 211, 252, .20);
    }

    .hero-shell:after {
        width: 190px;
        height: 190px;
        left: -85px;
        bottom: -100px;
        background: rgba(196, 181, 253, .22);
    }

    .announcement {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border: 1px solid #dbeafe;
        border-radius: 999px;
        background: rgba(255,255,255,.82);
        color: #475569;
        font-size: .83rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    .announcement span {
        color: #2563eb;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        max-width: 900px;
        margin: 0 auto;
        color: #172554;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.45rem, 6vw, 4.7rem);
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -.065em;
    }

    .gradient-text {
        background: linear-gradient(90deg, #0284c7 5%, #4f46e5 55%, #7c3aed 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-description {
        position: relative;
        z-index: 1;
        max-width: 690px;
        margin: 1.5rem auto 0;
        color: #64748b;
        font-size: clamp(1rem, 2vw, 1.15rem);
        line-height: 1.8;
    }

    .hero-note {
        margin-top: 1.2rem;
        color: #94a3b8;
        font-size: .82rem;
    }

    .section-kicker {
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .75rem;
        font-weight: 800;
    }

    .section-heading {
        color: #172554;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: clamp(1.65rem, 3vw, 2.25rem);
        letter-spacing: -.045em;
        margin: .35rem 0 .5rem;
    }

    .section-copy {
        color: #64748b;
        line-height: 1.7;
    }

    .feature-card, .metric-card, .result-card, .info-card {
        height: 100%;
        background: rgba(255,255,255,.94);
        border: 1px solid #e4edf8;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 10px 30px rgba(30, 64, 175, .045);
    }

    .feature-icon {
        display: inline-flex;
        width: 46px;
        height: 46px;
        align-items: center;
        justify-content: center;
        border-radius: 15px;
        background: linear-gradient(135deg, #e0f2fe, #ede9fe);
        font-size: 1.35rem;
        margin-bottom: 14px;
    }

    .feature-title {
        color: #1e293b;
        font-weight: 800;
        font-size: 1.03rem;
        margin-bottom: 7px;
    }

    .feature-copy {
        color: #64748b;
        font-size: .91rem;
        line-height: 1.65;
    }

    .metric-label {
        color: #64748b;
        font-size: .84rem;
        font-weight: 600;
    }

    .metric-value {
        color: #172554;
        font-family: 'Manrope', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .result-card {
        text-align: center;
        padding: 30px 20px;
        background: linear-gradient(135deg, #ffffff, #eff6ff);
        border-color: #dbeafe;
    }

    .result-label {
        color: #64748b;
        font-size: .85rem;
        font-weight: 700;
        letter-spacing: .08em;
    }

    .result-value {
        color: #2563eb;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.6rem, 4vw, 2.35rem);
        font-weight: 800;
        margin: 12px 0;
    }

    .page-heading {
        color: #172554;
        font-family: 'Manrope', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -.05em;
        margin-bottom: .2rem;
    }

    .page-subtitle {
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 12px;
        min-height: 45px;
        font-weight: 700;
        border: 1px solid #dbeafe;
        transition: all .2s ease;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        color: white;
        border: none;
        background: linear-gradient(100deg, #0284c7, #4f46e5);
        box-shadow: 0 8px 18px rgba(37, 99, 235, .17);
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        color: white;
        border: none;
        filter: brightness(1.05);
        box-shadow: 0 10px 24px rgba(37, 99, 235, .24);
    }

    div.stButton > button:not([kind="primary"]) {
        background: #ffffff;
        color: #334155;
    }

    div.stButton > button:not([kind="primary"]):hover {
        border-color: #93c5fd;
        color: #1d4ed8;
        background: #f8fbff;
    }

    [data-testid="stForm"] {
        border: 1px solid #e2eaf5;
        border-radius: 18px;
        padding: 22px;
        background: rgba(255,255,255,.82);
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5edf7;
        border-radius: 16px;
        padding: 15px 17px;
        box-shadow: 0 7px 22px rgba(30,64,175,.035);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #172554 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #ffffff;
        border-color: #dbe5f1;
        border-radius: 10px;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: #2563eb;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 30px 0 10px;
        font-size: .84rem;
    }

    .dashboard-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
        padding-bottom: 12px;
        border-bottom: 1px solid #e5edf7;
        margin-bottom: 22px;
    }

    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero-shell {
            border-radius: 22px;
            padding: 2.4rem 1.1rem;
        }
        .feature-card, .metric-card {
            padding: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# LOAD MODEL AND DATA
# ==================================================

@st.cache_data(show_spinner=False)
def get_weather_summary(df):
    total_records = len(df)
    rain_count = (df["RainTomorrow"] == "Yes").sum()

    rain_percentage = (
        rain_count / total_records * 100
        if total_records else 0
    )

    average_temperature = df["MaxTemp"].mean()

    return total_records, rain_percentage, average_temperature

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_weather_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_model_info():
    if INFO_PATH.exists():
        with open(INFO_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"best_model": "Unknown"}


try:
    model = load_model()
    weather_df = load_weather_data()
    model_info = load_model_info()
except Exception as error:
    st.error(f"Unable to load project resources: {error}")
    st.stop()


# ==================================================
# SHARED HELPERS
# ==================================================

def render_brand():
    st.markdown(
        """
        <div class="brand">
            <span class="brand-mark">☀</span>
            <span>WeatherIQ</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="footer">
            WeatherIQ · AI-Powered Weather Intelligence<br>
            Built with Python, Machine Learning & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig, height=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="DM Sans, sans-serif", color="#475569"),
        title_font=dict(color="#172554", size=17),
        margin=dict(l=18, r=18, t=55, b=18),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color="#64748b"),
        ),
    )
    fig.update_xaxes(
        gridcolor="#edf2f8",
        zerolinecolor="#edf2f8",
        linecolor="#dbe5f1",
    )
    fig.update_yaxes(
        gridcolor="#edf2f8",
        zerolinecolor="#edf2f8",
        linecolor="#dbe5f1",
    )
    if height:
        fig.update_layout(height=height)
    return fig


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# HOME / LANDING PAGE
# ==================================================

def render_home():
    # Top navigation
    nav_left, nav_mid, nav_right = st.columns([2.2, 3.8, 2.0])
    if st.session_state.get("show_features", False):
     st.markdown(
        '<div id="features"></div>',
        unsafe_allow_html=True
    )
    with nav_left:
        render_brand()

    with nav_mid:
     n1, n2, n3 = st.columns(3)

    with n1:
        if st.button("Home", key="nav_home", use_container_width=True):
            st.session_state.show_features = False
            navigate("home")
            st.rerun()

    with n2:
        if st.button("Features", key="nav_features", use_container_width=True):
            st.session_state.show_features = True
            navigate("home")
            st.rerun()

    with n3:
        if st.button("About", key="nav_about", use_container_width=True):
            navigate("dashboard", "About Project")
            st.rerun()

    with nav_right:
        if st.button(
            "Open Dashboard →",
            key="nav_dashboard",
            type="primary",
            use_container_width=True,
        ):
            navigate("dashboard", "Rain Prediction")
            st.rerun()

    st.write("")

    st.markdown(
        """
        <section class="hero-shell">
            <div class="announcement">
                <span>✦</span> AI-powered weather intelligence
                <span>·</span> Explore smarter predictions
            </div>
            <h1 class="hero-title">
                Understand Weather.<br>
                <span class="gradient-text">Predict Rain.</span><br>
                Make Smarter Decisions.
            </h1>
            <p class="hero-description">
                Turn weather observations into meaningful insights.
                Explore historical patterns and estimate the likelihood
                of rainfall tomorrow with a machine-learning model.
            </p>
            <div class="hero-note">
                Historical Australian weather data · Random Forest ML model
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    cta1, cta2, cta3 = st.columns([1.2, 1.5, 1.2])
    with cta2:
        if st.button(
            "🌦️  Get Started — Explore WeatherIQ",
            key="hero_get_started",
            type="primary",
            use_container_width=True,
        ):
            navigate("dashboard", "Rain Prediction")
            st.rerun()

    st.write("")
    st.markdown(
        """
        <div style="text-align:center; padding: 1.2rem 0 1rem;">
            <div class="section-kicker">A clearer view of the weather</div>
            <div class="section-heading">Everything you need to explore tomorrow</div>
            <div class="section-copy">
                From machine-learning predictions to historical weather trends,
                all your insights are in one place.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)
    features = [
        (
            f1,
            "🌧️",
            "AI Rain Prediction",
            "Enter weather conditions and get a model-based estimate of rain and no-rain probabilities.",
        ),
        (
            f2,
            "📊",
            "Weather Analytics",
            "Explore rainfall distribution, monthly patterns, temperature trends and location comparisons.",
        ),
        (
            f3,
            "🧠",
            "Machine Learning",
            "Powered by a trained Random Forest pipeline with preprocessing and feature handling.",
        ),
    ]

    for column, icon, title, description in features:
        with column:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-copy">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        """
        <div class="section-kicker" style="text-align:center;">Built for exploration</div>
        <div class="section-heading" style="text-align:center;">Weather insights at a glance</div>
        """,
        unsafe_allow_html=True,
    )

    total_records = len(weather_df)
    rain_count = (weather_df["RainTomorrow"] == "Yes").sum()
    rain_percentage = rain_count / total_records * 100
    avg_max_temp = weather_df["MaxTemp"].mean()

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Historical observations", f"{total_records:,}")
    with m2:
        metric_card("Rain tomorrow in dataset", f"{rain_percentage:.1f}%")
    with m3:
        metric_card("Average maximum temperature", f"{avg_max_temp:.1f} °C")

    st.write("")
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown(
            """
            <div class="info-card">
                <div class="section-kicker">Your next step</div>
                <div class="section-heading" style="font-size:1.55rem;">
                    Explore the prediction dashboard
                </div>
                <div class="section-copy">
                    Adjust weather inputs, run a prediction, and inspect
                    the model's estimated probabilities. You can also
                    switch to historical analytics at any time.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if st.button(
            "Open Rain Prediction →",
            key="home_prediction",
            type="primary",
            use_container_width=True,
        ):
            navigate("dashboard", "Rain Prediction")
            st.rerun()
        if st.button(
            "View Historical Analytics",
            key="home_analytics",
            use_container_width=True,
        ):
            navigate("dashboard", "Weather Analytics")
            st.rerun()

    render_footer()


# ==================================================
# RAIN PREDICTION PAGE
# ==================================================

def render_prediction():
    st.markdown('<div class="page-heading">Rain Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Enter weather observations to estimate the likelihood of rainfall tomorrow.</div>',
        unsafe_allow_html=True,
    )

    locations = sorted(weather_df["Location"].dropna().unique().tolist())
    wind_directions = [
        "N", "NE", "E", "SE", "S", "SW", "W", "NW",
        "NNE", "ENE", "ESE", "SSE", "SSW", "WSW", "WNW", "NNW",
    ]

    with st.form("weather_prediction_form"):
        st.markdown("### 📍 Location & weather conditions")
        col1, col2 = st.columns(2)

        with col1:
            location = st.selectbox(
                "Weather Location",
                locations,
                index=locations.index("Albury") if "Albury" in locations else 0,
            )
            rain_today = st.selectbox("Is it raining today?", ["No", "Yes"])
            min_temp = st.number_input("Minimum Temperature (°C)", -10.0, 50.0, 15.0)
            max_temp = st.number_input("Maximum Temperature (°C)", -10.0, 55.0, 25.0)
            rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 2.0)
            evaporation = st.number_input("Evaporation (mm)", 0.0, 100.0, 5.0)
            sunshine = st.number_input("Sunshine (hours)", 0.0, 15.0, 7.0)
            wind_gust_dir = st.selectbox("Wind Gust Direction", wind_directions)
            wind_gust_speed = st.number_input("Wind Gust Speed (km/h)", 0.0, 200.0, 35.0)

        with col2:
            wind_dir_9am = st.selectbox("Wind Direction at 9 AM", wind_directions)
            wind_dir_3pm = st.selectbox("Wind Direction at 3 PM", wind_directions)
            wind_speed_9am = st.number_input("Wind Speed at 9 AM (km/h)", 0.0, 150.0, 15.0)
            wind_speed_3pm = st.number_input("Wind Speed at 3 PM (km/h)", 0.0, 150.0, 20.0)
            humidity_9am = st.slider("Humidity at 9 AM (%)", 0, 100, 70)
            humidity_3pm = st.slider("Humidity at 3 PM (%)", 0, 100, 65)
            pressure_9am = st.number_input("Pressure at 9 AM (hPa)", 900.0, 1100.0, 1015.0)
            pressure_3pm = st.number_input("Pressure at 3 PM (hPa)", 900.0, 1100.0, 1012.0)
            cloud_9am = st.slider("Cloud at 9 AM (oktas)", 0, 9, 6)
            cloud_3pm = st.slider("Cloud at 3 PM (oktas)", 0, 9, 7)
            temp_9am = st.number_input("Temperature at 9 AM (°C)", -10.0, 50.0, 18.0)
            temp_3pm = st.number_input("Temperature at 3 PM (°C)", -10.0, 55.0, 23.0)

        st.markdown("### 📅 Date information")
        date_col1, date_col2, date_col3 = st.columns(3)
        with date_col1:
            year = st.number_input("Year", 2008, 2035, 2016)
        with date_col2:
            month = st.selectbox("Month", list(range(1, 13)), index=5)
        with date_col3:
            day = st.selectbox("Day", list(range(1, 32)), index=14)

        predict_button = st.form_submit_button(
            "🌧️ Predict Tomorrow's Weather",
            use_container_width=True,
            type="primary",
        )

    if predict_button:
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
            "Day": day,
        }

        try:
            input_df = pd.DataFrame([weather_input])
            probabilities = model.predict_proba(input_df)[0]
            prediction = int(model.predict(input_df)[0])

            st.session_state.prediction_result = {
                "prediction": prediction,
                "rain_probability": float(probabilities[1]) * 100,
                "no_rain_probability": float(probabilities[0]) * 100,
            }
        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result
        is_rain = result["prediction"] == 1
        result_label = "RAIN EXPECTED 🌧️" if is_rain else "NO RAIN EXPECTED ☀️"

        st.write("")
        st.markdown("### 🔮 Prediction results")
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">AI WEATHER PREDICTION</div>
                <div class="result-value">{result_label}</div>
                <div class="result-label">Model: {model_info.get('best_model', 'Unknown')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🌧️ Rain Probability", f"{result['rain_probability']:.2f}%")
        with c2:
            st.metric("☀️ No Rain Probability", f"{result['no_rain_probability']:.2f}%")

        probability_chart = go.Figure(
            go.Bar(
                x=["Rain", "No Rain"],
                y=[result["rain_probability"], result["no_rain_probability"]],
                marker_color=["#0ea5e9", "#6366f1"],
                text=[
                    f"{result['rain_probability']:.2f}%",
                    f"{result['no_rain_probability']:.2f}%",
                ],
                textposition="auto",
            )
        )
        probability_chart.update_layout(
            title="Prediction probability",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            showlegend=False,
        )
        st.plotly_chart(style_plotly(probability_chart, 360), use_container_width=True)
        st.caption(
            "This is a machine-learning estimate based on the supplied weather conditions, "
            "not an official or live weather forecast."
        )


# ==================================================
# WEATHER ANALYTICS PAGE
# ==================================================

def render_analytics():
    st.markdown('<div class="page-heading">Weather Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Explore historical weather patterns in the WeatherAUS dataset.</div>',
        unsafe_allow_html=True,
    )

    total_records = len(weather_df)
    rain_count = (weather_df["RainTomorrow"] == "Yes").sum()
    rain_percentage = rain_count / total_records * 100
    avg_max_temp = weather_df["MaxTemp"].mean()
    avg_rainfall = weather_df["Rainfall"].mean()

    col1, col2, col3, col4 = st.columns(4)
    values = [
        (col1, "Weather records", f"{total_records:,}"),
        (col2, "Rain tomorrow", f"{rain_percentage:.1f}%"),
        (col3, "Average max temp", f"{avg_max_temp:.1f} °C"),
        (col4, "Average rainfall", f"{avg_rainfall:.2f} mm"),
    ]
    for column, label, value in values:
        with column:
            metric_card(label, value)

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
            hole=0.58,
            title="Rain tomorrow distribution",
            color="Weather",
            color_discrete_map={"Yes": "#0ea5e9", "No": "#6366f1"},
        )
        st.plotly_chart(style_plotly(fig, 390), use_container_width=True)

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
            title="Monthly rain probability",
            color="RainProbability",
            color_continuous_scale=["#bfdbfe", "#2563eb", "#4f46e5"],
        )
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(title="Rain probability (%)")
        st.plotly_chart(style_plotly(fig, 390), use_container_width=True)

    temperature = (
        weather_df.groupby("Month")[["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]]
        .mean()
        .reset_index()
        .melt(id_vars="Month", var_name="Temperature", value_name="Value")
    )
    fig = px.line(
        temperature,
        x="Month",
        y="Value",
        color="Temperature",
        markers=True,
        title="Monthly temperature trends",
        color_discrete_sequence=["#0ea5e9", "#4f46e5", "#14b8a6", "#f59e0b"],
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title="Temperature (°C)")
    st.plotly_chart(style_plotly(fig, 410), use_container_width=True)

    st.markdown("### 📍 Location-wise rain probability")
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
        title="Top 15 locations by rain probability",
        color="RainProbability",
        color_continuous_scale=["#bfdbfe", "#2563eb", "#4f46e5"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.update_xaxes(title="Rain probability (%)")
    st.plotly_chart(style_plotly(fig, 520), use_container_width=True)


# ==================================================
# ABOUT PAGE
# ==================================================

def render_about():
    st.markdown('<div class="page-heading">About WeatherIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">The technology and data behind the weather intelligence experience.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <div class="section-kicker">Project overview</div>
            <div class="section-heading">AI-Powered Weather Analysis and Rain Prediction</div>
            <div class="section-copy">
                WeatherIQ is a machine-learning application that analyzes historical
                weather observations and estimates the likelihood of rainfall tomorrow.
                It combines a trained classification pipeline with interactive
                visualizations to make weather data easier to explore.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🤖 Machine learning")
        st.markdown(
            """
            - Logistic Regression
            - Random Forest Classifier
            - Selected model: **{}**
            - Selection metric: F1 Score
            - Prediction target: `RainTomorrow`
            """.format(model_info.get("best_model", "Unknown"))
        )
    with c2:
        st.markdown("### 🧰 Technologies")
        st.markdown(
            """
            - Python, Pandas and NumPy
            - Scikit-learn
            - Streamlit
            - Plotly
            - Joblib
            """
        )

    st.markdown("### 📂 Dataset")
    d1, d2, d3 = st.columns(3)
    with d1:
        metric_card("Original records", "145,460")
    with d2:
        metric_card("Original columns", "23")
    with d3:
        metric_card("Labeled records", f"{len(weather_df):,}")

    st.info(
        "This educational project uses the Australian Weather Dataset. "
        "Its model outputs are estimates, not official meteorological forecasts."
    )
    st.caption("Developed as part of the Codec Technologies AI Internship.")


# ==================================================
# DASHBOARD SHELL
# ==================================================

def render_dashboard():
    with st.sidebar:
        render_brand()
        st.caption("AI Weather Intelligence Platform")
        st.divider()

        selected_page = st.radio(
            "NAVIGATION",
            ["Rain Prediction", "Weather Analytics", "About Project"],
            index=[
                "Rain Prediction",
                "Weather Analytics",
                "About Project",
            ].index(st.session_state.dashboard_page),
        )
        st.session_state.dashboard_page = selected_page

        st.divider()
        st.markdown("### 🤖 Model information")
        st.write(f"**Selected model:** {model_info.get('best_model', 'Unknown')}")
        st.write("**Prediction target:** Rain Tomorrow")
        st.write("**Dataset:** WeatherAUS")
        st.caption("Predictions are estimates, not live forecasts.")

    top_left, top_right = st.columns([4, 1.3])
    with top_left:
        render_brand()
    with top_right:
        if st.button("← Back to Home", use_container_width=True):
            navigate("home")
            st.rerun()

    st.write("")
    if st.session_state.dashboard_page == "Rain Prediction":
        render_prediction()
    elif st.session_state.dashboard_page == "Weather Analytics":
        render_analytics()
    else:
        render_about()

    render_footer()


# ==================================================
# APP ROUTER
# ==================================================

if st.session_state.view == "home":
    render_home()
else:
    render_dashboard()
