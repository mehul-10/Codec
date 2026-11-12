# 🌦️ WeatherIQ – AI-Powered Weather Analysis & Rain Prediction

**WeatherIQ** is an intelligent weather analysis and rainfall prediction application built using Python, Machine Learning, Streamlit, and Plotly. It analyzes historical Australian weather data and predicts whether it will rain tomorrow based on atmospheric and environmental conditions.

The application combines machine learning predictions with interactive weather analytics through a modern, premium dark-themed dashboard.

---

## ✨ Features

### 🌧️ 1. AI Rain Prediction
- Predicts whether it will rain tomorrow.
- Uses a trained Machine Learning classification model.
- Supports multiple Australian weather locations.
- Accepts 24 weather input features.
- Displays Rain Probability and No Rain Probability.
- Interactive prediction probability visualization.
- Real-time prediction results based on user inputs.

### 📊 2. Weather Analytics
- Total historical weather records.
- Historical rain probability.
- Average maximum temperature.
- Average rainfall.
- Rain Tomorrow distribution visualization.
- Monthly rain probability analysis.
- Monthly temperature trends.
- Top 15 locations by rain probability.

### 🤖 3. Machine Learning
The project explores two classification algorithms:

- Logistic Regression
- Random Forest Classifier

The best-performing model is selected using the F1 Score evaluation metric.

### 🎨 4. Modern Dashboard UI
- Premium dark-themed interface.
- Gradient backgrounds and glass-style cards.
- Interactive sidebar navigation.
- Responsive layouts.
- Modern typography.
- Interactive Plotly charts.
- Organized weather prediction forms.

### ℹ️ 5. About Project
- Machine learning model information.
- Technology stack overview.
- Dataset statistics.
- Prediction target explanation.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Streamlit | Web application framework |
| Plotly | Interactive visualizations |
| Joblib | Model serialization |
| JSON | Model configuration |

---

## 📂 Project Structure

```text
WeatherIQ/
│
├── app.py
│
├── models/
│   ├── best_weather_model.pkl
│   └── model_info.json
│
├── data/
│   └── processed_weather.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset Information

**Dataset:** Australian Weather Dataset (WeatherAUS)

| Description | Details |
|---|---|
| Original Records | 145,460 |
| Original Columns | 23 |
| Labeled Records | 142,193 |
| Prediction Target | RainTomorrow |
| Target Classes | Yes / No |

The dataset contains historical Australian weather observations, including:

- Temperature
- Rainfall
- Evaporation
- Sunshine
- Humidity
- Atmospheric Pressure
- Wind Direction
- Wind Speed
- Cloud Coverage
- Location and Date Information

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd WeatherIQ
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```powershell
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a requirements file, install:

```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## 📦 Model Files

The application loads its trained model and configuration from:

```text
models/best_weather_model.pkl
models/model_info.json
```

The processed dataset is loaded from:

```text
data/processed_weather.csv
```

Make sure these files are available before launching the application.

---

## 🎯 Prediction Workflow

1. Select a weather location.
2. Enter current weather conditions.
3. Provide wind, humidity, pressure and cloud information.
4. Enter date information.
5. Click **Generate Rain Prediction**.
6. View the predicted weather condition.
7. Analyze rain and no-rain probabilities through interactive charts.

---

## 📈 Machine Learning Objective

The primary objective of WeatherIQ is to develop a classification system capable of estimating the probability of rainfall on the following day.

**Target Variable:**

```python
RainTomorrow
```

**Classification Classes:**

- `Yes` – Rain expected tomorrow.
- `No` – Rain not expected tomorrow.

The trained model uses historical weather patterns to generate predictions based on user-provided environmental conditions.

---

## 🔮 Future Enhancements

- Live weather API integration.
- Interactive geographical weather maps.
- Seven-day weather prediction dashboard.
- Advanced model comparison.
- Feature importance visualization.
- Historical weather trend forecasting.
- User authentication and personalized dashboards.
- Downloadable weather prediction reports.

---

## ⚠️ Disclaimer

WeatherIQ is an educational Machine Learning project. Its predictions are estimates based on historical weather observations and user-provided inputs. They should not replace official meteorological forecasts or weather warnings.

---

## 👨‍💻 Developer

**Mehul Gupta**

B.Tech Computer Science Engineering

### Connect With Me

- GitHub: [mehul-10](https://github.com/mehul-10)
- LinkedIn: [Mehul Gupta](https://www.linkedin.com/in/mehulgupta-developer/)

---

## 🙌 Acknowledgement

Developed as part of the **Codec Technologies AI Internship**.

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with ❤️ using Python, Machine Learning & Streamlit.**