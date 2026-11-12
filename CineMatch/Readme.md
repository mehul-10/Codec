# 🎬 CineMatch – AI Movie Recommendation System

CineMatch is an AI-powered movie recommendation web application built with Python and Streamlit. It uses TF-IDF vectorization and cosine similarity to recommend movies based on movie genres and content similarity.

The application also integrates The Movie Database (TMDB) API to retrieve movie posters and detailed movie information.

## ✨ Features

- **AI Movie Recommendations:** Find similar movies using TF-IDF and cosine similarity.
- **Movie Posters:** Display movie posters using TMDB.
- **Movie Details:** Explore movie overviews, release dates, ratings, directors, and cast.
- **Top Rated Movies:** Discover highly rated movies using a weighted rating formula.
- **Genre and Year Filters:** Filter movies based on genres and release years.
- **Watchlist:** Save movies to watch later.
- **Favourites:** Maintain a personal collection of favourite movies.
- **Analytics Dashboard:** Explore movie datasets through interactive Plotly charts.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- Plotly
- TMDB API

## 📂 Project Structure

```text
CineMatch/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   ├── links.csv
│   └── watchlist.json
│
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── recommendation.py
    ├── tmdb.py
    ├── watchlist.py
    ├── popular.py
    └── analytics.py
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CineMatch
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure TMDB API

Create a folder named `.streamlit` in the project root and create a file named `secrets.toml` inside it.

Add your TMDB API key:

```toml
TMDB_API_KEY = "YOUR_TMDB_API_KEY"
```

Never upload your API key or `secrets.toml` to GitHub.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🧠 How the Recommendation System Works

1. Loads movie information and ratings from the MovieLens dataset.
2. Extracts movie titles, release years, and genres.
3. Combines movie genres into content features.
4. Converts textual features into numerical vectors using TF-IDF.
5. Calculates cosine similarity between movie vectors.
6. Recommends movies with the highest similarity scores.

## 📊 Dataset

CineMatch uses the MovieLens Latest Small dataset, containing:

- 9,742 movies
- 100,836 ratings
- Movie-to-TMDB ID mappings

## 🔐 API Integration

CineMatch uses the TMDB API to retrieve movie posters and additional movie information.

Movie data and images are provided by TMDB. This project is not endorsed or certified by TMDB.

## 👨‍💻 Developer

**Mehul Gupta**

AI and Machine Learning Enthusiast | B.Tech Computer Science Engineering Student

## 📌 Project Status

Core development completed. Testing, documentation, and deployment preparation are in progress.
