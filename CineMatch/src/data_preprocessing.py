
import pandas as pd
from pathlib import Path

# Dataset paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"


def load_data():
    """Load movie and rating datasets."""

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)

    return movies, ratings


def preprocess_movies(movies):
    """Clean and prepare movie information."""

    movies = movies.copy()

    # Handle missing values
    movies["genres"] = movies["genres"].fillna("")

    # Extract release year from title
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")

    # Remove year from title
    movies["title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)$", "", regex=True
    )

    # Create combined features
    movies["combined_features"] = (
        movies["genres"].str.replace("|", " ", regex=False)
    )

    return movies


if __name__ == "__main__":

    movies, ratings = load_data()
    movies = preprocess_movies(movies)

    print("Movies Dataset:", movies.shape)
    print("Ratings Dataset:", ratings.shape)

    print("\nFirst 5 Movies:")
    print(movies.head())

    print("\nMissing Values:")
    print(movies.isnull().sum())
