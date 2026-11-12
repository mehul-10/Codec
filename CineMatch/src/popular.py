
from anyio import Path
import pandas as pd
from pathlib import Path

try:
    from .data_preprocessing import load_data, preprocess_movies
except ImportError:
    from data_preprocessing import load_data, preprocess_movies


def get_popular_movies(movies, ratings, top_n=20, min_ratings=50):

    # Calculate average rating and number of ratings per movie
    movie_stats = ratings.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    # Calculate the overall average rating
    C = ratings["rating"].mean()

    # Minimum number of ratings required
    m = min_ratings

    # Apply weighted rating formula
    movie_stats["weighted_rating"] = (
        (movie_stats["rating_count"] /
         (movie_stats["rating_count"] + m))
        * movie_stats["average_rating"]
        +
        (m /
         (movie_stats["rating_count"] + m))
        * C
    )

    # Keep movies with sufficient ratings
    movie_stats = movie_stats[
        movie_stats["rating_count"] >= min_ratings
    ]

    # Merge rating statistics with movie information
    popular = movies.merge(
        movie_stats,
        on="movieId",
        how="inner"
    )

    # Sort by weighted rating
    popular = popular.sort_values(
        by=["weighted_rating", "rating_count"],
        ascending=False
    )

    # Select required columns
    result = popular.head(top_n).copy()

    # Convert TMDB IDs safely
    result["tmdbId"] = pd.to_numeric(
        result["tmdbId"], errors="coerce"
    ).astype("Int64")

    # Return a clean result
    return result[
        [
            "movieId",
            "tmdbId",
            "title",
            "genres",
            "year",
            "average_rating",
            "rating_count",
            "weighted_rating"
        ]
    ].reset_index(drop=True)


if __name__ == "__main__":

    movies, ratings = load_data()

    # Load TMDB mappings
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    links = pd.read_csv(DATA_DIR / "links.csv")

    movies = movies.merge(
        links[["movieId", "tmdbId"]],
        on="movieId",
        how="left"
    )

    movies = preprocess_movies(movies)

    popular_movies = get_popular_movies(
        movies,
        ratings,
        top_n=20,
        min_ratings=50
    )

    print("\nTop Rated Movies:\n")

    print(
        popular_movies[
            [
                "title",
                "average_rating",
                "rating_count",
                "weighted_rating"
            ]
        ].to_string(index=False)
    )
