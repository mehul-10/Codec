
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .data_preprocessing import load_data, preprocess_movies
except ImportError:
    from data_preprocessing import load_data, preprocess_movies


def build_recommendation_engine():

    # Load movie and rating datasets
    movies, ratings = load_data()

    # Load MovieLens-to-TMDB ID mappings
    links = pd.read_csv("data/links.csv")

    # Merge TMDB IDs into movie dataset
    movies = movies.merge(
        links[["movieId", "tmdbId"]],
        on="movieId",
        how="left"
    )

    # Preprocess movie data
    movies = preprocess_movies(movies)

    # Convert genres into numerical vectors
    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(
        movies["combined_features"]
    )

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    print("Movies loaded:", len(movies))
    print("TMDB IDs available:", movies["tmdbId"].notna().sum())

    return movies, similarity_matrix


def recommend_movies(
    movie_title,
    movies,
    similarity_matrix,
    top_n=10
):

    # Find selected movie
    movie_indices = movies[
        movies["title"].str.lower() == movie_title.lower()
    ].index

    if len(movie_indices) == 0:
        return None

    movie_index = movie_indices[0]

    # Calculate similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    # Sort by highest similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Exclude selected movie
    similarity_scores = similarity_scores[1:top_n + 1]

    recommendations = []

    for index, score in similarity_scores:

        movie = movies.iloc[index]

        tmdb_id = (
            int(movie["tmdbId"])
            if pd.notna(movie["tmdbId"])
            else None
        )

        recommendations.append({

            "movieId": int(movie["movieId"]),

            "tmdbId": tmdb_id,

            "title": movie["title"],

            "genres": movie["genres"],

            "year": movie["year"],

            "similarity_score": round(
                float(score) * 100, 2
            )
        })

    return pd.DataFrame(recommendations)


if __name__ == "__main__":

    movies, similarity_matrix = build_recommendation_engine()

    print("\nRecommendation engine built successfully!")

    movie_name = "Toy Story"

    recommendations = recommend_movies(
        movie_name,
        movies,
        similarity_matrix
    )

    print(f"\nMovies similar to {movie_name}:\n")

    print(recommendations.to_string(index=False))
