
import json
from pathlib import Path

# Persistent storage location
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"


def load_saved_movies():
    """Load Watchlist and Favourites from JSON."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not WATCHLIST_FILE.exists():
        return {
            "watchlist": [],
            "favourites": []
        }

    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            "watchlist": data.get("watchlist", []),
            "favourites": data.get("favourites", [])
        }

    except (json.JSONDecodeError, OSError):
        return {
            "watchlist": [],
            "favourites": []
        }


def save_movies(data):
    """Save collections to JSON."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(WATCHLIST_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def add_movie(movie, collection="watchlist"):
    """Add a movie to Watchlist or Favourites."""

    if collection not in ("watchlist", "favourites"):
        return False

    data = load_saved_movies()

    movie_id = str(movie.get("movieId"))

    if any(
        str(item.get("movieId")) == movie_id
        for item in data[collection]
    ):
        return False

    data[collection].append(movie)
    save_movies(data)

    return True


def remove_movie(movie_id, collection="watchlist"):
    """Remove a movie from a collection."""

    if collection not in ("watchlist", "favourites"):
        return False

    data = load_saved_movies()

    original_length = len(data[collection])

    data[collection] = [
        movie for movie in data[collection]
        if str(movie.get("movieId")) != str(movie_id)
    ]

    if len(data[collection]) == original_length:
        return False

    save_movies(data)

    return True
