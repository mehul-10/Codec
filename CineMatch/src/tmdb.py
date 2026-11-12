
import requests
import streamlit as st

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"


# =========================================================
# REUSABLE TMDB SESSION WITH AUTOMATIC RETRIES
# =========================================================

@st.cache_resource
def get_tmdb_session():

    session = requests.Session()

    retry_strategy = Retry(
        total=4,
        connect=4,
        read=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "accept": "application/json",
        "User-Agent": "CineMatch/1.0"
    })

    return session


# =========================================================
# API KEY
# =========================================================

def get_api_key():

    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        print("TMDB API key not configured.")
        return None


# =========================================================
# GET MOVIE POSTER
# =========================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_poster(movie_title, year=None, tmdb_id=None):

    api_key = get_api_key()

    if not api_key:
        return None

    session = get_tmdb_session()

    headers = {
        "accept": "application/json"
    }

    # Convert TMDB ID safely
    try:
        if tmdb_id is not None:
            tmdb_id = int(float(tmdb_id))
        else:
            tmdb_id = None
    except (ValueError, TypeError, OverflowError):
        tmdb_id = None

    # -----------------------------------------------------
    # METHOD 1: DIRECT TMDB ID LOOKUP
    # -----------------------------------------------------

    if tmdb_id is not None:

        try:

            movie_url = f"{BASE_URL}/movie/{tmdb_id}"

            response = session.get(
                movie_url,
                params={"api_key": api_key},
                headers=headers,
                timeout=(10, 30)
            )

            if response.status_code == 200:

                movie_data = response.json()

                poster_path = movie_data.get("poster_path")

                if poster_path:
                    return IMAGE_URL + poster_path

                # -------------------------------------------------
                # METHOD 2: ALTERNATIVE POSTERS
                # -------------------------------------------------

                images_url = f"{BASE_URL}/movie/{tmdb_id}/images"

                images_response = session.get(
                    images_url,
                    params={
                        "api_key": api_key,
                        "include_image_language": "en,null"
                    },
                    headers=headers,
                    timeout=(10, 30)
                )

                if images_response.status_code == 200:

                    posters = images_response.json().get(
                        "posters", []
                    )

                    if posters:

                        english_posters = [
                            poster for poster in posters
                            if poster.get("iso_639_1") == "en"
                        ]

                        neutral_posters = [
                            poster for poster in posters
                            if poster.get("iso_639_1") is None
                        ]

                        selected = (
                            english_posters[0]
                            if english_posters
                            else neutral_posters[0]
                            if neutral_posters
                            else posters[0]
                        )

                        if selected.get("file_path"):
                            return IMAGE_URL + selected["file_path"]

        except (
            requests.exceptions.RequestException,
            ValueError,
            TypeError
        ) as error:

            print(f"TMDB ID lookup error: {error}")

    # -----------------------------------------------------
    # METHOD 3: TITLE-BASED SEARCH FALLBACK
    # -----------------------------------------------------

    try:

        search_url = f"{BASE_URL}/search/movie"

        params = {
            "api_key": api_key,
            "query": movie_title,
            "include_adult": "false"
        }

        response = session.get(
            search_url,
            params=params,
            headers=headers,
            timeout=(10, 30)
        )

        response.raise_for_status()

        results = response.json().get("results", [])

        candidates = [
            movie for movie in results
            if movie.get("poster_path")
        ]

        if not candidates:
            return None

        # Prefer matching release year
        if year is not None:

            year_matches = [
                movie for movie in candidates
                if movie.get("release_date", "").startswith(str(year))
            ]

            if year_matches:
                candidates = year_matches

        # Prefer exact title match
        exact_matches = [
            movie for movie in candidates
            if movie.get("title", "").lower().strip()
            == movie_title.lower().strip()
        ]

        selected = (
            exact_matches[0]
            if exact_matches
            else candidates[0]
        )

        return IMAGE_URL + selected["poster_path"]

    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError
    ) as error:

        print(f"TMDB search error for {movie_title}: {error}")

        return None


# =========================================================
# GET MOVIE DETAILS
# =========================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_movie_details(tmdb_id):

    if tmdb_id is None:
        return None

    api_key = get_api_key()

    if not api_key:
        return None

    try:

        tmdb_id = int(float(tmdb_id))

        session = get_tmdb_session()

        # -------------------------------------------------
        # FETCH BASIC MOVIE INFORMATION
        # -------------------------------------------------

        movie_url = f"{BASE_URL}/movie/{tmdb_id}"

        movie_response = session.get(
            movie_url,
            params={"api_key": api_key},
            timeout=(10, 30)
        )

        if movie_response.status_code != 200:

            print(
                "Movie API Error:",
                movie_response.status_code
            )

            return None

        movie_data = movie_response.json()

        # Default values
        director = "Not available"
        cast = []

        # -------------------------------------------------
        # FETCH CAST AND CREW
        # -------------------------------------------------

        credits_url = f"{BASE_URL}/movie/{tmdb_id}/credits"

        try:

            credits_response = session.get(
                credits_url,
                params={"api_key": api_key},
                timeout=(10, 30)
            )

            if credits_response.status_code == 200:

                credits_data = credits_response.json()

                director = next(
                    (
                        person.get("name", "Unknown")
                        for person in credits_data.get("crew", [])
                        if person.get("job") == "Director"
                    ),
                    "Not available"
                )

                cast = [
                    person.get("name", "Unknown")
                    for person in credits_data.get("cast", [])[:5]
                ]

            else:

                print(
                    "Credits API Error:",
                    credits_response.status_code
                )

        except requests.exceptions.RequestException as error:

            print("Credits Request Error:", error)

        # -------------------------------------------------
        # RETURN MOVIE DETAILS
        # -------------------------------------------------

        return {
            "title": movie_data.get("title", "Unknown"),

            "overview": movie_data.get(
                "overview",
                "No overview available."
            ),

            "release_date": movie_data.get(
                "release_date",
                "Not available"
            ),

            "runtime": movie_data.get("runtime"),

            "rating": movie_data.get("vote_average", 0),

            "genres": [
                genre.get("name", "Unknown")
                for genre in movie_data.get("genres", [])
            ],

            "director": director,

            "cast": cast,

            "language": movie_data.get(
                "original_language",
                "Unknown"
            ).upper(),

            "status": movie_data.get(
                "status",
                "Unknown"
            )
        }

    except requests.exceptions.RequestException as error:

        print("Movie Details Request Error:", error)
        return None

    except (
        ValueError,
        KeyError,
        TypeError,
        OverflowError
    ) as error:

        print("Movie Details Data Error:", error)
        return None

    except Exception as error:

        print("Unexpected Movie Details Error:", error)
        return None
