import html

import pandas as pd
import streamlit as st

from src.tmdb import get_movie_poster, get_movie_details
from src.recommendation import (
    build_recommendation_engine,
    recommend_movies,
)

from src.popular import get_popular_movies
from src.data_preprocessing import load_data

from src.watchlist import (
    load_saved_movies,
    add_movie,
    remove_movie,
)
from src.analytics import render_analytics_dashboard


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CineMatch | AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "recommendations": None,
    "selected_detail_id": None,
    "selected_detail_title": None,
    "selected_movie": None,
    "open_details": False,
    # Bumped on every new search so filter widgets reset instead of
    # keeping values (e.g. a year range) that no longer exist.
    "rec_version": 0,
}

for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# STREAMLIT VERSION COMPATIBILITY
# `use_container_width` is deprecated in newer Streamlit
# releases in favour of `width="stretch"`.
# =========================================================

def _streamlit_version():
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        return (major, minor)
    except Exception:
        return (0, 0)


STRETCH = (
    {"width": "stretch"}
    if _streamlit_version() >= (1, 50)
    else {"use_container_width": True}
)

# st.dialog (1.37+) lets movie details open in a pop-up, so they are
# visible no matter how far down the page the button was clicked.
HAS_DIALOG = hasattr(st, "dialog")


# =========================================================
# CUSTOM CSS
# Light, editorial look: big bold type, wide-tracked caps,
# black pill buttons and a single yellow accent.
# (Pair with .streamlit/config.toml for a fully light theme.)
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #fafaf9;
    --surface: #ffffff;
    --soft: #f5f5f4;
    --border: #e7e5e4;
    --fg: #0c0a09;
    --muted: #78716c;
    --accent: #facc15;
    --accent-soft: rgba(250, 204, 21, 0.22);
    color-scheme: light;
}

html, body, .stApp, button, input, select, textarea {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--fg);
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* ---------- Typography ---------- */
h1, h2, h3, h4 {
    color: var(--fg) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--fg);
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

label, label p,
[data-testid="stWidgetLabel"] p {
    color: var(--fg) !important;
    font-size: 13px;
    font-weight: 500;
}

hr {
    border-color: var(--border);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] h2 {
    font-size: 22px;
    font-weight: 800 !important;
}

/* ---------- Hero ---------- */
@keyframes rise {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: none; }
}

@keyframes pop {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: none; }
}

.hero {
    padding: 4px 0 8px 0;
}

.hero-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 8px;
    animation: rise 0.5s ease both;
}

.hero-logo {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: var(--fg);
}

.hero-links {
    display: flex;
    gap: 32px;
}

.hero-links a {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.18em;
    color: var(--muted) !important;
    text-decoration: none !important;
    transition: color 0.15s ease;
}

.hero-links a:hover {
    color: var(--fg) !important;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1fr 1.1fr 1fr;
    align-items: center;
    gap: 24px;
    min-height: 400px;
}

.hero-left {
    animation: rise 0.6s ease 0.5s both;
}

.hero-tag {
    color: var(--fg);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.hero-description {
    font-size: 15px;
    line-height: 1.7;
    color: var(--muted);
    max-width: 300px;
}

.hero-center {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 340px;
}

.hero-circle {
    position: absolute;
    width: min(360px, 100%);
    aspect-ratio: 1 / 1;
    border-radius: 50%;
    background: var(--accent);
    animation: pop 0.8s cubic-bezier(0.22, 1, 0.36, 1) 0.1s both;
}

.hero-art {
    position: relative;
    width: clamp(190px, 21vw, 270px);
    height: auto;
    animation: rise 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.3s both;
}

.hero-art .clap-top {
    transform-box: view-box;
    transform-origin: 38px 84px;
    transform: rotate(-16deg);
    animation: clap 1.1s ease 1s both;
}

@keyframes clap {
    0%   { transform: rotate(-16deg); }
    45%  { transform: rotate(0deg); }
    100% { transform: rotate(-16deg); }
}

.hero-right {
    animation: rise 0.6s ease 0.7s both;
}

.hero-title {
    font-size: clamp(56px, 7.5vw, 108px);
    font-weight: 800;
    line-height: 0.95;
    letter-spacing: -0.04em;
    color: var(--fg);
}

/* ---------- Stats strip ---------- */
.stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    margin: 20px 0 36px 0;
}

.stat {
    padding: 22px 0 22px 28px;
    border-left: 1px solid var(--border);
}

.stat:first-child {
    padding-left: 0;
    border-left: none;
}

.stat-value {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--fg);
}

.stat-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 4px;
}

/* ---------- Section headings ---------- */
.section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--muted);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.section-label::before {
    content: "";
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
}

.section-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--fg);
    margin-bottom: 14px;
}

/* ---------- Movie cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border-color: var(--border);
    border-radius: 16px;
    transition: border-color 0.15s ease;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #d6d3d1;
}

[data-testid="stImage"] img {
    border-radius: 10px;
}

.movie-title {
    color: var(--fg);
    font-size: 15px;
    font-weight: 700;
    line-height: 1.35;
    letter-spacing: -0.01em;
    margin-top: 10px;
    min-height: 42px;
}

.movie-meta {
    color: var(--muted);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}

.score-text {
    color: var(--fg);
    font-size: 13px;
    font-weight: 700;
    margin: 4px 0;
}

.poster-placeholder {
    height: 230px;
    background: var(--soft);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 13px;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: transparent;
    color: var(--fg);
    border: 1px solid var(--fg);
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 16px;
    transition: background 0.15s ease, color 0.15s ease;
}

.stButton > button:hover,
.stButton > button:focus-visible {
    background: var(--fg);
    color: #ffffff;
    border-color: var(--fg);
}

.stButton > button p {
    color: inherit !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background: var(--fg);
    color: #ffffff;
    padding: 9px 16px;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--accent);
    color: var(--fg);
    border-color: var(--accent);
}

/* ---------- Inputs ---------- */
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: var(--surface);
    color: var(--fg);
    border-color: var(--border);
    border-radius: 10px;
}

span[data-baseweb="tag"] {
    background: var(--accent) !important;
    color: var(--fg) !important;
    border-radius: 999px;
}

span[data-baseweb="tag"] span {
    color: var(--fg) !important;
}

/* ---------- Progress, metrics, tabs ---------- */
div[data-testid="stProgress"] > div > div {
    background-color: #e7e5e4;
}

div[data-testid="stProgress"] > div > div > div,
div[role="progressbar"] > div {
    background-color: var(--accent);
}

div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 14px 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] p {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--fg);
    font-weight: 800;
    letter-spacing: -0.02em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] p {
    font-weight: 600;
    color: var(--muted);
}

.stTabs [aria-selected="true"] p {
    color: var(--fg);
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent);
    height: 3px;
}

div[role="dialog"] {
    background: var(--bg);
    border-radius: 18px;
}

/* ---------- Empty state ---------- */
.empty-state {
    position: relative;
    text-align: center;
    padding: 56px 20px;
    border: 1px dashed #d6d3d1;
    border-radius: 18px;
}

.empty-circle {
    width: 96px;
    height: 96px;
    margin: 0 auto 8px auto;
    border-radius: 50%;
    background: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 42px;
}

.empty-state h3 {
    margin: 14px 0 6px 0;
    font-size: 26px;
    font-weight: 800 !important;
}

.empty-state p {
    margin: 0 auto;
    max-width: 380px;
    color: var(--muted) !important;
}

/* ---------- Footer ---------- */
.footer {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 12px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.7;
    padding: 24px 0 8px 0;
    border-top: 1px solid var(--border);
    margin-top: 48px;
}

.footer a {
    color: var(--fg) !important;
    font-weight: 600;
    text-decoration: underline;
}

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .hero-links { display: none; }
    .hero-grid {
        grid-template-columns: 1fr;
        text-align: center;
        min-height: 0;
    }
    .hero-center { order: -1; min-height: 280px; }
    .hero-description { margin: 0 auto; }
    .stats { grid-template-columns: 1fr; }
    .stat, .stat:first-child {
        padding: 16px 0;
        border-left: none;
        border-top: 1px solid var(--border);
    }
    .stat:first-child { border-top: none; }
    .section-title { font-size: 28px; }
}

@media (prefers-reduced-motion: reduce) {
    .hero *, .hero { animation: none !important; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# LOAD RECOMMENDATION ENGINE
# =========================================================

@st.cache_resource
def load_engine():
    return build_recommendation_engine()


try:
    with st.spinner("Preparing your movie discovery engine..."):
        movies, similarity_matrix = load_engine()

except Exception as error:
    st.error(f"Unable to load the recommendation engine: {error}")
    st.stop()


# =========================================================
# TOP RATED MOVIES DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_rating_data():
    _, ratings = load_data()
    return ratings


# Underscore-prefixed arguments are not hashed by Streamlit, which avoids
# hashing two large DataFrames on every rerun.
@st.cache_data(show_spinner=False)
def load_popular_data(_movie_data, _rating_data):
    return get_popular_movies(
        _movie_data,
        _rating_data,
        top_n=20,
        min_ratings=50,
    )


ratings_data = None
popular_movies = pd.DataFrame()

try:
    with st.spinner("Discovering top rated movies..."):
        ratings_data = load_rating_data()
        popular_movies = load_popular_data(movies, ratings_data)

except Exception as error:
    st.error(f"Unable to load top rated movies: {error}")
    popular_movies = pd.DataFrame()


# =========================================================
# CACHED TMDB CALLS
# Without caching, every button click re-fetches every poster.
# =========================================================

@st.cache_data(show_spinner=False, ttl=3600)
def cached_poster(title, year, tmdb_id):
    return get_movie_poster(title, year, tmdb_id)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_details(tmdb_id):
    return get_movie_details(tmdb_id)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_value(value, default="Not available"):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return value


def safe_tmdb_id(value):
    value = safe_value(value, None)

    if value is None:
        return None

    try:
        return int(float(value))
    except (ValueError, TypeError, OverflowError):
        return None


def safe_year(value):
    """Return the year as an int (1995.0 -> 1995) or None.

    Also copes with saved values such as "None" or "".
    """
    value = safe_value(value, None)

    if value is None:
        return None

    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def safe_percent(value):
    """Return a similarity score as a 0-100 percentage.

    Scores in the 0-1 range (raw cosine similarity) are scaled up.
    """
    try:
        score = float(safe_value(value, 0))
    except (ValueError, TypeError):
        return 0.0

    if 0 <= score <= 1:
        score *= 100

    return min(max(score, 0.0), 100.0)


def clean_genres(value, default="Genres unavailable"):
    """Turn 'Action|Adventure' or a list into 'Action · Adventure'."""
    value = safe_value(value, None)

    if value is None or str(value).strip() in ("", "None"):
        return default

    if isinstance(value, (list, tuple, set)):
        return " · ".join(str(g) for g in value)

    return str(value).replace("|", " · ")


def esc(value):
    """Escape text before inserting it into raw HTML."""
    return html.escape(str(value))


def collect_genres(genre_series):
    """Unique, sorted genres from a column of 'A|B|C' strings."""
    return sorted({
        genre.strip()
        for genres in genre_series.dropna()
        for genre in str(genres).split("|")
        if genre.strip() and genre.strip().lower() != "(no genres listed)"
    })


def has_any_genre(genres, selected):
    return any(g.strip() in selected for g in str(genres).split("|"))


def sort_by_year(df, ascending):
    """Sort by release year; the year column may be text, so coerce first."""
    if "year" not in df.columns:
        return df

    return (
        df.assign(_sort_year=pd.to_numeric(df["year"], errors="coerce"))
        .sort_values("_sort_year", ascending=ascending, na_position="last")
        .drop(columns=["_sort_year"])
    )


def normalize_movie(row):
    """Convert a recommendation row or a saved-movie dict into one shape."""
    tmdb_id = safe_tmdb_id(row.get("tmdbId"))
    title = str(safe_value(row.get("title"), "Unknown Movie"))

    raw_id = safe_value(row.get("movieId"), None)
    if isinstance(raw_id, float) and raw_id.is_integer():
        raw_id = int(raw_id)
    movie_id = str(raw_id) if raw_id is not None else str(tmdb_id or title)

    try:
        raw_score = float(safe_value(row.get("similarity_score"), 0))
    except (ValueError, TypeError):
        raw_score = 0.0

    return {
        "movieId": movie_id,
        "tmdbId": tmdb_id,
        "title": title,
        "year": safe_year(row.get("year")),
        "genres": clean_genres(row.get("genres")),
        "raw_score": raw_score,
        "score": safe_percent(raw_score),
        "average_rating": safe_value(row.get("average_rating"), 0),
        "rating_count": safe_value(row.get("rating_count"), 0),
        "weighted_rating": safe_value(row.get("weighted_rating"), 0),
    }


def show_poster(poster_url):
    if poster_url:
        st.image(poster_url, **STRETCH)
    else:
        st.markdown(
            '<div class="poster-placeholder">Poster unavailable</div>',
            unsafe_allow_html=True,
        )


def open_details(tmdb_id, title):
    st.session_state.selected_detail_id = tmdb_id
    st.session_state.selected_detail_title = title
    st.session_state.open_details = True
    st.rerun()


def close_details():
    st.session_state.selected_detail_id = None
    st.session_state.selected_detail_title = None
    st.session_state.open_details = False
    st.rerun()


# =========================================================
# MOVIE DETAILS
# =========================================================

def render_movie_details():
    tmdb_id = st.session_state.selected_detail_id
    movie_title = st.session_state.selected_detail_title

    st.markdown(
        '<div class="section-label">Explore the movie</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-title">Movie Details</div>',
        unsafe_allow_html=True,
    )

    details = None

    if tmdb_id is None:
        st.warning("TMDB information is unavailable for this movie.")

    else:
        with st.spinner("Loading movie information..."):
            try:
                details = cached_details(tmdb_id)
            except Exception as error:
                st.error(f"Unable to load movie details: {error}")

        if not details:
            st.warning("Movie details could not be loaded. Please try again.")

    if details:
        detail_col1, detail_col2 = st.columns([1, 2], gap="large")

        with detail_col1:
            try:
                show_poster(cached_poster(movie_title, None, tmdb_id))
            except Exception:
                st.info("Poster unavailable.")

        with detail_col2:
            language = details.get("language") or "Unknown"

            st.markdown(f"### {details.get('title') or movie_title}")

            st.caption(
                f"{details.get('release_date') or 'Not available'} "
                f"· {language} "
                f"· {details.get('status') or 'Unknown'}"
            )

            st.write(details.get("overview") or "No overview available.")

            st.write("")

            info1, info2, info3 = st.columns(3)

            with info1:
                try:
                    rating_text = f"{float(details.get('rating') or 0):.1f}/10"
                except (ValueError, TypeError):
                    rating_text = "N/A"

                st.metric("TMDB Rating", rating_text)

            with info2:
                runtime = details.get("runtime")
                st.metric("Runtime", f"{runtime} min" if runtime else "N/A")

            with info3:
                st.metric("Language", language)

            st.markdown("**Genres**")
            genres = details.get("genres") or []
            st.write(" · ".join(genres) if genres else "Not available")

            st.markdown("**Director**")
            st.write(details.get("director") or "Not available")

            st.markdown("**Cast**")
            cast = details.get("cast") or []
            st.write(", ".join(cast) if cast else "Not available")

    st.write("")

    if st.button("Close Movie Details", key="close_movie_details"):
        close_details()


if HAS_DIALOG:

    @st.dialog("Movie Details", width="large")
    def details_dialog():
        render_movie_details()


# =========================================================
# MOVIE CARD (shared by recommendations, top rated and collections)
# =========================================================

def render_movie_card(
    movie,
    key_prefix,
    position,
    collection=None,
    show_rating=False,
):
    """Draw one movie card.

    collection=None  -> recommendation or top-rated card
    collection="..." -> saved movie card with Remove button
    """

    key = f"{key_prefix}_{position}_{movie['movieId']}"

    with st.container(border=True):

        # Movie poster
        try:
            show_poster(
                cached_poster(
                    movie["title"],
                    movie["year"],
                    movie["tmdbId"],
                )
            )
        except Exception:
            show_poster(None)

        # Movie title
        st.markdown(
            f'<div class="movie-title">{esc(movie["title"])}</div>',
            unsafe_allow_html=True,
        )

        # Release year
        st.markdown(
            f'<div class="movie-meta">'
            f'{movie["year"] if movie["year"] else "Year unavailable"}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Genres
        st.caption(movie["genres"])

        # Rating or recommendation match score
        if collection is None:

            if show_rating:

                try:
                    rating = float(movie.get("average_rating", 0) or 0)
                except (ValueError, TypeError):
                    rating = 0.0

                try:
                    rating_count = int(movie.get("rating_count", 0) or 0)
                except (ValueError, TypeError):
                    rating_count = 0

                st.markdown(
                    f'<div class="score-text">'
                    f'⭐ {rating:.2f}/5 · {rating_count:,} ratings'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            else:

                score = safe_percent(
                    movie.get("raw_score", movie.get("score", 0))
                )

                st.markdown(
                    f'<div class="score-text">{score:.2f}% Match</div>',
                    unsafe_allow_html=True,
                )

                st.progress(score / 100)

        # View Details button
        if st.button("View Details", key=f"details_{key}", **STRETCH):
            open_details(movie["tmdbId"], movie["title"])

        # Watchlist and Favourite buttons
        if collection is None:

            movie_to_save = {
                "movieId": movie["movieId"],
                "tmdbId": movie["tmdbId"],
                "title": movie["title"],
                "year": str(movie["year"]) if movie["year"] else "",
                "genres": clean_genres(movie["genres"], default=""),
                "similarity_score": movie.get(
                    "raw_score",
                    movie.get("score", 0),
                ),
            }

            watch_col, fav_col = st.columns(2, gap="small")

            with watch_col:
                if st.button(
                    "＋ Watchlist",
                    key=f"watchlist_{key}",
                    **STRETCH,
                ):
                    if add_movie(movie_to_save, "watchlist"):
                        st.toast(f"Added {movie['title']} to Watchlist!")
                    else:
                        st.toast("Already in Watchlist!")

            with fav_col:
                if st.button(
                    "♥ Favourite",
                    key=f"favourite_{key}",
                    **STRETCH,
                ):
                    if add_movie(movie_to_save, "favourites"):
                        st.toast(f"Added {movie['title']} to Favourites!")
                    else:
                        st.toast("Already in Favourites!")

        # Remove button for saved movies
        else:
            if st.button("Remove", key=f"remove_{key}", **STRETCH):
                remove_movie(movie["movieId"], collection)
                st.toast(f"Removed {movie['title']}")
                st.rerun()


def render_movie_grid(
    rows,
    key_prefix,
    collection=None,
    show_rating=False,
):
    """Lay movies out in rows of four."""

    for start in range(0, len(rows), 4):

        columns = st.columns(4, gap="medium")

        for offset, (column, row) in enumerate(
            zip(columns, rows[start:start + 4])
        ):
            with column:
                render_movie_card(
                    normalize_movie(row),
                    key_prefix,
                    start + offset,
                    collection,
                    show_rating,
                )


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
<div class="hero">
<div class="hero-nav">
<div class="hero-logo">cinematch.</div>
<div class="hero-links">
<a href="#recommendations">DISCOVER</a>
<a href="#top-rated">TOP RATED</a>
<a href="#analytics">ANALYTICS</a>
<a href="#collection">COLLECTION</a>
</div>
</div>
<div class="hero-grid">
<div class="hero-left">
<div class="hero-tag">AI powered movie discovery</div>
<div class="hero-description">
Discover your next favourite movie.
Intelligent recommendations powered by
Machine Learning and TMDB.
</div>
</div>
<div class="hero-center">
<div class="hero-circle"></div>
<svg class="hero-art" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Movie clapperboard">
<rect x="38" y="84" width="124" height="84" rx="10" fill="#0c0a09"/>
<rect x="38" y="84" width="124" height="14" fill="#fafaf9"/>
<polygon points="52,98 62,84 76,84 66,98" fill="#0c0a09"/>
<polygon points="80,98 90,84 104,84 94,98" fill="#0c0a09"/>
<polygon points="108,98 118,84 132,84 122,98" fill="#0c0a09"/>
<polygon points="136,98 146,84 160,84 150,98" fill="#0c0a09"/>
<polygon points="94,112 94,148 126,130" fill="#facc15"/>
<g class="clap-top">
<rect x="38" y="62" width="124" height="18" rx="4" fill="#fafaf9"/>
<polygon points="52,80 62,62 76,62 66,80" fill="#0c0a09"/>
<polygon points="80,80 90,62 104,62 94,80" fill="#0c0a09"/>
<polygon points="108,80 118,62 132,62 122,80" fill="#0c0a09"/>
<polygon points="136,80 146,62 160,62 150,80" fill="#0c0a09"/>
</g>
<circle cx="44" cy="82" r="4" fill="#facc15"/>
</svg>
</div>
<div class="hero-right">
<div class="hero-title">Cine<br>Match.</div>
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# PROJECT METRICS
# =========================================================

st.markdown(
    f"""
<div class="stats">
    <div class="stat">
        <div class="stat-value">{len(movies):,}</div>
        <div class="stat-label">Movies Available</div>
    </div>
    <div class="stat">
        <div class="stat-value">AI / ML</div>
        <div class="stat-label">Recommendation Engine</div>
    </div>
    <div class="stat">
        <div class="stat-value">TMDB</div>
        <div class="stat-label">Movie Information</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## CineMatch")
    st.caption("Your personal AI movie assistant")

    st.divider()

    st.markdown("### Find Your Next Movie")

    movie_options = sorted(movies["title"].dropna().unique().tolist())

    selected_movie = st.selectbox(
        "Select a movie you already like",
        movie_options,
        index=None,
        placeholder="Search for a movie...",
    )

    number_of_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
    )

    recommend_button = st.button("Recommend Movies", **STRETCH)

    st.divider()

    st.markdown("### About CineMatch")

    st.caption(
        "CineMatch uses TF-IDF vectorization and "
        "cosine similarity to discover movies "
        "with similar genres and content."
    )

    st.caption("Built with Python, Scikit-learn and Streamlit.")


# =========================================================
# GENERATE RECOMMENDATIONS
# =========================================================

if recommend_button:

    if not selected_movie:
        st.warning("Please select a movie first.")

    else:
        with st.spinner("Finding movies you'll love..."):
            try:
                result = recommend_movies(
                    selected_movie,
                    movies,
                    similarity_matrix,
                    top_n=number_of_recommendations,
                )

                if result is None or result.empty:
                    st.session_state.recommendations = None
                    st.session_state.selected_movie = None
                    st.warning("No recommendations found for this movie.")

                else:
                    # Save recommendations so they survive reruns.
                    st.session_state.recommendations = result.reset_index(
                        drop=True
                    )
                    st.session_state.selected_movie = selected_movie

                    # New filter widget keys -> filters start fresh.
                    st.session_state.rec_version += 1

                # A new search always closes any open movie details.
                st.session_state.selected_detail_id = None
                st.session_state.selected_detail_title = None
                st.session_state.open_details = False

            except Exception as error:
                st.error(
                    f"Something went wrong while recommending movies: {error}"
                )


# =========================================================
# MOVIE DETAILS (pop-up, or inline on older Streamlit)
# Works for movies opened from any section of the page.
# =========================================================

if st.session_state.selected_detail_title is not None:

    if HAS_DIALOG:
        # Open only on the run right after a click; otherwise the pop-up
        # would reopen on every unrelated rerun.
        if st.session_state.open_details:
            st.session_state.open_details = False
            details_dialog()

    else:
        render_movie_details()
        st.divider()


# =========================================================
# RESULTS + SEARCH + FILTERS
# =========================================================

recommendations = st.session_state.recommendations
version = st.session_state.rec_version

if recommendations is not None and not recommendations.empty:

    st.markdown(
        '<div class="section-label" id="recommendations">'
        'Your personalized results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Recommended For You</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"Because you liked **{st.session_state.selected_movie}**"
    )

    st.write("")

    # -----------------------------------------
    # SEARCH AND FILTER CONTROLS
    # -----------------------------------------

    st.markdown("### Find Your Perfect Movie")

    search_col, sort_col = st.columns([2, 1])

    with search_col:
        search_query = st.text_input(
            "🔍 Search by movie title",
            placeholder="Search movies...",
            key=f"movie_search_{version}",
        )

    with sort_col:
        sort_option = st.selectbox(
            "Sort movies by",
            ["Best Match", "Newest First", "Oldest First"],
            key=f"movie_sort_{version}",
        )

    # Work on a separate copy
    filtered_recommendations = recommendations.copy()

    # -----------------------------------------
    # GENRE FILTER
    # -----------------------------------------

    genre_options = (
        collect_genres(filtered_recommendations["genres"])
        if "genres" in filtered_recommendations.columns
        else []
    )

    selected_genres = st.multiselect(
        "Filter by genre",
        options=genre_options,
        placeholder="Select genres...",
        key=f"movie_genre_filter_{version}",
    )

    # -----------------------------------------
    # YEAR FILTER
    # -----------------------------------------

    available_years = []

    if "year" in filtered_recommendations.columns:
        year_series = pd.to_numeric(
            filtered_recommendations["year"],
            errors="coerce",
        )
        available_years = sorted(
            year_series.dropna().astype(int).unique().tolist()
        )

    selected_year_range = None

    # select_slider needs at least two options.
    if len(available_years) >= 2:
        selected_year_range = st.select_slider(
            "Release year range",
            options=available_years,
            value=(available_years[0], available_years[-1]),
            key=f"movie_year_filter_{version}",
        )

    # -----------------------------------------
    # APPLY SEARCH
    # -----------------------------------------

    if search_query.strip() and "title" in filtered_recommendations.columns:
        filtered_recommendations = filtered_recommendations[
            filtered_recommendations["title"].str.contains(
                search_query.strip(),
                case=False,
                na=False,
                regex=False,  # titles contain "(" etc.; avoid regex errors
            )
        ]

    # -----------------------------------------
    # APPLY GENRE FILTER
    # -----------------------------------------

    if selected_genres:
        filtered_recommendations = filtered_recommendations[
            filtered_recommendations["genres"].apply(
                lambda genres: has_any_genre(genres, selected_genres)
            )
        ]

    # -----------------------------------------
    # APPLY YEAR FILTER
    # -----------------------------------------

    if selected_year_range:
        filtered_years = pd.to_numeric(
            filtered_recommendations["year"],
            errors="coerce",
        )

        filtered_recommendations = filtered_recommendations[
            filtered_years.between(
                selected_year_range[0],
                selected_year_range[1],
            )
        ]

    # -----------------------------------------
    # SORT RESULTS
    # -----------------------------------------

    if sort_option == "Newest First":
        filtered_recommendations = sort_by_year(
            filtered_recommendations, ascending=False
        )

    elif sort_option == "Oldest First":
        filtered_recommendations = sort_by_year(
            filtered_recommendations, ascending=True
        )

    elif "similarity_score" in filtered_recommendations.columns:
        filtered_recommendations = filtered_recommendations.sort_values(
            "similarity_score",
            ascending=False,
        )

    # -----------------------------------------
    # RESULTS COUNT
    # -----------------------------------------

    st.divider()

    count_col1, count_col2 = st.columns(2)

    with count_col1:
        st.metric("Movies Found", len(filtered_recommendations))

    with count_col2:
        st.metric("Total Recommendations", len(recommendations))

    st.write("")

    # -----------------------------------------
    # DISPLAY FILTERED MOVIES
    # -----------------------------------------

    if not filtered_recommendations.empty:

        render_movie_grid(
            [row for _, row in filtered_recommendations.iterrows()],
            key_prefix="rec",
        )

    else:

        st.info(
            "No movies match your search or filters. "
            "Try changing your search criteria."
        )

else:

    st.markdown(
        """
    <div class="empty-state">
        <div class="empty-circle">🍿</div>
        <h3>Ready for your next movie?</h3>
        <p>
            Choose a movie from the sidebar and
            let CineMatch find something similar.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# =========================================================
# TOP RATED MOVIES
# =========================================================

st.divider()

st.markdown(
    '<div class="section-label" id="top-rated">Community favourites</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">Top Rated Movies ⭐</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Discover highly rated movies based on MovieLens user ratings."
)

if not popular_movies.empty:

    popular_genres = (
        collect_genres(popular_movies["genres"])
        if "genres" in popular_movies.columns
        else []
    )

    filter_col, popular_sort_col = st.columns([2, 1])

    with filter_col:
        popular_selected_genres = st.multiselect(
            "Filter popular movies by genre",
            options=popular_genres,
            placeholder="Select genres...",
            key="popular_genre_filter",
        )

    with popular_sort_col:
        popular_sort = st.selectbox(
            "Sort popular movies",
            ["Highest Rated", "Most Ratings", "Newest First"],
            key="popular_sort",
        )

    filtered_popular = popular_movies.copy()

    # Genre filter
    if popular_selected_genres:
        filtered_popular = filtered_popular[
            filtered_popular["genres"].apply(
                lambda genres: has_any_genre(genres, popular_selected_genres)
            )
        ]

    # Sorting
    if popular_sort == "Most Ratings" and "rating_count" in filtered_popular:
        filtered_popular = filtered_popular.sort_values(
            "rating_count",
            ascending=False,
        )

    elif popular_sort == "Newest First":
        filtered_popular = sort_by_year(filtered_popular, ascending=False)

    elif "weighted_rating" in filtered_popular:
        filtered_popular = filtered_popular.sort_values(
            "weighted_rating",
            ascending=False,
        )

    st.write("")

    st.metric("Top Rated Movies Found", len(filtered_popular))

    st.write("")

    if not filtered_popular.empty:
        render_movie_grid(
            [row for _, row in filtered_popular.iterrows()],
            key_prefix="popular",
            show_rating=True,
        )

    else:
        st.info(
            "No movies match the selected genre. "
            "Try another genre."
        )

else:
    st.info("Top rated movies are currently unavailable.")


# =========================================================
# ANALYTICS DASHBOARD
# =========================================================

st.markdown(
    '<div id="analytics"></div>',
    unsafe_allow_html=True,
)

if ratings_data is not None:
    try:
        render_analytics_dashboard(movies, ratings_data)
    except Exception as error:
        st.error(f"Unable to load the analytics dashboard: {error}")


# =========================================================
# WATCHLIST & FAVOURITES
# Rendered after the results, so a movie added above is
# already counted and listed in the same run.
# =========================================================

st.divider()

st.markdown(
    '<div class="section-label" id="collection">'
    'Your personal collection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">My Movie Collection ❤️</div>',
    unsafe_allow_html=True,
)

try:
    saved_data = load_saved_movies() or {}
except Exception as error:
    st.error(f"Unable to load your saved movies: {error}")
    saved_data = {}

watchlist_movies = saved_data.get("watchlist", [])
favourite_movies = saved_data.get("favourites", [])

tab_watchlist, tab_favourites = st.tabs([
    f"🎬 Watchlist ({len(watchlist_movies)})",
    f"❤️ Favourites ({len(favourite_movies)})",
])


def display_saved_collection(saved_movies, collection_name):
    if not saved_movies:
        st.info(
            f"Your {collection_name} is empty. "
            "Add movies from your recommendations!"
        )
        return

    render_movie_grid(
        saved_movies,
        key_prefix=f"saved_{collection_name}",
        collection=collection_name,
    )


with tab_watchlist:
    display_saved_collection(watchlist_movies, "watchlist")

with tab_favourites:
    display_saved_collection(favourite_movies, "favourites")


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
    <div>
        CineMatch · AI Movie Recommendation System<br>
        Built with Python | Streamlit | Scikit-learn | TMDB
    </div>
    <div>
        Movie data and images provided by
        <a href="https://www.themoviedb.org/" target="_blank">
            The Movie Database (TMDB)
        </a>
    </div>
</div>
""",
    unsafe_allow_html=True,
)