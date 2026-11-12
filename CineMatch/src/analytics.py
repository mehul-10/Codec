
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# CINE MATCH ANALYTICS DASHBOARD
# =========================================================

def render_analytics_dashboard(movies, ratings):

    st.markdown("---")

    st.markdown(
        """
        <div class="section-heading">
            <h2>📊 Movie Analytics</h2>
            <p>Explore insights from the MovieLens dataset.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if movies is None or ratings is None:
        st.warning("Analytics data is currently unavailable.")
        return

    if movies.empty or ratings.empty:
        st.warning("Not enough data available to display analytics.")
        return

    # -----------------------------------------------------
    # PREPARE DATA
    # -----------------------------------------------------

    movie_data = movies.copy()
    rating_data = ratings.copy()

    rating_data["rating"] = pd.to_numeric(
        rating_data["rating"],
        errors="coerce"
    )

    rating_data = rating_data.dropna(subset=["rating"])

    movie_data["year"] = pd.to_numeric(
        movie_data["year"],
        errors="coerce"
    )

    # -----------------------------------------------------
    # CALCULATE METRICS
    # -----------------------------------------------------

    total_movies = len(movie_data)

    total_ratings = len(rating_data)

    average_rating = (
        rating_data["rating"].mean()
        if not rating_data.empty
        else 0
    )

    total_users = (
        rating_data["userId"].nunique()
        if "userId" in rating_data.columns
        else 0
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Total Movies",
        f"{total_movies:,}"
    )

    metric2.metric(
        "Total Ratings",
        f"{total_ratings:,}"
    )

    metric3.metric(
        "Average Rating",
        f"{average_rating:.2f}/5"
    )

    metric4.metric(
        "Active Users",
        f"{total_users:,}"
    )

    st.markdown("---")

    # -----------------------------------------------------
    # CHART 1: GENRE DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("🎭 Genre Distribution")

    genre_data = movie_data[["genres"]].copy()

    genre_data["genres"] = genre_data["genres"].fillna("")

    genre_data["genres"] = genre_data["genres"].str.split("|")

    genre_data = genre_data.explode("genres")

    genre_data["genres"] = genre_data["genres"].str.strip()

    genre_data = genre_data[
        (genre_data["genres"] != "")
        & (genre_data["genres"] != "(no genres listed)")
    ]

    genre_counts = (
        genre_data["genres"]
        .value_counts()
        .reset_index()
    )

    genre_counts.columns = ["Genre", "Movie Count"]

    if not genre_counts.empty:

        genre_chart = px.bar(
            genre_counts.head(15),
            x="Genre",
            y="Movie Count",
            title="Movies by Genre",
            template="plotly_dark",
            text="Movie Count"
        )

        genre_chart.update_layout(
            xaxis_title="Genre",
            yaxis_title="Number of Movies",
            xaxis_tickangle=-35,
            height=420
        )

        st.plotly_chart(
            genre_chart,
            use_container_width=True
        )

    # -----------------------------------------------------
    # CHART 2: RATING DISTRIBUTION
    # -----------------------------------------------------

    st.subheader("⭐ Rating Distribution")

    rating_chart = px.histogram(
        rating_data,
        x="rating",
        nbins=10,
        title="Distribution of User Ratings",
        template="plotly_dark",
        labels={
            "rating": "Movie Rating",
            "count": "Number of Ratings"
        }
    )

    rating_chart.update_layout(
        xaxis_title="Rating",
        yaxis_title="Number of Ratings",
        height=420
    )

    st.plotly_chart(
        rating_chart,
        use_container_width=True
    )

    # -----------------------------------------------------
    # CHART 3: MOVIES RELEASED BY YEAR
    # -----------------------------------------------------

    st.subheader("📅 Movies Released Over the Years")

    year_data = (
        movie_data.dropna(subset=["year"])
        .groupby("year")
        .size()
        .reset_index(name="Movie Count")
        .sort_values("year")
    )

    year_data = year_data[
        (year_data["year"] >= 1900)
        & (year_data["year"] <= 2030)
    ]

    if not year_data.empty:

        year_chart = px.line(
            year_data,
            x="year",
            y="Movie Count",
            title="Movie Releases by Year",
            markers=True,
            template="plotly_dark"
        )

        year_chart.update_layout(
            xaxis_title="Release Year",
            yaxis_title="Number of Movies",
            height=420
        )

        st.plotly_chart(
            year_chart,
            use_container_width=True
        )

    # -----------------------------------------------------
    # CHART 4: TOP 10 HIGHEST-RATED MOVIES
    # -----------------------------------------------------

    st.subheader("🏆 Top 10 Highest-Rated Movies")

    movie_ratings = (
        rating_data.groupby("movieId")
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    # Require at least 50 ratings for a fair comparison
    movie_ratings = movie_ratings[
        movie_ratings["rating_count"] >= 50
    ]

    top_movies = movie_data.merge(
        movie_ratings,
        on="movieId",
        how="inner"
    )

    if not top_movies.empty:

        overall_average = rating_data["rating"].mean()
        minimum_ratings = 50

        # Bayesian weighted rating
        top_movies["weighted_rating"] = (
            (
                top_movies["rating_count"]
                / (top_movies["rating_count"] + minimum_ratings)
            )
            * top_movies["average_rating"]
            +
            (
                minimum_ratings
                / (top_movies["rating_count"] + minimum_ratings)
            )
            * overall_average
        )

        top_movies = top_movies.nlargest(
            10,
            "weighted_rating"
        ).sort_values("weighted_rating")

        top_movies["display_title"] = (
            top_movies["title"].astype(str).str.slice(0, 45)
        )

        top_chart = px.bar(
            top_movies,
            x="weighted_rating",
            y="display_title",
            orientation="h",
            title="Top 10 Movies by Weighted Rating",
            template="plotly_dark",
            text="weighted_rating",
            hover_data=[
                "average_rating",
                "rating_count"
            ]
        )

        top_chart.update_layout(
            xaxis_title="Weighted Rating",
            yaxis_title="Movie",
            xaxis_range=[0, 5],
            height=500
        )

        top_chart.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            top_chart,
            use_container_width=True
        )

    else:
        st.info(
            "Not enough movie ratings to calculate the Top 10."
        )

    st.markdown("---")

    st.caption(
        "CineMatch Analytics | Insights powered by MovieLens data."
    )
