import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="Content-Based Movie Recommender", layout="centered")

# Load data
@st.cache_data
def load_movies():
    movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', names=[
        'movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ])
    genre_cols = movies.columns[5:]
    movies['genres'] = movies[genre_cols].apply(lambda row: " ".join([col for col, val in zip(genre_cols, row) if val == 1]), axis=1)
    return movies

movies = load_movies()

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Input parser
def parse_user_input(text):
    user_id_match = re.search(r'user\s*(number)?\s*(\d+)', text.lower())
    user_id = int(user_id_match.group(2)) if user_id_match else None

    movie_match = re.search(r'liked\s+(.*?)\s+(and|i\'m)', text, re.IGNORECASE)
    movie_title = movie_match.group(1).strip() if movie_match else None

    return user_id, movie_title

# Content-based recommender
def content_recommend(movie_title, top_n=10):
    if movie_title not in movies["movie_title"].values:
        return pd.DataFrame()

    idx = movies[movies["movie_title"] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["movie_title", "genres"]]

# UI
st.title("🎬 Content-Based Movie Recommender")
st.markdown("Example: `I liked Toy Story (1995) and I’m user number 5`")

user_input = st.text_input("Enter what you liked:")
if st.button("Recommend"):
    _, movie_title = parse_user_input(user_input)
    if movie_title:
        recs = content_recommend(movie_title)
        if not recs.empty:
            st.dataframe(recs)
        else:
            st.warning("Movie not found. Try an exact title from the dataset.")
    else:
        st.warning("Please follow the format: I liked [movie] and I'm user [number]")
