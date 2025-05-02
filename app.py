# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split

st.set_page_config(page_title="Hybrid Movie Recommender", layout="centered")

# --- Load MovieLens 100K ---
@st.cache_data
def load_data():
    base_path = './ml-100k/'
    ratings = pd.read_csv(base_path + 'u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies = pd.read_csv(base_path + 'u.item', sep='|', encoding='latin-1', names=[
        'movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL',
        'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ])
    return ratings, movies

ratings, movies = load_data()

# --- Build Content-Based Model ---
genre_cols = movies.columns[5:]
movies["genres"] = movies[genre_cols].apply(lambda row: " ".join([col for col, val in zip(genre_cols, row) if val == 1]), axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# --- Build Collaborative Model ---
dataset = Dataset()
dataset.fit(ratings["user_id"].unique(), ratings["item_id"].unique())

item_mapping = dict(dataset.mapping()[1])
item_inv_mapping = {v: k for k, v in item_mapping.items()}

(interactions, _) = dataset.build_interactions(
    ((row["user_id"], row["item_id"], row["rating"]) for _, row in ratings.iterrows())
)

train, _ = random_train_test_split(interactions, test_percentage=0.2)
model = LightFM(loss='warp', no_components=32, learning_rate=0.05)
model.fit(train, epochs=10, num_threads=4)

# --- NLP Parser ---
def parse_user_input(text):
    user_id_match = re.search(r'user\s*(number)?\s*(\d+)', text.lower())
    user_id = int(user_id_match.group(2)) if user_id_match else None

    movie_match = re.search(r'liked\s+(.*?)\s+(and|i\'m)', text, re.IGNORECASE)
    movie_title = movie_match.group(1).strip() if movie_match else None

    return user_id, movie_title

# --- Hybrid Recommendation Function ---
def hybrid_recommend(user_id, movie_title, alpha=0.5, top_n=10):
    try:
        idx = movies[movies["movie_title"] == movie_title].index[0]
    except IndexError:
        st.error("❌ Movie title not found.")
        return pd.DataFrame()

    cb_scores = cosine_sim[idx]

    movie_ids = movies["movie_id"].tolist()
    known_ids = [mid for mid in movie_ids if mid in item_mapping]
    internal_ids = [item_mapping[mid] for mid in known_ids]

    user_ids = [user_id] * len(internal_ids)
    cf_scores = model.predict(user_ids, internal_ids)

    cb_scores = [cb_scores[movies[movies["movie_id"] == mid].index[0]] for mid in known_ids]

    cb_scaled = MinMaxScaler().fit_transform(np.array(cb_scores).reshape(-1, 1)).flatten()
    cf_scaled = MinMaxScaler().fit_transform(cf_scores.reshape(-1, 1)).flatten()

    final_scores = alpha * cb_scaled + (1 - alpha) * cf_scaled
    top_indices = np.argsort(final_scores)[::-1][:top_n]

    top_movie_ids = [known_ids[i] for i in top_indices]
    return movies[movies["movie_id"].isin(top_movie_ids)][["movie_title", "genres"]]

# --- Streamlit UI ---
st.title("🎬 Hybrid Movie Recommender System")

st.markdown("""
Enter something like:  
`I liked Toy Story (1995) and I’m user number 10`
""")

user_input = st.text_input("Your sentence:")
alpha = st.slider("🔄 Content-Based Weight (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

if st.button("🎥 Recommend"):
    user_id, movie_title = parse_user_input(user_input)
    if user_id is not None and movie_title is not None:
        movie_title_match = movies[movies["movie_title"].str.lower() == movie_title.lower()]
        if not movie_title_match.empty:
            correct_title = movie_title_match.iloc[0]["movie_title"]
            st.success(f"Recommending for user **{user_id}** based on movie: **{correct_title}**")
            recs = hybrid_recommend(user_id, correct_title, alpha)
            st.dataframe(recs)
        else:
            st.warning(f"❌ Movie '{movie_title}' not found. Try an exact title from MovieLens dataset.")
    else:
        st.warning("⚠️ Could not parse user input. Use format like: I liked Titanic and I’m user 5")
