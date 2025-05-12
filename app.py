import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
import string
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess merged dataset
@st.cache_data
def load_and_prepare_data():
    base_path = '/Users/zeynepsalihoglu/MovieRecommendation/'
    movies = pd.read_csv(base_path + 'movies_metadata.csv', low_memory=False)
    credits = pd.read_csv(base_path + 'credits.csv')
    keywords = pd.read_csv(base_path + 'keywords.csv')

    # Drop known bad rows
    movies = movies.drop([19730, 29503, 35587])
    movies['id'] = movies['id'].astype('int64')
    
    # Merge
    df = movies.merge(keywords, on='id').merge(credits, on='id')
    df['original_language'] = df['original_language'].fillna('')
    df['runtime'] = df['runtime'].fillna(0)
    df['tagline'] = df['tagline'].fillna('')
    df.dropna(inplace=True)

    # Convert types
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['budget'] = df['budget'].astype('float64')
    df['popularity'] = df['popularity'].astype('float64')
    df['vote_average'] = df['vote_average'].astype('float64')
    df['vote_count'] = df['vote_count'].astype('float64')
    
    return df

df = load_and_prepare_data()

# Calculate weighted average
R = df['vote_average']
v = df['vote_count']
m = df['vote_count'].quantile(0.8)
C = df['vote_average'].mean()
df['weighted_average'] = (R * v + C * m) / (v + m)

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['popularity', 'weighted_average']])
weighted_df = pd.DataFrame(scaled, columns=['popularity', 'weighted_average'])
weighted_df.index = df['original_title']
weighted_df['score'] = weighted_df['weighted_average'] * 0.4 + weighted_df['popularity'] * 0.6
weighted_df_sorted = weighted_df.sort_values(by='score', ascending=False)

# Helpers to parse and clean
def parse_list(s):
    try:
        return ' '.join([i['name'].replace(' ', '').lower() for i in ast.literal_eval(s)])
    except:
        return ''

def clean_text(text):
    cleaned = str(text).translate(str.maketrans('', '', string.punctuation)).lower()
    return cleaned.translate(str.maketrans('', '', string.digits))

# Extract features
df['genres_clean'] = df['genres'].apply(parse_list)
df['keywords_clean'] = df['keywords'].apply(parse_list)
df['cast_clean'] = df['cast'].apply(lambda x: ' '.join([i['name'].replace(' ', '').lower() for i in ast.literal_eval(x)][:3]) if pd.notnull(x) else '')
df['crew_clean'] = df['crew'].apply(lambda x: ' '.join([i['name'].replace(' ', '').lower() for i in ast.literal_eval(x) if i['job'] == 'Director']) if pd.notnull(x) else '')
df['overview_clean'] = df['overview'].apply(clean_text)
df['tagline_clean'] = df['tagline'].apply(clean_text)

# Build content dataframe
content_df = pd.DataFrame()
content_df['original_title'] = df['original_title']
content_df['bag_of_words'] = df['genres_clean'] + ' ' + df['keywords_clean'] + ' ' + df['cast_clean'] + ' ' + df['crew_clean'] + ' ' + df['overview_clean'] + ' ' + df['tagline_clean']
content_df.set_index('original_title', inplace=True)

# Merge top 10k with scores
content_df = weighted_df_sorted[:10000].merge(content_df, left_index=True, right_index=True, how='left')

# TF-IDF and similarity
tfidf = TfidfVectorizer(stop_words='english', min_df=5)
tfidf_matrix = tfidf.fit_transform(content_df['bag_of_words'])
cos_sim = cosine_similarity(tfidf_matrix)

# Recommendation function
def predict(title, similarity_weight=0.7, top_n=10):
    data = content_df.reset_index()
    if title not in data['original_title'].values:
        return None
    idx = data[data['original_title'] == title].index[0]
    similarity = cos_sim[idx]
    sim_df = pd.DataFrame({'similarity': similarity})
    final_df = pd.concat([data, sim_df], axis=1)
    final_df['final_score'] = final_df['score'] * (1 - similarity_weight) + final_df['similarity'] * similarity_weight
    final_df_sorted = final_df.sort_values(by='final_score', ascending=False).head(top_n)
    final_df_sorted.set_index('original_title', inplace=True)
    return final_df_sorted[['score', 'similarity', 'final_score']]

# Streamlit UI
st.title("🎥 Movie Recommendation App")
st.markdown("Choose a movie to get personalized content-based recommendations!")

movie_list = content_df.index.tolist()
selected_movie = st.selectbox("Select a movie:", sorted(movie_list))
sim_weight = st.slider("Similarity Weight", 0.0, 1.0, 0.7, 0.1)
top_n = st.slider("Top N Recommendations", 5, 20, 10)

if st.button("Get Recommendations"):
    result = predict(selected_movie, similarity_weight=sim_weight, top_n=top_n)
    if result is not None:
        st.subheader(f"Top {top_n} Recommendations for '{selected_movie}':")
        st.dataframe(result.style.format({"score": "{:.2f}", "similarity": "{:.2f}", "final_score": "{:.2f}"}))
    else:
        st.error("❌ Movie not found in the dataset.")
