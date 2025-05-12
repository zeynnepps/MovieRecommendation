import pandas as pd
from ast import literal_eval

# Step 1: Load the CSV files (assumes all are in the same folder)
movies = pd.read_csv('movies_metadata.csv', low_memory=False)
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# Step 2: Clean problematic rows and cast types
movies = movies.drop([19730, 29503, 35587])
movies['id'] = movies['id'].astype('int64')

# Step 3: Merge datasets on 'id'
df = movies.merge(keywords, on='id').merge(credits, on='id')

# Step 4: Handle missing values
df['original_language'] = df['original_language'].fillna('')
df['runtime'] = df['runtime'].fillna(0)
df['tagline'] = df['tagline'].fillna('')
df.dropna(inplace=True)

# Step 5: Convert columns with list-like strings into comma-separated text
def get_text(text, obj='name'):
    text = literal_eval(text)
    if len(text) == 1:
        for i in text:
            return i[obj]
    else:
        return ', '.join([i[obj] for i in text])

df['genres'] = df['genres'].apply(get_text)
df['production_companies'] = df['production_companies'].apply(get_text)
df['production_countries'] = df['production_countries'].apply(get_text)
df['crew'] = df['crew'].apply(get_text)
df['spoken_languages'] = df['spoken_languages'].apply(get_text)
df['keywords'] = df['keywords'].apply(get_text)

# Step 6: Extract characters and actors
df['characters'] = df['cast'].apply(get_text, obj='character')
df['actors'] = df['cast'].apply(get_text)
df.drop('cast', axis=1, inplace=True)

# Step 7: Drop duplicates
df = df[~df['original_title'].duplicated()].reset_index(drop=True)

# Step 8: Fix data types for analysis
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0)
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)

# Optional: Save to a cleaned CSV for Streamlit use
df.to_csv('cleaned_movie_data.csv', index=False)
print("✅ Preprocessing complete. Saved as 'cleaned_movie_data.csv'.")
