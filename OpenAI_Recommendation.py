import openai
import requests
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

# Configuration
TMDB_API_KEY = 'enter-your-tmdb-key'
OPENAI_API_KEY = 'enter-your-OpenAI-key'
BASE_URL = 'https://api.themoviedb.org/3'

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)

# Fetch all currently playing movies from TMDb
def fetch_all_movies():
    movies = []
    page = 1
    while True:
        url = f"{BASE_URL}/movie/now_playing"
        params = {
            'api_key': TMDB_API_KEY,
            'language': 'en-US',
            'page': page
        }
        response = requests.get(url, params=params)
        if response.status_code != 200:
            break
        page_movies = response.json().get('results', [])
        if not page_movies:
            break
        movies.extend(page_movies)
        page += 1
    return movies

# Fetch genre mappings from TMDb
def fetch_genres():
    url = f"{BASE_URL}/genre/movie/list"
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        genres = response.json().get('genres', [])
        return {genre['name'].lower(): genre['id'] for genre in genres}
    else:
        print("Failed to fetch genres from TMDb API:", response.status_code, response.text)
        return {}

# Prepare movie data for the model
def prepare_movie_data(movies):
    movie_data = {
        'title': [],
        'overview': [],
        'genre_ids': []
    }

    for movie in movies:
        title = movie['title']
        overview = movie['overview']
        genre_ids = movie['genre_ids']
        
        # Check if overview is not empty and contains meaningful text
        if overview and len(overview.split()) > 2:  # Adjust as needed
            movie_data['title'].append(title)
            movie_data['overview'].append(overview)
            movie_data['genre_ids'].append(genre_ids)

    return pd.DataFrame(movie_data)

# Train the recommendation model
def train_model(movies_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Generate recommendations based on cosine similarity
def get_recommendations(title, genre, movies_df, cosine_sim, genre_map):
    # Convert to lowercase for case-insensitive matching
    title = title.lower()
    genre = genre.lower()

    # Partial matching for titles
    movies_df['title_lower'] = movies_df['title'].str.lower()
    matched_titles = movies_df[movies_df['title_lower'].str.contains(title)]
    
    # Filter by genre if specified
    if genre in genre_map:
        genre_id = genre_map[genre]
        matched_titles = matched_titles[matched_titles['genre_ids'].apply(lambda x: genre_id in x)]

    if matched_titles.empty:
        return ["No recommendations found."]

    idx = matched_titles.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()

# Use OpenAI to refine recommendations
def openai_refine_recommendations(user_input, recommendations):
    prompt = f"User is interested in movies related to '{user_input}'. Here are some recommendations: {recommendations}. Refine these recommendations considering user's interest."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # enter your model name
        messages=[
            {"role": "system", "content": "You are a helpful movie recommendation assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    refined_recommendations = response['choices'][0]['message']['content'].strip()
    return refined_recommendations.split(', ')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for generating recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input']
    genre_input = request.form.get('genre_input', '')
    recommendations = []

    # Fetch and prepare movie data
    all_movies = fetch_all_movies()
    movies_df = prepare_movie_data(all_movies)

    # Fetch genres
    genre_map = fetch_genres()

    # Check if there are movies with valid overviews
    if not movies_df.empty:
        cosine_sim = train_model(movies_df)

        # Get recommendations
        recommendations = get_recommendations(user_input, genre_input, movies_df, cosine_sim, genre_map)

        # Refine recommendations using OpenAI
        recommendations = openai_refine_recommendations(user_input, recommendations)

    return jsonify(recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
