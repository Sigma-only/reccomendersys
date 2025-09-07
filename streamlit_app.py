# streamlit_recommendation_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
import os
import glob

# Set page config
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon="🎮",
    layout="wide"
)

# --- Data Loading Functions ---
@st.cache_data
def load_metadata():
    """Load game metadata"""
    return pd.read_csv('gamedata.csv', low_memory=False)

@st.cache_data
def load_users():
    """Load user data"""
    return pd.read_csv('steamuser.csv')

@st.cache_data
def load_ratings():
    """
    Load and combine all rating chunks
    This handles the split n_ratings.csv files
    """
    rating_files = glob.glob('split_ratings/n_ratings_chunk_*.csv')
    
    if not rating_files:
        st.error("No rating files found! Please make sure split_ratings directory exists with n_ratings_chunk_*.csv files")
        return None
    
    rating_chunks = []
    for file in sorted(rating_files):
        chunk = pd.read_csv(file)
        rating_chunks.append(chunk)
    
    return pd.concat(rating_chunks, ignore_index=True)

# --- Content-Based Filtering Functions ---
def combine_features(row, fields_to_include):
    features = []
    if 'description' in fields_to_include:
        features.append(row['short_description'])
        features.append(row['detailed_description'])
        features.append(row['about_the_game'])
    if 'genres' in fields_to_include:
        features.append(row['genres'])
    if 'developer' in fields_to_include:
        features.append(row['developer'])
    if 'publisher' in fields_to_include:
        features.append(row['publisher'])
    if 'platforms' in fields_to_include:
        features.append(row['platforms'])
    if 'required_age' in fields_to_include:
        features.append(str(row['required_age']))
    if 'steamspy_tags' in fields_to_include:
        features.append(row['steamspy_tags'])
    return ' '.join(features)

def get_content_based_recommendations(game_name='', description_keywords='', developer='', 
                                    publisher='', platforms='', required_age=None, 
                                    genres='', steamspy_tags_input='', metadata=None, top_n=50):
    
    filtered_metadata = metadata.copy()
    
    # Apply filters
    if game_name.strip():
        filtered_metadata = filtered_metadata[filtered_metadata['name'] == game_name]
    
    if developer.strip():
        filtered_metadata = filtered_metadata[filtered_metadata['developer'].str.contains(developer, case=False, na=False)]
    
    if publisher.strip():
        filtered_metadata = filtered_metadata[filtered_metadata['publisher'].str.contains(publisher, case=False, na=False)]
    
    if platforms.strip():
        filtered_metadata = filtered_metadata[filtered_metadata['platforms'].str.contains(platforms, case=False, na=False)]
    
    if required_age is not None:
        filtered_metadata = filtered_metadata[filtered_metadata['required_age'] == required_age]
    
    if genres.strip():
        filtered_metadata = filtered_metadata[filtered_metadata['genres'].str.contains(genres, case=False, na=False)]
    
    if steamspy_tags_input.strip():
        tags = steamspy_tags_input.lower().split()
        if tags:
            if 'or' in tags:
                or_filter = filtered_metadata['steamspy_tags'].str.contains(tags[0], case=False, na=False)
                for tag in tags[1:]:
                    if tag != 'or':
                        or_filter = or_filter | filtered_metadata['steamspy_tags'].str.contains(tag, case=False, na=False)
                filtered_metadata = filtered_metadata[or_filter]
            else:
                for tag in tags:
                    if tag != 'and':
                        filtered_metadata = filtered_metadata[filtered_metadata['steamspy_tags'].str.contains(tag, case=False, na=False)]
    
    if filtered_metadata.empty:
        return "No games found matching the specified filters."
    
    # If description keywords provided, use TF-IDF
    if description_keywords.strip():
        fields_to_include = ['description', 'genres', 'developer', 'publisher', 'platforms', 'required_age', 'steamspy_tags']
        filtered_metadata['combined_features'] = filtered_metadata.apply(
            lambda row: combine_features(row, fields_to_include), axis=1
        )
        
        tfidf_filtered = TfidfVectorizer(stop_words='english')
        tfidf_matrix_filtered = tfidf_filtered.fit_transform(filtered_metadata['combined_features'])
        
        input_vec = tfidf_filtered.transform([description_keywords])
        sim_scores = linear_kernel(input_vec, tfidf_matrix_filtered).flatten()
        sim_indices = sim_scores.argsort()[-top_n:][::-1]
        
        return filtered_metadata['name'].iloc[sim_indices]
    else:
        return filtered_metadata['name'].head(top_n)

# --- Collaborative Filtering Functions ---
@st.cache_data
def prepare_collaborative_data(_games_df, _ratings_df, _users_df):
    """Prepare data for collaborative filtering"""
    merged_df = pd.merge(_ratings_df, _games_df, on='appid')
    merged_df = pd.merge(merged_df, _users_df, on='userID')
    
    user_item_matrix = merged_df.pivot_table(index='userID', columns='appid', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    
    return user_item_matrix, merged_df

def generate_collaborative_recommendations(selected_game_ratings_list, user_item_matrix, games_df, similarity_metric='cosine'):
    """Generate collaborative recommendations"""
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, 
                                    index=user_item_matrix.columns, 
                                    columns=user_item_matrix.columns)
    
    all_game_weighted_scores = {}
    
    for selected_game, user_rating in selected_game_ratings_list:
        selected_game_appid = selected_game['appid'].iloc[0]
        try:
            game_similarity_scores = item_similarity_df[selected_game_appid]
            weighted_scores = game_similarity_scores * user_rating
            
            for appid, score in weighted_scores.items():
                if appid not in all_game_weighted_scores:
                    all_game_weighted_scores[appid] = 0
                all_game_weighted_scores[appid] += score
                
        except KeyError:
            st.warning(f"Game with appid {selected_game_appid} not found in user-item matrix. Skipping.")
            continue
    
    all_game_weighted_scores_series = pd.Series(all_game_weighted_scores)
    sorted_weighted_scores = all_game_weighted_scores_series.sort_values(ascending=False)
    
    # Remove already rated games
    rated_game_appids = [game['appid'].iloc[0] for game, rating in selected_game_ratings_list]
    recommendations = sorted_weighted_scores.drop(rated_game_appids, errors='ignore')
    
    return recommendations

# --- Streamlit App ---
def main():
    st.title("🎮 Game Recommendation System")
    st.write("Discover new games based on your preferences!")
    
    # Load data
    with st.spinner("Loading data..."):
        metadata = load_metadata()
        users_df = load_users()
        ratings_df = load_ratings()
        
        if ratings_df is None:
            st.stop()
        
        # Fill NaN values
        for col in ['short_description', 'developer', 'publisher', 'platforms', 
                   'required_age', 'categories', 'genres', 'steamspy_tags', 
                   'detailed_description', 'about_the_game']:
            metadata[col] = metadata[col].fillna('')
        
        # Prepare collaborative data
        user_item_matrix, merged_df = prepare_collaborative_data(metadata, ratings_df, users_df)
    
    # Recommendation type selection
    st.sidebar.header("Recommendation Type")
    rec_type = st.sidebar.radio("Choose recommendation type:", 
                               ["Content-Based Filtering", "Collaborative Filtering"])
    
    if rec_type == "Content-Based Filtering":
        content_based_ui(metadata)
    else:
        collaborative_ui(metadata, user_item_matrix)

def content_based_ui(metadata):
    st.header("Content-Based Recommendations")
    st.write("Find games similar to your preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Filter Criteria")
        
        # Game name search
        game_names = [''] + sorted(metadata['name'].dropna().unique().tolist())
        selected_game = st.selectbox("Search for a specific game:", game_names)
        
        # Text filters
        description_keywords = st.text_input("Description keywords:")
        developer = st.text_input("Developer:")
        publisher = st.text_input("Publisher:")
        platforms = st.text_input("Platforms:")
        
    with col2:
        st.subheader("Additional Filters")
        
        # Age filter
        age_options = ['Any'] + sorted(metadata['required_age'].dropna().unique().tolist())
        required_age = st.selectbox("Required Age:", age_options)
        required_age = None if required_age == 'Any' else required_age
        
        # Genre filter
        genres = st.text_input("Genres:")
        steamspy_tags = st.text_input("SteamSpy Tags (use 'and' or 'or'):")
        
        # Number of results
        top_n = st.slider("Number of recommendations:", 5, 100, 20)
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Finding recommendations..."):
            results = get_content_based_recommendations(
                game_name=selected_game,
                description_keywords=description_keywords,
                developer=developer,
                publisher=publisher,
                platforms=platforms,
                required_age=required_age,
                genres=genres,
                steamspy_tags_input=steamspy_tags,
                metadata=metadata,
                top_n=top_n
            )
        
        if isinstance(results, str):
            st.warning(results)
        else:
            st.success(f"Found {len(results)} recommendations!")
            for i, game in enumerate(results, 1):
                st.write(f"{i}. {game}")

def collaborative_ui(metadata, user_item_matrix):
    st.header("Collaborative Filtering Recommendations")
    st.write("Get recommendations based on games you've enjoyed")
    
    if 'rated_games' not in st.session_state:
        st.session_state.rated_games = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Games You Like")
        
        # Game search
        search_term = st.text_input("Search for games:")
        if search_term:
            matching_games = metadata[metadata['name'].str.contains(search_term, case=False, na=False)]
            if not matching_games.empty:
                selected_game_name = st.selectbox("Select game:", matching_games['name'].tolist())
                rating = st.slider("Your rating (1-5):", 1, 5, 3)
                
                if st.button("Add Game"):
                    selected_game = metadata[metadata['name'] == selected_game_name]
                    st.session_state.rated_games.append((selected_game, rating))
                    st.success(f"Added {selected_game_name} with rating {rating}")
            else:
                st.write("No games found matching your search.")
    
    with col2:
        st.subheader("Your Rated Games")
        if st.session_state.rated_games:
            for i, (game, rating) in enumerate(st.session_state.rated_games):
                game_name = game['name'].iloc[0]
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(game_name)
                with col_b:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.rated_games.pop(i)
                        st.rerun()
            
            if st.button("Generate Recommendations", type="primary"):
                with st.spinner("Analyzing your preferences..."):
                    recommendations = generate_collaborative_recommendations(
                        st.session_state.rated_games, user_item_matrix, metadata
                    )
                
                st.success("Top recommendations for you:")
                for i, (appid, score) in enumerate(recommendations.head(10).items(), 1):
                    game_name = metadata[metadata['appid'] == appid]['name'].iloc[0]
                    st.write(f"{i}. {game_name} (Score: {score:.2f})")
        else:
            st.info("Add some games you've enjoyed to get recommendations!")

if __name__ == "__main__":
    main()
