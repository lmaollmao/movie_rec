
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load datasets
def load_data(ratings_file, movies_file):
    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)
    data = pd.merge(ratings, movies, on='movieId')
    return data, ratings, movies

# Create user-item interaction matrix
def create_user_item_matrix(data):
    return data.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Collaborative Filtering - User Similarity Matrix
def calculate_user_similarity(user_item_matrix):
    user_similarity = cosine_similarity(user_item_matrix)
    np.fill_diagonal(user_similarity, 0)
    return user_similarity

# Predict ratings based on similarity matrix
def predict_ratings(user_similarity, user_item_matrix):
    return user_similarity.dot(user_item_matrix) / np.array([np.abs(user_similarity).sum(axis=1)]).T

# Matrix Factorization using SVD
def apply_svd(user_item_matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components)
    reduced_matrix = svd.fit_transform(user_item_matrix)
    reconstructed_matrix = np.dot(reduced_matrix, svd.components_)
    return reconstructed_matrix

# Recommend top N movies for a user
def recommend_movies(user_id, user_item_matrix, reconstructed_matrix, movies, top_n=5):
    user_idx = user_id - 1  # Adjusting for zero-based index
    predicted_ratings = reconstructed_matrix[user_idx]
    movie_ids = np.argsort(predicted_ratings)[::-1][:top_n]
    return movies.iloc[movie_ids]

# Evaluate model using RMSE
def evaluate_model(test, predicted_ratings):
    return np.sqrt(mean_squared_error(test['rating'], predicted_ratings[test['movieId']]))

# Main function
def main():
    # Load data
    data, ratings, movies = load_data('ratings.csv', 'movies.csv')
    
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(data)
    
    # Collaborative Filtering
    user_similarity = calculate_user_similarity(user_item_matrix)
    predicted_ratings_cf = predict_ratings(user_similarity, user_item_matrix)
    
    # Matrix Factorization
    reconstructed_matrix = apply_svd(user_item_matrix, n_components=50)
    
    # Get recommendations for a user
    user_id = 1  # Example user
    recommendations = recommend_movies(user_id, user_item_matrix, reconstructed_matrix, movies, top_n=5)
    print("Top 5 Recommendations for User", user_id)
    print(recommendations)
    
    # Evaluation
    train, test = train_test_split(ratings, test_size=0.2)
    rmse = evaluate_model(test, predicted_ratings_cf)
    print(f"RMSE: {rmse}")

if __name__ == "__main__":
    main()
