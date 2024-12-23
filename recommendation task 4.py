import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

data = {
    'User': ['kanniga', 'divya', 'swetha', 'ashika', 'mahathi', 'yagavi'],
    'A thrilling adventure movie': [5, 4, np.nan, 3, np.nan, 4],
    'A romantic comedy film': [4, np.nan, 4, 2, 1, 5],
    'A science fiction novel': [np.nan, 5, 3, 2, 1, np.nan],
    'A historical drama film': [2, 3, np.nan, 5, 4, 3],
    'A horror film': [1, np.nan, 2, np.nan, 5, 4]
}

df = pd.DataFrame(data)
df.set_index('User', inplace=True)

scaler = MinMaxScaler()
normalized_df = pd.DataFrame(
    scaler.fit_transform(df.fillna(0)),
    index=df.index,
    columns=df.columns
)

similarity_matrix = cosine_similarity(normalized_df)
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=df.index,
    columns=df.index
)

def predict_ratings(user, item):
    if np.isnan(df.loc[user, item]):  
        similar_users = similarity_df[user]
        ratings = df[item]
        weighted_sum = np.dot(similar_users, ratings.fillna(0))
        normalization_factor = np.sum(similar_users)
        return weighted_sum / normalization_factor if normalization_factor != 0 else 0
    return df.loc[user, item]  

def recommend_items(user, top_n=3):
    items = df.columns
    predicted_ratings = {item: predict_ratings(user, item) for item in items}
    recommended_items = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
    return [item for item, rating in recommended_items[:top_n]]

user = 'kanniga'
recommended_movies = recommend_items(user, top_n=3)
print(f"{recommended_movies} is the top pick for {user}.")
