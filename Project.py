from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

movies_data = {
    'Title': [
        "Avengers Endgame", "The Amazing Spider Man 2", "Transformers", 
        "Demon Slayer Infinity Castle", "Jujutsu Kaisen 0", "Your Name", 
        "The Batman"
    ],
    'Action': [9.5, 7.0, 9.0, 8.5, 8.0, 2.0, 6.5],
    'Comedy': [7.0, 5.0, 4.0, 2.0, 1.5, 6.0, 1.0],
    'Drama':  [7.5, 5.0, 3.0, 8.0, 6.5, 9.0, 9.5],
    'Sci-Fi': [8.0, 6.0, 8.5, 3.0, 2.0, 7.0, 1.5]
}

df = pd.DataFrame(movies_data)
feature = ['Action', 'Comedy', 'Drama', 'Sci-Fi']
movie_features = df[feature].values
titles = df['Title'].tolist()
features_list = movie_features.tolist()

k = 3
model = NearestNeighbors(n_neighbors=k, metric='euclidean')
model.fit(movie_features)

print("Enter Your Movie Preference (0-10)")

user_action = float(input("Action Score: "))
user_comedy = float(input("Comedy Score: "))
user_drama = float(input("Drama Score: "))
user_scifi = float(input("Sci-Fi Score: "))
user_rating = np.array([[user_action, user_comedy, user_drama, user_scifi]])

distances, indices = model.kneighbors(user_rating)
recommended_indices = indices[0]
recommended_distances = distances[0]

print("Top 3 Movie Recommendations")
for i in range(k):
    index = recommended_indices[i]
    distance = recommended_distances[i]
    movie_title = titles[index]
    movie_features_list = features_list[index]
    features_number = [round(f, 1) for f in features_list[index]]
    feature_string = str(features_number)
    similarity = 10 - distance 
    if similarity < 0: 
        similarity = 0
    
    print(f"\n{i+1}. {movie_title}")
    print(f"Similarity Score: {similarity:.1f}")
    print(f" Features: {feature_string}")
#hello
