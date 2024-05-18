from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Load the dataset
df = pd.read_csv('modified_coursea_data.csv')

# Accept user input for preferences
user_preferences = input("Enter your preferences: ")

# Combine course titles and categories into a single text
course_data = df['course_title'] + " " + df['category']

# Initialize TF-IDF vectorizer and transform course data into feature vectors
tfidf_vectorizer = TfidfVectorizer()
course_features = tfidf_vectorizer.fit_transform(course_data)

# Train a Nearest Neighbors model using all the data
model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
model.fit(course_features)

# Transform user preferences into a feature vector
user_vector = tfidf_vectorizer.transform([user_preferences])

# Find the nearest neighbors (similar courses) to the user preferences
distances, indices = model.kneighbors(user_vector)

# Get the titles of the recommended courses
recommended_courses = df.iloc[indices[0]]['course_title']
print("Recommended Courses:")
for course in recommended_courses:
    print(course)
