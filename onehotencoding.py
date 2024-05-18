import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import re

# Load the dataset
try:
    df = pd.read_csv('coursea_data_with_categories.csv')
except FileNotFoundError:
    print("The file 'coursea_data.csv' was not found.")
    exit()


# Tokenize Course Titles and Filter Out Stop Words
stop_words = set([
    'course', 'the', 'to', 'on', 'en', 'of', 'introduction', 'platform', 
    'thinking', 'understanding', 'systems', 'platforms', 'your', 'and', 
    'in', 'with', 'for', 'a', 'learning', 'fundamentals', 'google', 'de', 
    '&', 'global', 'international', 'foundations', 'y', 'skills', 'la', 
    'public', 'from'
])  # Define stop words

tokens = ' '.join(df['course_title'].str.lower()).split()
tokens_filtered = [token for token in tokens if token not in stop_words]

# Count Keyword Frequencies
keyword_counter = Counter(tokens_filtered)

# Select Top Keywords
num_top_keywords = 25  # Adjust as needed
top_keywords = [keyword for keyword, _ in keyword_counter.most_common(num_top_keywords)]

# Assign Categories
def assign_category(title):
    for keyword in top_keywords:
        if keyword in title.lower():
            return keyword
    return 'Other'

df['category'] = df['course_title'].apply(assign_category)

# Display the top keywords and their frequencies
print("Top Keywords:")
for keyword, frequency in keyword_counter.most_common(num_top_keywords):
    print(f"{keyword}: {frequency}")

# Display the first few rows of the DataFrame with categories
print("\nData with Categories:")
print(df[['course_title', 'category']].head())

# Display the first few rows of the original DataFrame for debugging
print("\nOriginal DataFrame:")
print(df.head())

# Clean 'course_title' column
df['course_title'] = df['course_title'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))  # Remove special characters and punctuation
df['course_title'] = df['course_title'].str.lower()  # Convert text to lowercase


df['course_organization'] = df['course_organization'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))  # Remove special characters and punctuation
df['course_organization'] = df['course_organization'].str.lower()  # Convert text to lowercase
df['course_difficulty'] = df['course_difficulty'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))  # Remove special characters and punctuation
df['course_difficulty'] = df['course_difficulty'].str.lower()  # Convert text to lowercase

df['category'] = df['category'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))  # Remove special characters and punctuation
df['category'] = df['category'].str.lower()  # Convert text to lowercase

# Function to convert 'k' and 'm' suffixes to numeric values
def convert_enrollment(value):
    value = value.replace(',', '')
    match = re.match(r'(\d+(?:\.\d+)?)([km])', value, re.IGNORECASE)
    if match:
        number, suffix = match.groups()
        number = float(number)
        if suffix.lower() == 'k':
            return number * 1000
        elif suffix.lower() == 'm':
            return number * 1000000
    return float(value)

# Clean 'course_students_enrolled' column
df['course_students_enrolled'] = df['course_students_enrolled'].apply(convert_enrollment)

# Replace unrealistic values (less than 10) with a minimum value of 10
df.loc[df['course_students_enrolled'] < 10, 'course_students_enrolled'] = 10

# Display the DataFrame after cleaning for debugging
print("\nDataFrame after cleaning:")
print(df.head())

# Remove unnecessary columns
df = df.drop(['course_rating', 'course_Certificate_type'], axis=1)

# Display the final DataFrame for debugging
print("\nFinal DataFrame:")
print(df.head())

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_coursea_data.csv', index=False)

# Confirm saving by reading the saved file and displaying its content
try:
    df_saved = pd.read_csv('modified_coursea_data.csv')
    print("\nSaved DataFrame:")
    print(df_saved.head())
except FileNotFoundError:
    print("The file 'modified_coursea_data.csv' was not saved correctly.")
