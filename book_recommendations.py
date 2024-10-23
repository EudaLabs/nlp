import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import seaborn as sns
import requests
import os

# Download the dataset if it does not exist
# The dataset contains information about books, including title, author, and average ratings
if not os.path.exists('books.csv'):
    url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
    response = requests.get(url)
    with open('books.csv', 'wb') as file:
        file.write(response.content)

# Load the dataset into a pandas DataFrame
df = pd.read_csv('books.csv')

# Display the first 5 rows of the dataset to get an overview of the data
print(df.head())

# Print the column names to understand the available features in the dataset
print(df.columns)

# Plot the distribution of average ratings
# This will help visualize the spread of ratings across all books
plt.figure(figsize=(10, 5))
sns.histplot(df['average_rating'], bins=30, kde=True)
plt.title('Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Number of Books')
plt.show()

# Combine title, authors, and description into a single text field
# This combined content will be used to create book embeddings
df['content'] = df['title'] + ' ' + df['authors']

# Display a few rows to verify the combined content
print(df[['title', 'content']].head())

# Initialize the TfidfVectorizer
# TF-IDF is used to convert the text content into numerical vectors, emphasizing important words
tfidf = TfidfVectorizer(stop_words='english', max_features=2000)

# Transform the content field into a TF-IDF matrix
X_tfidf = tfidf.fit_transform(df['content'])

# Display the shape of the TF-IDF matrix
print(f"TF-IDF matrix shape: {X_tfidf.shape}")


def recommend_book_tfidf(title, df=df, X=X_tfidf):
    # Find the index of the given book title
    idx = df[df['title'].str.contains(title, case=False)].index[0]

    # Get the TF-IDF vector for the query book
    query = X[idx]

    # Calculate cosine similarity between the query book and all other books
    scores = cosine_similarity(query, X).flatten()

    # Get the indices of the top 5 most similar books (excluding the query book)
    recommended_idx = (-scores).argsort()[1:6]

    # Return the titles of the recommended books
    return df['title'].iloc[recommended_idx]


# Load pre-trained word embedding models
# These models are used to convert text into numerical vectors that represent the meaning of the text
word2vec = api.load("word2vec-google-news-300")
glove = api.load("glove-wiki-gigaword-100")
fasttext = api.load("fasttext-wiki-news-subwords-300")


def get_average_word_embedding(text, model):
    # Split the text into words and convert them to lowercase
    words = text.lower().split()
    # Extract embeddings for words that exist in the model's vocabulary
    valid_embeddings = [model[word] for word in words if word in model]
    # If no words are found in the model, return a zero vector
    if not valid_embeddings:
        return np.zeros(model.vector_size)
    # Calculate the average of all word embeddings to represent the entire text
    return np.mean(valid_embeddings, axis=0)


# Compute the word embeddings for each book using the Word2Vec model
df['word2vec_embedding'] = df['content'].apply(lambda x: get_average_word_embedding(x, word2vec))
# Compute the word embeddings for each book using the GloVe model
df['glove_embedding'] = df['content'].apply(lambda x: get_average_word_embedding(x, glove))
# Compute the word embeddings for each book using the FastText model
df['fasttext_embedding'] = df['content'].apply(lambda x: get_average_word_embedding(x, fasttext))


def recommend_book(title, model_name, df=df):
    # Select the embedding to use based on the model_name
    if model_name == 'word2vec':
        embeddings = np.vstack(df['word2vec_embedding'].values)
    elif model_name == 'glove':
        embeddings = np.vstack(df['glove_embedding'].values)
    elif model_name == 'fasttext':
        embeddings = np.vstack(df['fasttext_embedding'].values)
    else:
        # Raise an error if an invalid model name is provided
        raise ValueError("Invalid model name. Choose from 'word2vec', 'glove', or 'fasttext'.")

    # Find the index of the given book title (case-insensitive search)
    idx = df[df['title'].str.contains(title, case=False)].index[0]

    # Get the embedding for the query book
    query = embeddings[idx].reshape(1, -1)

    # Calculate cosine similarity between the query book and all other books
    # Cosine similarity measures the angle between two vectors, indicating their similarity
    scores = cosine_similarity(query, embeddings).flatten()

    # Get the indices of the top 5 most similar books (excluding the query book itself)
    recommended_idx = (-scores).argsort()[1:6]

    # Return the titles of the recommended books
    return df['title'].iloc[recommended_idx]


# Get recommendations for 'The Hobbit' using TF-IDF
recommendations_tfidf = recommend_book_tfidf('The Hobbit')
print(f"Recommended books for 'The Hobbit' using TF-IDF:")
print(recommendations_tfidf)

# Get recommendations for 'The Hobbit' using Word2Vec
recommendations_word2vec = recommend_book('The Hobbit', 'word2vec')
print(f"Recommended books for 'The Hobbit' using Word2Vec:")
print(recommendations_word2vec)

# Get recommendations for 'The Hobbit' using GloVe
recommendations_glove = recommend_book('The Hobbit', 'glove')
print(f"Recommended books for 'The Hobbit' using GloVe:")
print(recommendations_glove)

# Get recommendations for 'The Hobbit' using FastText
recommendations_fasttext = recommend_book('The Hobbit', 'fasttext')
print(f"Recommended books for 'The Hobbit' using FastText:")
print(recommendations_fasttext)
