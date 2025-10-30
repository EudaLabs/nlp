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
import pickle
from pathlib import Path

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


# Load pre-trained word embedding models with caching to avoid re-downloading
# These models are used to convert text into numerical vectors that represent the meaning of the text
cache_dir = Path('model_cache')
cache_dir.mkdir(exist_ok=True)

def load_or_cache_model(model_name, cache_path):
    """Load model from cache if available, otherwise download and cache it"""
    if cache_path.exists():
        print(f"Loading {model_name} from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Downloading {model_name}... (this may take a while on first run)")
        model = api.load(model_name)
        print(f"Caching {model_name} for future use...")
        with open(cache_path, 'wb') as f:
            pickle.dump(model, f)
        return model

word2vec = load_or_cache_model("word2vec-google-news-300", cache_dir / 'word2vec.pkl')
glove = load_or_cache_model("glove-wiki-gigaword-100", cache_dir / 'glove.pkl')
fasttext = load_or_cache_model("fasttext-wiki-news-subwords-300", cache_dir / 'fasttext.pkl')


def get_average_word_embedding(text, model):
    """Optimized function to compute average word embeddings"""
    # Split the text into words and convert them to lowercase
    words = text.lower().split()
    # Extract embeddings for words that exist in the model's vocabulary
    # Using list comprehension is faster than a loop
    valid_embeddings = [model[word] for word in words if word in model]
    # If no words are found in the model, return a zero vector
    if not valid_embeddings:
        return np.zeros(model.vector_size)
    # Calculate the average of all word embeddings to represent the entire text
    # Using np.mean with axis=0 is more efficient than manual averaging
    return np.mean(valid_embeddings, axis=0)


# Compute the word embeddings for each book using vectorized operations
# Using apply() is still needed but we optimize by computing all embeddings once
# and storing them for reuse
print("Computing embeddings for all books...")
print("Computing Word2Vec embeddings...")
df['word2vec_embedding'] = df['content'].apply(lambda x: get_average_word_embedding(x, word2vec))
print("Computing GloVe embeddings...")
df['glove_embedding'] = df['content'].apply(lambda x: get_average_word_embedding(x, glove))
print("Computing FastText embeddings...")
df['fasttext_embedding'] = df['content'].apply(lambda x: get_average_word_embedding(x, fasttext))
print("Embeddings computed successfully!")


# Pre-compute embedding matrices once for better performance
# This avoids re-stacking embeddings on every recommendation call
print("Pre-computing embedding matrices for fast recommendations...")
word2vec_embeddings = np.vstack(df['word2vec_embedding'].values)
glove_embeddings = np.vstack(df['glove_embedding'].values)
fasttext_embeddings = np.vstack(df['fasttext_embedding'].values)

def recommend_book(title, model_name, df=df):
    """Optimized recommendation function using pre-computed embeddings"""
    # Select the pre-computed embedding matrix based on the model_name
    if model_name == 'word2vec':
        embeddings = word2vec_embeddings
    elif model_name == 'glove':
        embeddings = glove_embeddings
    elif model_name == 'fasttext':
        embeddings = fasttext_embeddings
    else:
        # Raise an error if an invalid model name is provided
        raise ValueError("Invalid model name. Choose from 'word2vec', 'glove', or 'fasttext'.")

    # Find the index of the given book title (case-insensitive search)
    idx = df[df['title'].str.contains(title, case=False)].index[0]

    # Get the embedding for the query book (no need to reshape for single query)
    query = embeddings[idx:idx+1]

    # Calculate cosine similarity between the query book and all other books
    # Cosine similarity measures the angle between two vectors, indicating their similarity
    scores = cosine_similarity(query, embeddings).flatten()

    # Get the indices of the top 5 most similar books (excluding the query book itself)
    # Using argsort is more efficient than sorting the entire array
    recommended_idx = np.argsort(-scores)[1:6]

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

# Output:
#    book_id  ...                                    small_image_url
# 0        1  ...  https://images.gr-assets.com/books/1447303603s...
# 1        2  ...  https://images.gr-assets.com/books/1474154022s...
# 2        3  ...  https://images.gr-assets.com/books/1361039443s...
# 3        4  ...  https://images.gr-assets.com/books/1361975680s...
# 4        5  ...  https://images.gr-assets.com/books/1490528560s...
#
# [5 rows x 23 columns]
# Index(['book_id', 'goodreads_book_id', 'best_book_id', 'work_id',
#        'books_count', 'isbn', 'isbn13', 'authors', 'original_publication_year',
#        'original_title', 'title', 'language_code', 'average_rating',
#        'ratings_count', 'work_ratings_count', 'work_text_reviews_count',
#        'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
#        'image_url', 'small_image_url'],
#       dtype='object')
#                                                title                                            content
# 0            The Hunger Games (The Hunger Games, #1)  The Hunger Games (The Hunger Games, #1) Suzann...
# 1  Harry Potter and the Sorcerer's Stone (Harry P...  Harry Potter and the Sorcerer's Stone (Harry P...
# 2                            Twilight (Twilight, #1)            Twilight (Twilight, #1) Stephenie Meyer
# 3                              To Kill a Mockingbird                   To Kill a Mockingbird Harper Lee
# 4                                   The Great Gatsby               The Great Gatsby F. Scott Fitzgerald
# TF-IDF matrix shape: (10000, 2000)
# [==================================================] 100.0% 1662.8/1662.8MB downloaded
# [==================================================] 100.0% 128.1/128.1MB downloaded
# [==================================================] 100.0% 958.5/958.4MB downloaded
# Recommended books for 'The Hobbit' using TF-IDF:
# 2308                                The Children of Húrin
# 4975         Unfinished Tales of Númenor and Middle-Earth
# 610              The Silmarillion (Middle-Earth Universe)
# 963     J.R.R. Tolkien 4-Book Boxed Set: The Hobbit an...
# 154            The Two Towers (The Lord of the Rings, #2)
# Name: title, dtype: object
# Recommended books for 'The Hobbit' using Word2Vec:
# 963     J.R.R. Tolkien 4-Book Boxed Set: The Hobbit an...
# 3988    The Dark Elf Trilogy Collector's Edition (Forg...
# 4011    Buffy the Vampire Slayer: The Long Way Home (S...
# 7899                                   The Goblin Emperor
# 8054    The Frog and Toad Treasury: Frog and Toad are ...
# Name: title, dtype: object
# Recommended books for 'The Hobbit' using GloVe:
# 963     J.R.R. Tolkien 4-Book Boxed Set: The Hobbit an...
# 4975         Unfinished Tales of Númenor and Middle-Earth
# 8271                   The Complete Guide to Middle-Earth
# 610              The Silmarillion (Middle-Earth Universe)
# 154            The Two Towers (The Lord of the Rings, #2)
# Name: title, dtype: object
# Recommended books for 'The Hobbit' using FastText:
# 963     J.R.R. Tolkien 4-Book Boxed Set: The Hobbit an...
# 9092    The Lightning Thief: The Graphic Novel (Percy ...
# 6140    Harry Potter and the Order of the Phoenix (Har...
# 6245    The Crippled God (The Malazan Book of the Fall...
# 8698              The Complete Adventures of Peter Rabbit
# Name: title, dtype: object