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