from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Sample corpus
corpus = [
    "Natural language processing is a subfield of artificial intelligence.",
    "Word embeddings are used to represent words in a vector space.",
    "Machine learning and deep learning are popular in NLP.",
    "Transformers are state-of-the-art models for many NLP tasks.",
    "GloVe and Word2Vec are two popular word embedding techniques."
]

# Preprocess the corpus by tokenizing each sentence
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train a Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=3, min_count=1, workers=4)
# Parameters:
# - `sentences=tokenized_corpus`: The input corpus of tokenized sentences to train the Word2Vec model.
# - `vector_size=50`: The dimensionality of the word embeddings. Higher values allow more nuanced representations but require more computation.
# - `window=3`: The maximum distance between the current and predicted word within a sentence. A larger window size means the model takes a wider context into account.
# - `min_count=1`: The minimum number of occurrences a word must have to be included in the training. Words appearing less than this value are ignored.
# - `workers=4`: The number of CPU cores to use for training. Using more workers can speed up training.


# Test the model by finding similar words
similar_words = model.wv.most_similar("nlp", topn=5)
print("Words similar to 'nlp':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.2f}")

# Get the embedding for a specific word
word_vector = model.wv['nlp']
print("\nVector representation of 'nlp':")
print(word_vector)