import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download necessary NLTK data files
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Sample words
words = ["running", "ate", "better", "geese"]

# Lemmatize each word
lemmatized_words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words]

# Print the original and lemmatized words
print("Original words:", words)
print("Lemmatized words:", lemmatized_words)
#Output:
# Original words: ['running', 'ate', 'better', 'geese']
# Lemmatized words: ['run', 'eat', 'better', 'geese']