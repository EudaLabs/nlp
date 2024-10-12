from sklearn.feature_extraction.text import CountVectorizer

# Sample sentences
sentences = [
    "I love natural language processing.",
    "Language models are fascinating.",
    "I love building NLP projects."
]

# Creating a CountVectorizer instance
vectorizer = CountVectorizer()

# Transforming the sentences to a bag-of-words representation
bag_of_words = vectorizer.fit_transform(sentences)

# Display the vocabulary and the bag-of-words array
print("Vocabulary (Unique Words):", vectorizer.get_feature_names_out())
print("Bag of Words Array (Representation of Sentences):")
print(bag_of_words.toarray())
#Sample output:
# Vocabulary (Unique Words): ['are' 'building' 'fascinating' 'language' 'love' 'models' 'natural' 'nlp'
#  'processing' 'projects']
# Bag of Words Array (Representation of Sentences):
# [[0 0 0 1 1 0 1 0 1 0]
#  [1 0 1 1 0 1 0 0 0 0]
#  [0 1 0 0 1 0 0 1 0 1]]