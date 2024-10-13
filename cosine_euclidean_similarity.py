from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Define sentences
sentences = [
    "Cats are very cute animals.",
    "Cats and dogs are the most popular pets.",
    "Dogs are our loyal friends."
]

# Vectorize sentences (Bag of Words method)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Display vectors
print("\nBag of Words Vectors:")
print(X.toarray())

# Calculate Cosine Similarity
cosine_sim = cosine_similarity(X, X)
print("\nCosine Similarity Results:")
print(cosine_sim)

# Calculate Euclidean Distance
euclidean_dist = euclidean_distances(X, X)
print("\nEuclidean Distance Results:")
print(euclidean_dist)


#Output:
# Bag of Words Vectors:
# [[0 1 1 1 1 0 0 0 0 0 0 0 0 1]
#  [1 0 1 1 0 1 0 0 1 0 1 1 1 0]
#  [0 0 1 0 0 1 1 1 0 1 0 0 0 0]]
#
# Cosine Similarity Results:
# [[1.         0.31622777 0.2       ]
#  [0.31622777 1.         0.31622777]
#  [0.2        0.31622777 1.        ]]
#
# Euclidean Distance Results:
# [[0.         3.         2.82842712]
#  [3.         0.         3.        ]
#  [2.82842712 3.         0.        ]]
