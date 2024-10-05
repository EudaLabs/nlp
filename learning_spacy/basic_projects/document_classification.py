from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups

# Fetch the 20 newsgroups dataset
categories = ['rec.autos', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# Prepare text data and their labels
texts = newsgroups_train.data
categories = newsgroups_train.target

# Convert the text data into TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.2, random_state=42)

# Create and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test data
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Classification Accuracy: {accuracy}")
