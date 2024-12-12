import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Prepare the data
# This data created by AI. It is not real data.
data = {
    "text": [
        "I love this product!",
        "This is the worst thing I bought.",
        "Absolutely fantastic service.",
        "I hate this place.",
        "Not bad, could be better.",
        "It's amazing, I will buy again!",
        "Terrible experience, not recommended.",
    ],
    "sentiment": [1, 0, 1, 0, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# 2. Convert text to numerical representation (TF-IDF)
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(df['text'])
y = df['sentiment']

# 3. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Test the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict sentiment for a new sentence
new_text = ["I really enjoy this service!"]
new_text_tfidf = tfidf.transform(new_text)
prediction = model.predict(new_text_tfidf)
print(f"Prediction for '{new_text[0]}':", "Positive" if prediction[0] == 1 else "Negative")
