import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

class LogisticRegressionNLP:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.vectorizer = CountVectorizer()
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X_text, y):
        # Convert text data to numerical features using bag of words
        X = self.vectorizer.fit_transform(X_text).toarray()
        
        # Initialize parameters
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            # Forward propagation
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Calculate gradients
            dw = (1/X.shape[0]) * np.dot(X.T, (predictions - y))
            db = (1/X.shape[0]) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X_text, threshold=0.5):
        # Convert text to numerical features using the fitted vectorizer
        X = self.vectorizer.transform(X_text).toarray()
        
        # Make predictions
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        
        # Convert probabilities to binary predictions
        return (predictions >= threshold).astype(int)
    
    def predict_proba(self, X_text):
        # Convert text to numerical features using the fitted vectorizer
        X = self.vectorizer.transform(X_text).toarray()
        
        # Calculate probabilities
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

# Real usage implementation
if __name__ == "__main__":
    # Create sample text data
    texts = [
        "This movie was fantastic! I loved every minute of it",
        "Terrible waste of time, worst movie ever",
        "Great acting and wonderful storyline",
        "I fell asleep during this boring movie",
        "Amazing special effects and great plot",
        "Disappointing and predictable, would not recommend"
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, 
                                                        test_size=0.2, 
                                                        random_state=42)

    # Initialize and train the model
    model = LogisticRegressionNLP(learning_rate=0.1, num_iterations=1000)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Print results
    print("\nPrediction Results:")
    for text, pred, prob in zip(X_test, predictions, probabilities):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"\nText: {text}")
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Probability: {prob:.3f}")

