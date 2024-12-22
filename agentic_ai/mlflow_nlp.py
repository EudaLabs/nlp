import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

def log_nlp_experiment(model_name, text_data, true_labels, predicted_labels, params=None):
    """
    Log NLP experiment results to MLflow
    
    Args:
        model_name: Name of the NLP model
        text_data: Input text data
        true_labels: Ground truth labels
        predicted_labels: Model predictions
        params: Optional model parameters to log
    """
    
    with mlflow.start_run(run_name=model_name):
        # Log parameters if provided
        if params:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
                
        # Calculate and log metrics
        metrics = {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "precision": precision_score(true_labels, predicted_labels, average='weighted'),
            "recall": recall_score(true_labels, predicted_labels, average='weighted'),
            "f1": f1_score(true_labels, predicted_labels, average='weighted')
        }
        
        mlflow.log_metrics(metrics)
        
        # Log text analysis metrics
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(text_data)
        
        text_metrics = {
            "vocabulary_size": len(vectorizer.get_feature_names_out()),
            "sparsity": 1.0 - (bow_matrix.nnz / float(bow_matrix.shape[0] * bow_matrix.shape[1]))
        }
        
        mlflow.log_metrics(text_metrics)
        
        # Log model if it supports MLflow model format
        try:
            mlflow.sklearn.log_model(model, f"{model_name}_model")
        except:
            pass
            
        return metrics, text_metrics

def compare_experiment_runs(experiment_name, top_n=5):
    """
    Compare different MLflow runs within an experiment
    
    Args:
        experiment_name: Name of the MLflow experiment to analyze
        top_n: Number of top runs to compare
    """
    
    # Get experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
        
    # Get all runs for the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if len(runs) == 0:
        print("No runs found in experiment")
        return
        
    # Sort runs by accuracy
    runs_sorted = runs.sort_values("metrics.accuracy", ascending=False)
    top_runs = runs_sorted.head(top_n)
    
    print(f"\nTop {top_n} runs for experiment '{experiment_name}':")
    print("\nMetrics comparison:")
    metrics = ["accuracy", "precision", "recall", "f1"]
    
    for metric in metrics:
        print(f"\n{metric.capitalize()}:")
        for idx, run in top_runs.iterrows():
            run_name = run["tags.mlflow.runName"]
            metric_value = run[f"metrics.{metric}"]
            print(f"{run_name}: {metric_value:.4f}")
            
    print("\nText analysis metrics:")
    text_metrics = ["vocabulary_size", "sparsity"]
    
    for metric in text_metrics:
        print(f"\n{metric.capitalize()}:")
        for idx, run in top_runs.iterrows():
            run_name = run["tags.mlflow.runName"]
            metric_value = run[f"metrics.{metric}"]
            print(f"{run_name}: {metric_value:.4f}")

            # Example practice in main
if __name__ == "__main__":
    experiment_name = "NLP_Experiment"
    
    # Create and set the experiment
    mlflow.set_experiment(experiment_name)
    
    # Sample data for demonstration
    texts = [
        "I love this product, it's amazing",
        "This is terrible, worst purchase ever",
        "Great customer service, very helpful",
        "Don't waste your money on this",
        "Excellent quality and fast delivery",
    ]
    labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

    # Create a simple model
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Create and train model
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)

    # Log the experiment
    metrics, text_metrics = log_nlp_experiment(
        model_name="NaiveBayes_Sentiment",
        text_data=texts,
        true_labels=y_test,
        predicted_labels=predictions,
        params={'model_type': 'naive_bayes'}
    )

    # Now compare the runs
    try:
        compare_experiment_runs(experiment_name, top_n=3)
    except ValueError as e:
        print(e)


