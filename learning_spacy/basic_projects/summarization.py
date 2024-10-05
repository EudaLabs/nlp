from gensim.summarization import summarize
from sklearn.datasets import fetch_20newsgroups

# Fetch the 20 newsgroups dataset
categories = ['rec.autos', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

# Function to summarize document content
def summarize_document(text):
    try:
        return summarize(text)
    except ValueError:
        return text  # If the text is too short for summarization

# Generate summaries for the first 5 articles
for i in range(5):
    text = newsgroups_train.data[i]
    summary = summarize_document(text)
    print(f"Article {i+1} Summary:\n{summary}\n")
