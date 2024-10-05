from sklearn.datasets import fetch_20newsgroups

# Fetch the 20 newsgroups dataset
categories = ['rec.autos', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# Let's inspect the dataset
print(f"Training set size: {len(newsgroups_train.data)}")
print(f"Test set size: {len(newsgroups_test.data)}")
print(f"Categories: {newsgroups_train.target_names}")
