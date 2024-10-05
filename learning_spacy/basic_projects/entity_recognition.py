import spacy
from sklearn.datasets import fetch_20newsgroups

# Ignore SSL certificate verification
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context
#
# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Fetch the 20 newsgroups dataset
categories = ['rec.autos', 'rec.sport.baseball', 'sci.electronics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)


# Function to extract entities from text
def extract_entities(text):
    doc = nlp(text)
    entities = {"Persons": [], "Organizations": [], "Dates": [], "Locations": []}

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["Persons"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["Organizations"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["Dates"].append(ent.text)
        elif ent.label_ == "GPE":
            entities["Locations"].append(ent.text)

    return entities


# Extract entities for the first 5 articles
for i in range(5):
    text = newsgroups_train.data[i]
    entities = extract_entities(text)
    print(f"Article {i + 1}:\nEntities: {entities}\n")
