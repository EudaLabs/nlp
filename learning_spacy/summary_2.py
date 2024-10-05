import spacy

# Load the SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = """
Apple is set to acquire U.K. startup for $1 billion. CEO Tim Cook announced this during the annual conference on September 15, 2023.
Microsoft is also planning to expand its business into Europe.
"""

# Process the text using SpaCy
doc = nlp(text)

# Print out named entities
for ent in doc.ents:
    print(ent.text, ent.label_)


def summarize_text(doc):
    summary = {"Persons": [], "Organizations": [], "Dates": []}

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            summary["Persons"].append(ent.text)
        elif ent.label_ == "ORG":
            summary["Organizations"].append(ent.text)
        elif ent.label_ == "DATE":
            summary["Dates"].append(ent.text)

    return summary


# Generate summary from text
summary = summarize_text(doc)
print(summary)
