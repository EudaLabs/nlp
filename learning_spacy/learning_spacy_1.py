import spacy

nlp = spacy.load('en_core_web_sm')

text = "I have big dreams and I am going to make them come true."
doc = nlp(text)

for token in doc:
    print(f"Word: {token.text}, Type: {token.pos_}")