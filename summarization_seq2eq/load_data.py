import os
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub('"', '', text)
    text = re.sub(r"'s\b", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join(text.split())
    return text


def load_stories(directory):
    articles = []
    summaries = []

    for filename in os.listdir(directory):
        if filename.endswith(".story"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                article, highlights = text.split("\n\n@highlight\n\n", 1)
                articles.append(clean_text(article))
                summaries.append(clean_text(highlights))

    return articles, summaries


def tokenize_and_pad(articles, summaries, max_article_len, max_summary_len):
    article_tokenizer = Tokenizer()
    article_tokenizer.fit_on_texts(articles)

    summary_tokenizer = Tokenizer()
    summary_tokenizer.fit_on_texts(summaries)

    article_seq = article_tokenizer.texts_to_sequences(articles)
    summary_seq = summary_tokenizer.texts_to_sequences(summaries)

    article_pad = pad_sequences(article_seq, maxlen=max_article_len, padding='post')
    summary_pad = pad_sequences(summary_seq, maxlen=max_summary_len, padding='post')

    return article_pad, summary_pad, article_tokenizer, summary_tokenizer


def load_and_preprocess_data(stories_dir, max_article_len, max_summary_len):
    articles, summaries = load_stories(stories_dir)
    return tokenize_and_pad(articles, summaries, max_article_len, max_summary_len)
