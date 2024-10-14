import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub('"', '', text)
    text = re.sub(r"'s\b", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


def load_and_preprocess_data(max_article_len, max_summary_len):
    dataset, info = tfds.load('cnn_dailymail', version='3.0.0', with_info=True)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    train_articles = []
    train_summaries = []

    for example in train_dataset.take(2000):
        article = example['article'].numpy().decode('utf-8')
        summary = example['highlights'].numpy().decode('utf-8')
        train_articles.append(clean_text(article))
        train_summaries.append(clean_text(summary))

    article_tokenizer = Tokenizer()
    article_tokenizer.fit_on_texts(train_articles)

    summary_tokenizer = Tokenizer()
    summary_tokenizer.fit_on_texts(train_summaries)

    article_seq = article_tokenizer.texts_to_sequences(train_articles)
    summary_seq = summary_tokenizer.texts_to_sequences(train_summaries)

    article_pad = pad_sequences(article_seq, maxlen=max_article_len, padding='post')
    summary_pad = pad_sequences(summary_seq, maxlen=max_summary_len, padding='post')

    return article_pad, summary_pad, article_tokenizer, summary_tokenizer, train_dataset, val_dataset
