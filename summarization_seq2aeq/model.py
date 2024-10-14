import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention
from tensorflow.keras.models import Model



def build_model(article_vocab_size, summary_vocab_size, max_article_len, max_summary_len):
    # Encoder
    encoder_inputs = Input(shape=(max_article_len,))
    enc_emb = Embedding(article_vocab_size, 128, trainable=True)(encoder_inputs)

    encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]