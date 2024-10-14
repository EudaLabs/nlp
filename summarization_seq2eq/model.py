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

    # Decoder
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(summary_vocab_size, 128, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # Attention Layer
    attn_layer = Attention()
    attn_out = attn_layer([decoder_outputs, encoder_outputs])

    # Dense Layer
    decoder_dense = Dense(summary_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(attn_out)

    # Create Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model