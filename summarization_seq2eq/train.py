import numpy as np
from load_data import load_and_preprocess_data
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint


max_article_len = 400
max_summary_len = 100

article_pad, summary_pad, article_tokenizer, summary_tokenizer, train_dataset, val_dataset = load_and_preprocess_data(max_article_len, max_summary_len)

model = build_model(len(article_tokenizer.word_index)+1, len(summary_tokenizer.word_index)+1, max_article_len, max_summary_len)


#Train model
checkpoint = ModelCheckpoint('model_weights.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit([article_pad, summary_pad[:, :-1]],
                    summary_pad.reshape(summary_pad.shape[0], summary_pad.shape[1], 1)[:, 1:],
                    epochs=10,
                    batch_size=64,
                    validation_split=0.2,
                    callbacks=[checkpoint])

model.save('summarization_model.h5')
