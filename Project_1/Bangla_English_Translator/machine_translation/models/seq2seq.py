import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model

# Define Encoder
def build_encoder(vocab_size, embedding_dim, hidden_units):
    encoder_inputs = Input(shape=(None,))
    embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(hidden_units, return_state=True)
    _, state_h, state_c = encoder_lstm(embedding)
    
    encoder_states = [state_h, state_c]
    return Model(encoder_inputs, encoder_states), encoder_inputs

# Define Decoder
def build_decoder(vocab_size, embedding_dim, hidden_units, encoder_states):
    decoder_inputs = Input(shape=(None,))
    embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    return Model(decoder_inputs, decoder_outputs)

# Build full Seq2Seq model
def build_seq2seq_model(vocab_size_en, vocab_size_bn, embedding_dim=256, hidden_units=512):
    encoder, encoder_inputs = build_encoder(vocab_size_en, embedding_dim, hidden_units)
    encoder_states = encoder(encoder_inputs)

    decoder = build_decoder(vocab_size_bn, embedding_dim, hidden_units, encoder_states)
    
    model = Model([encoder_inputs, decoder.input], decoder.output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model
