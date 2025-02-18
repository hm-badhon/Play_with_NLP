import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import re
import os
import io
import unicodedata
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from nltk.translate.bleu_score import corpus_bleu

# Load dataset
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    pairs = [line.strip().split('\t')[:2] for line in lines if '\t' in line]
    bn_sentences, en_sentences = zip(*pairs)
    return list(bn_sentences), list(en_sentences)

# Preprocessing
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^\u0980-\u09FFa-zA-Z!?.,' ]+", "", sentence)
    return sentence

def preprocess_data(bn_sentences, en_sentences):
    bn_sentences = [preprocess_sentence(sent) for sent in bn_sentences]
    en_sentences = [preprocess_sentence(sent) for sent in en_sentences]
    return bn_sentences, en_sentences

# Load & preprocess data
file_path = '/media/nsl47/hdd/Robotics_for_kids/Updated/New/For_client/Final/Materials/Others/Play_with_NLP/Project_1/Bangla_English_Translator/machine_translation/data/dataset.txt'  # Change this if needed
bn_sentences, en_sentences = load_data(file_path)
bn_sentences, en_sentences = preprocess_data(bn_sentences, en_sentences)

# Split dataset
train_bn, val_bn, train_en, val_en = train_test_split(bn_sentences, en_sentences, test_size=0.2, random_state=42)

# Tokenization
max_vocab_size = 10000
max_length = 20
vectorizer_bn = TextVectorization(max_tokens=max_vocab_size, output_mode='int', output_sequence_length=max_length)
vectorizer_en = TextVectorization(max_tokens=max_vocab_size, output_mode='int', output_sequence_length=max_length)

vectorizer_bn.adapt(train_bn)
vectorizer_en.adapt(train_en)

# Define the Seq2Seq Model with LSTM
def build_seq2seq_model(embed_dim=256, lstm_units=512):
    encoder_inputs = Input(shape=(max_length,), dtype=tf.int32)
    encoder_embedding = Embedding(max_vocab_size, embed_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

    decoder_inputs = Input(shape=(max_length,), dtype=tf.int32)
    decoder_embedding = Embedding(max_vocab_size, embed_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

    decoder_dense = Dense(max_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare data for training
train_bn_tokens = vectorizer_bn(train_bn)
train_en_tokens = vectorizer_en(train_en)
val_bn_tokens = vectorizer_bn(val_bn)
val_en_tokens = vectorizer_en(val_en)

# Shift the decoder input-output
decoder_input_data = train_en_tokens[:, :-1]
decoder_target_data = train_en_tokens[:, 1:]

# Build & train model
model = build_seq2seq_model()
model.fit([train_bn_tokens, decoder_input_data], decoder_target_data, validation_data=([val_bn_tokens, val_en_tokens[:, :-1]], val_en_tokens[:, 1:]), epochs=10, batch_size=32)

# Save model
model.save("bangla_nmt_model.h5")

# Inference
class Translator:
    def __init__(self, model, vectorizer_bn, vectorizer_en):
        self.model = model
        self.vectorizer_bn = vectorizer_bn
        self.vectorizer_en = vectorizer_en
    
    def translate(self, sentence):
        tokenized_input = self.vectorizer_bn([preprocess_sentence(sentence)])
        output_sentence = []
        
        for _ in range(max_length):
            predictions = self.model.predict([tokenized_input, np.array([output_sentence])])
            predicted_id = np.argmax(predictions[0, -1, :])
            output_sentence.append(predicted_id)
            if predicted_id == 0:
                break
        
        translated_words = [self.vectorizer_en.get_vocabulary()[idx] for idx in output_sentence if idx > 0]
        return " ".join(translated_words)

translator = Translator(model, vectorizer_bn, vectorizer_en)
print(translator.translate("আমি ভালো আছি।"))

# Evaluate using BLEU Score
def evaluate_bleu_score(test_bn, test_en, translator):
    references = [[ref.split()] for ref in test_en]
    candidates = [translator.translate(sent).split() for sent in test_bn]
    return corpus_bleu(references, candidates)

bleu_score = evaluate_bleu_score(val_bn, val_en, translator)
print("BLEU Score:", bleu_score)
