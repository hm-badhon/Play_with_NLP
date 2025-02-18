
import numpy as np
from tensorflow.keras.utils import to_categorical
from models.seq2seq import build_seq2seq_model
from utils.preprocess import preprocess_data

# Load and preprocess data
dataset_path = "data/dataset.txt"
en_seq, bn_seq, en_tokenizer, bn_tokenizer = preprocess_data(dataset_path)

# Define model parameters
vocab_size_en = len(en_tokenizer.word_index) + 1
vocab_size_bn = len(bn_tokenizer.word_index) + 1

# Build model
model = build_seq2seq_model(vocab_size_en, vocab_size_bn)

# Convert target sequences to categorical labels
bn_seq_categorical = np.array([to_categorical(seq, num_classes=vocab_size_bn) for seq in bn_seq])

# Train the model
model.fit([en_seq, bn_seq], bn_seq_categorical, batch_size=32, epochs=5, validation_split=0.2)

# Save model
model.save("models/translation_model.h5")
