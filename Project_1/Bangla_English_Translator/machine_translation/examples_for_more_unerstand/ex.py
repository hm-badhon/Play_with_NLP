import tensorflow as tf
import unicodedata
import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def tokenize(sentences):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    return sequences, tokenizer

# Input sentences
sentences = ["I love deep learning.", "Deep learning is fun!"]

# Tokenize the sentences
sequences, tokenizer = tokenize(sentences)

print("Sequences:", sequences)
print("Word Index:", tokenizer.word_index)