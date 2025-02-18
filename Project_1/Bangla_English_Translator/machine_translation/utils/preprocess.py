import tensorflow as tf
import unicodedata
import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Function to normalize text (removes unwanted spaces, unicode normalization)
def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^a-zA-Zঅ-হ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to load dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path, delimiter="\t", header=None, names=["English", "Bengali", "Meta"])
    data = data.drop(columns=["Meta"])  # Drop attribution metadata

    data["English"] = data["English"].apply(lambda x: "<sos> " + normalize_text(x.lower()) + " <eos>")
    data["Bengali"] = data["Bengali"].apply(lambda x: "<sos> " + normalize_text(x) + " <eos>")

    return data["English"].tolist(), data["Bengali"].tolist()

# Tokenization function
def tokenize(sentences):
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    return sequences, tokenizer

# Convert text into padded sequences
def preprocess_data(file_path, max_length=20):
    en_texts, bn_texts = load_dataset(file_path)

    en_seq, en_tokenizer = tokenize(en_texts)
    bn_seq, bn_tokenizer = tokenize(bn_texts)

    en_seq = pad_sequences(en_seq, maxlen=max_length, padding="post")
    bn_seq = pad_sequences(bn_seq, maxlen=max_length, padding="post")

    return en_seq, bn_seq, en_tokenizer, bn_tokenizer
