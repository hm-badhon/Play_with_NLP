import pandas as pd
import tensorflow as tf
import re
import os

def clean_text(text):
    """Remove unwanted characters and lower the text."""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Zঅ-হ0-9।?!]", " ", text)  # Remove special characters except punctuation
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text

def load_and_preprocess_data(filepath):
    """Load dataset, clean it, and tokenize."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    english_sentences, bangla_sentences = [], []

    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            english_sentences.append(clean_text(parts[0]))
            bangla_sentences.append(clean_text(parts[1]))

    df = pd.DataFrame({"English": english_sentences, "Bangla": bangla_sentences})
    df.to_csv("/media/hmb/hdd2/Ongoing_Projects/NLP/Play_with_NLP/Project_1/Bangla_English_Translator/machine_translation/data/dataset.csv", index=False, encoding="utf-8")

    return df

if __name__ == "__main__":
    df = load_and_preprocess_data("/media/hmb/hdd2/Ongoing_Projects/NLP/Play_with_NLP/Project_1/Bangla_English_Translator/machine_translation/data/dataset.txt")
    print(df.head())
