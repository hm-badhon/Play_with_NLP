
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_data

# Load trained model
model = load_model("models/translation_model.h5")

# Load tokenizer
_, _, en_tokenizer, bn_tokenizer = preprocess_data("data/dataset.csv")

def translate_sentence(sentence):
    """Translate English to Bangla"""
    sequence = en_tokenizer.texts_to_sequences([sentence])
    prediction = model.predict(sequence)
    predicted_tokens = np.argmax(prediction, axis=-1)
    
    translated_text = " ".join([bn_tokenizer.index_word[idx] for idx in predicted_tokens[0] if idx != 0])
    return translated_text

# Example usage
sentence = "go."
translated = translate_sentence(sentence)
print(f"English: {sentence} \nBengali: {translated}")
