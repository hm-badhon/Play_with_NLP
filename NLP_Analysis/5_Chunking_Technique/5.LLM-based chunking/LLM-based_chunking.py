import nltk
import re
from nltk.tokenize import sent_tokenize
from transformers import pipeline

nltk.download('punkt')

def count_tokens(text):
    """Counts the number of words in a given text."""
    return len(text.split())

def mistral_based_chunking(text, max_tokens=100):
    """Uses Mistral model to intelligently chunk text while preserving context."""
    summarizer = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct")
    
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = count_tokens(sentence)
        
        if current_length + sentence_length > max_tokens:
            chunk_text = " ".join(current_chunk)
            summary = summarizer(chunk_text, max_length=max_tokens, do_sample=False)[0]['generated_text']
            chunks.append(summary)
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        summary = summarizer(chunk_text, max_length=max_tokens, do_sample=False)[0]['generated_text']
        chunks.append(summary)
    
    return chunks

# Example Long Text
document_text = """
Artificial Intelligence is revolutionizing industries. Machine Learning (ML) and Deep Learning are subsets of AI.
ML allows systems to learn from data, while Deep Learning uses neural networks for complex tasks.

AI applications include healthcare, finance, and robotics. In healthcare, AI assists in disease diagnosis and personalized medicine.
Financial institutions use AI for fraud detection and risk assessment.

Self-driving cars rely on AI for real-time decision-making, while chatbots enhance customer support.

Despite its benefits, AI raises ethical concerns. Issues such as bias, privacy, and automation-driven job losses must be addressed.
"""

# Apply Mistral-Based Chunking
chunks = mistral_based_chunking(document_text, max_tokens=100)

# Print Results
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
