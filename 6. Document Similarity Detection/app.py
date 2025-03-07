# Import required libraries
import spacy
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load SpaCy for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text using SpaCy
def preprocess_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Function for TF-IDF Vectorization
def tfidf_vectorizer(documents):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    return tfidf_matrix

# Function for Word2Vec embedding
def word2vec_vectorizer(documents):
    tokenized_documents = [doc.split() for doc in documents]
    model = Word2Vec(tokenized_documents, vector_size=100, window=5, min_count=1, workers=4)
    doc_vectors = []
    for doc in tokenized_documents:
        vector = sum([model.wv[word] for word in doc if word in model.wv])
        doc_vectors.append(vector / len(doc))  # average vector for document
    return doc_vectors

# Function for BERT embedding using HuggingFace transformers
def bert_vectorizer(documents):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    doc_vectors = []
    for doc in documents:
        inputs = tokenizer(doc, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        doc_vectors.append(embeddings)
    
    return doc_vectors

# Function to compute similarity using cosine similarity
def compute_similarity(doc_vectors):
    cosine_sim = cosine_similarity(doc_vectors)
    return cosine_sim

# Flask route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route to handle document similarity requests
@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    if request.method == 'POST':
        # Get the input documents from the form
        doc1 = request.form['doc1']
        doc2 = request.form['doc2']
        
        # Preprocess the documents
        documents = [preprocess_text(doc1), preprocess_text(doc2)]
        
        # Vectorization Methods
        # TF-IDF
        tfidf_matrix = tfidf_vectorizer(documents)
        tfidf_similarity = cosine_similarity(tfidf_matrix)[0][1]

        # Word2Vec
        word2vec_vectors = word2vec_vectorizer(documents)
        word2vec_similarity = compute_similarity(word2vec_vectors)[0][1]

        # BERT
        bert_vectors = bert_vectorizer(documents)
        bert_similarity = compute_similarity(bert_vectors)[0][1]

        # Return the similarity scores
        return jsonify({
            'TF-IDF Similarity': tfidf_similarity,
            'Word2Vec Similarity': word2vec_similarity,
            'BERT Similarity': bert_similarity
        })

if __name__ == '__main__':
    app.run(debug=True)
