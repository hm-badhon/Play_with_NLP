
# Document Similarity Detection

This project implements a **Document Similarity Detection System** using advanced **Natural Language Processing (NLP)** techniques. The system allows users to compare two documents and measure their **semantic similarity** using multiple methods such as **TF-IDF**, **Word2Vec**, and **BERT**.

## Features

- **Text Preprocessing**: Tokenization, stemming, lemmatization, and stop word removal.
- **Document Vectorization**: 
  - TF-IDF
  - Word2Vec
  - BERT embeddings (Transformer-based)
- **Similarity Computation**: Cosine similarity to calculate the similarity between two documents.
- **Web Interface**: Built using **Flask** to interact with users and display similarity results.

## Technologies Used

- **Python**: Primary programming language.
- **Flask**: Web framework for creating the web application.
- **SpaCy**: For text preprocessing (tokenization, lemmatization).
- **Gensim**: For Word2Vec and TF-IDF vectorization.
- **Transformers (HuggingFace)**: For BERT-based embeddings and semantic analysis.
- **scikit-learn**: For cosine similarity computation.

## Setup Instructions

To run this project locally, follow the steps below:

### 1. Clone the repository

```bash
git clone https://github.com/your-username/document-similarity-detection.git
cd document-similarity-detection
```

### 2. Install dependencies

Make sure you have Python 3.7+ installed, and then install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Run the application

Run the Flask app using the following command:

```bash
python app.py
```

The Flask server will start at `http://127.0.0.1:5000/`.

### 4. Open in browser

Open your browser and go to `http://127.0.0.1:5000/` to use the document similarity checker.

## Deployment

You can deploy this project on **Render** or any other cloud platform that supports Python. For deployment instructions on **Render**, follow the steps below:

1. Sign up for an account on [Render](https://render.com/).
2. Create a new Web Service and link it to this GitHub repository.
3. Set the build command to `pip install -r requirements.txt` and the start command to `python app.py`.
4. Once deployed, you can access your web service at the provided URL.

## Example

After starting the Flask app, go to the homepage, and enter two documents in the provided text areas. Once submitted, the system will return similarity scores for each method:

- **TF-IDF Similarity**: Traditional vectorization method.
- **Word2Vec Similarity**: Using Word2Vec embeddings.
- **BERT Similarity**: Using a transformer model (BERT) to understand the contextual meaning of the documents.

## Requirements

- Python 3.7 or higher
- Flask
- SpaCy
- Gensim
- scikit-learn
- HuggingFace Transformers

### Example of `requirements.txt`:

```
flask==2.0.2
spacy==3.0.6
gensim==4.1.2
scikit-learn==0.24.2
transformers==4.10.2
torch==1.9.1
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for providing pre-trained transformer models like BERT.
- [SpaCy](https://spacy.io/) for NLP preprocessing.
- [Gensim](https://radimrehurek.com/gensim/) for vectorization methods.
```
