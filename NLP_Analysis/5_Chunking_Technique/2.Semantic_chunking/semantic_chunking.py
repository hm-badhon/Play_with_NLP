import spacy

# Load spaCy model for sentence segmentation
nlp = spacy.load("en_core_web_sm")

def simple_semantic_chunking(text, max_sentences):
    """Splits text into chunks of at most 'max_sentences' sentences."""
    doc = nlp(text)
    print('doc setence:', doc.sents)
    sentences = [sent.text for sent in doc.sents]

    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        chunks.append(chunk)

    return chunks

# Example text
text = """Semantic chunking groups sentences based on meaning rather than fixed sizes. 
For example, two sentences discussing the same idea should be in the same chunk. 
If the topic shifts, a new chunk is created. This ensures that chunks remain meaningful and contextually intact. 
Traditional chunking methods, like fixed-size chunking, often break important information. 
With semantic chunking, text retrieval and understanding improve significantly."""

# Perform chunking
chunks = simple_semantic_chunking(text, max_sentences=2)

# Display results
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}\n")
