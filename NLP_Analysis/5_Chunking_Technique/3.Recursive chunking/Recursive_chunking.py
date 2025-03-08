import re

def split_into_paragraphs(text):
    """Splits text into paragraphs based on double newlines."""
    return text.strip().split('\n\n')

def recursive_chunking(text, max_size=50):
    """Recursively splits text into smaller chunks if it exceeds max_size."""
    
    # Base case: If text length is within the limit, return as a chunk
    if len(text) <= max_size:
        return [text]
    
    # Try splitting by sentence (using `. ` as delimiter)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Prevent infinite recursion by ensuring chunk size is decreasing
    if len(chunks) == 1 and len(chunks[0]) > max_size:
        mid = len(chunks[0]) // 2
        chunks = [chunks[0][:mid], chunks[0][mid:]]
    
    return chunks

large_text = """
Artificial intelligence (AI) is a rapidly evolving field. It includes machine learning, natural language processing, and computer vision. AI systems learn from data and improve over time. 

Deep learning, a subset of machine learning, uses neural networks. These networks mimic the human brain. Convolutional Neural Networks (CNNs) are widely used for image processing. Recurrent Neural Networks (RNNs) handle sequential data. 

AI applications range from healthcare to finance. AI-powered chatbots assist customers. Self-driving cars rely on AI for navigation.
"""

# Step 1: Split by paragraphs
# paragraphs = split_into_paragraphs(test_text)
paragraphs = split_into_paragraphs(large_text)

# Step 2: Apply recursive chunking
final_chunks = []
for para in paragraphs:
    final_chunks.extend(recursive_chunking(para, max_size=50))

# Print results
for i, chunk in enumerate(final_chunks):
    print(f"Chunk {i+1}: {chunk}")

