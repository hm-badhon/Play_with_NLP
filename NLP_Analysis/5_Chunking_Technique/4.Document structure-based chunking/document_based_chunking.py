import re

def split_by_headings(text):
    """Splits text into chunks based on headings (H1, H2, H3, etc.)."""
    pattern = r'(#+\s[^\n]+)'  # Matches Markdown-style headings (e.g., # Title, ## Subtitle)
    sections = re.split(pattern, text)
    
    chunks = []
    current_chunk = ""

    for section in sections:
        if section.startswith("#"):  # If it's a heading, start a new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())  # Save previous chunk
            current_chunk = section  # Start new chunk with heading
        else:
            current_chunk += "\n" + section  # Append content to current chunk

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
document_text = """
# Introduction
Artificial Intelligence (AI) is transforming various industries. It includes fields such as machine learning and deep learning.

## Machine Learning
Machine Learning (ML) enables computers to learn from data. It can be supervised, unsupervised, or reinforcement learning.

### Supervised Learning
Supervised learning uses labeled data. Examples include classification and regression tasks.

### Unsupervised Learning
Unsupervised learning finds patterns in data without labels. Clustering and anomaly detection are common techniques.

## Applications of AI
AI is used in healthcare, finance, and robotics. Self-driving cars and chatbots rely on AI technology.
"""

# Apply chunking
chunks = split_by_headings(document_text)

# Print results
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")



