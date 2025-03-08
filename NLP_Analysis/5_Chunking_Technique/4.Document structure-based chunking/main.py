import re

def split_document(text):
    """Splits text into chunks based on section breaks, bullet points, or paragraphs."""
    
    # Step 1: Try splitting by double newlines (indicating a paragraph or section break)
    sections = re.split(r'\n\s*\n+', text.strip())  
    
    # Step 2: Further split if bullet points (*) or numbered lists (1., 2.)
    refined_chunks = []
    for section in sections:
        if re.match(r'^\d+\.\s|\*|\-', section):  # Detects "1. Text" or "* Text" or "- Text"
            refined_chunks.extend(re.split(r'\n', section))  # Split by line
        else:
            refined_chunks.append(section)  # Treat it as a standalone paragraph
    
    return [chunk.strip() for chunk in refined_chunks if chunk.strip()]

# Example Document Without Headings
document_text = """
Artificial Intelligence is transforming industries. It includes fields such as machine learning and deep learning.

1. Machine Learning (ML) enables computers to learn from data.
2. It can be supervised, unsupervised, or reinforcement learning.

Deep learning uses neural networks. It mimics how humans learn from experience.

- AI is used in healthcare, finance, and robotics.
- Self-driving cars and chatbots rely on AI technology.
"""

# Apply Chunking
chunks = split_document(document_text)

# Print Results
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
