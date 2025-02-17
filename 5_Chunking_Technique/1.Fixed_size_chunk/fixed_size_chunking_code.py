
def fixed_size_chuning(text, chunk_size, overlap):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        print('chunk_size',chunk_size)
        print('overlap', overlap)
        chunk = words[i:i + chunk_size]
        print(chunk)
        chunks.append(" ".join(chunk))
    return chunks

# Example text
text = """Fixed-size chunking is one of the simplest ways to divide a text. 
It involves splitting the text into equal-sized chunks. However, this method 
may break sentences, causing key information to be split between chunks. 
Adding an overlap helps maintain context, but it does not completely prevent 
semantic disruptions."""
chunks = fixed_size_chuning(text, 10,3)
# print(chunks)

for i , chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}\n")
