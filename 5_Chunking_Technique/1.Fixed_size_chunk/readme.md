


### **Explanation:**
1. **Splitting Words:** The text is split into words to ensure even segmentation.
2. **Chunking with Overlap:** Each chunk contains a fixed number of words (`chunk_size`). The overlap (`overlap`) ensures continuity between consecutive chunks.
3. **Issue:** Even with overlap, some sentences get broken, leading to potential loss of meaning.

### **Output Example:**
```
Chunk 1: Fixed-size chunking is one of the simplest ways to divide

Chunk 2: ways to divide a text. It involves splitting the text

Chunk 3: splitting the text into equal-sized chunks. However, this method may

Chunk 4: this method may break sentences, causing key information to be

Chunk 5: information to be split between chunks. Adding an overlap helps

Chunk 6: an overlap helps maintain context, but it does not completely

Chunk 7: does not completely prevent semantic disruptions.

```
👉 **Problem:** The sentence **"It involves splitting the text into equal-sized chunks."** is split across two chunks, causing loss of meaning.

### **Better Alternative?**
- **Sentence-aware chunking:** Using NLP libraries like `nltk` or `spaCy` to split text by sentences instead of fixed-size chunks.
- **Semantic-aware chunking:** Using embeddings (e.g., `BERT`, `GPT`) to split based on meaning rather than word count.

