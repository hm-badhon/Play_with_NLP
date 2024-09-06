## Traditional Approach for Text Representation

The traditional approach for text representation involves creating a list of unique terms and assigning each term a unique integer value or ID. These IDs are then used to represent words in a sentence. Each word in the vocabulary becomes a feature, leading to a large feature size for extensive vocabularies.

## Common Traditional Methods
## 1. **One-Hot Encoding**: 
   - Represents each word as a unique vector.
   - The vector's dimensionality equals the vocabulary size.
   - All vector elements are set to 0 except for the one corresponding to the word's index, which is set to 1.

## Example of One-Hot Encoding 
```python
def one_hot_encode(text):
    words = text.split()
    vocabulary = set(words)
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    one_hot_encoded = []
    for word in words:
        one_hot_vector = [0] * len(vocabulary)
        one_hot_vector[word_to_index[word]] = 1
        one_hot_encoded.append(one_hot_vector)
 
    return one_hot_encoded, word_to_index, vocabulary
 
# sample
example_text = "cat in the hat dog on the mat bird in the tree"
 
one_hot_encoded, word_to_index, vocabulary = one_hot_encode(example_text)
 
print("Vocabulary:", vocabulary)
print("Word to Index Mapping:", word_to_index)
print("One-Hot Encoded Matrix:")
for word, encoding in zip(example_text.split(), one_hot_encoded):
    print(f"{word}: {encoding}")
```

### Output :
```python
Vocabulary: {'mat', 'hat', 'cat', 'bird', 'tree', 'dog', 'in', 'the', 'on'}
Word to Index Mapping: {'mat': 0, 'hat': 1, 'cat': 2, 'bird': 3, 'tree': 4, 'dog': 5, 'in': 6, 'the': 7, 'on': 8}
One-Hot Encoded Matrix:
cat: [0, 0, 1, 0, 0, 0, 0, 0, 0]
in: [0, 0, 0, 0, 0, 0, 1, 0, 0]
the: [0, 0, 0, 0, 0, 0, 0, 1, 0]
hat: [0, 1, 0, 0, 0, 0, 0, 0, 0]
dog: [0, 0, 0, 0, 0, 1, 0, 0, 0]
on: [0, 0, 0, 0, 0, 0, 0, 0, 1]
the: [0, 0, 0, 0, 0, 0, 0, 1, 0]
mat: [1, 0, 0, 0, 0, 0, 0, 0, 0]
bird: [0, 0, 0, 1, 0, 0, 0, 0, 0]
in: [0, 0, 0, 0, 0, 0, 1, 0, 0]
the: [0, 0, 0, 0, 0, 0, 0, 1, 0]
tree: [0, 0, 0, 0, 1, 0, 0, 0, 0]
```

### Disadvantages of One-Hot Encoding

1. **High Dimensionality**:
   - One-hot encoding creates high-dimensional vectors, which are computationally expensive and require significant memory, especially with large vocabularies.

2. **Lack of Semantic Relationships**:
   - It does not capture the semantic relationships between words. Each word is treated as an isolated entity, without considering its meaning or context.

3. **Vocabulary Limitation**:
   - One-hot encoding is limited to the vocabulary seen during training, making it ineffective for handling out-of-vocabulary (OOV) words.



## 2. Bag of Words (BoW) Model 

The Bag of Words (BoW) model is a text preprocessing technique used in Natural Language Processing (NLP) to convert text into numerical data, which can then be used in machine learning algorithms. The model creates a "bag" of words, counting the frequency of each word in a text, while disregarding grammar and word order.

### Steps to Apply the Bag of Words Model:

1. **Preprocessing the Text**:
   - **Convert text to lower case**: Ensures uniformity by treating words with different cases as the same.
   - **Remove non-word characters**: Eliminates characters that are not part of words, such as numbers or symbols.
   - **Remove all punctuations**: Strips punctuation marks to focus solely on words.

   Example Python code for preprocessing:

   ```python
   import nltk 
   import re 

   # text
   text = """Beans. I was trying to explain to somebody as we were flying in, that’s corn. That’s beans. And they were very impressed at my agricultural knowledge. Please give it up for Amaury once again for that outstanding introduction. I have a bunch of good friends here today, including somebody who I served with, who is one of the finest senators in the country, and we’re lucky to have him, your Senator, Dick Durbin is here. I also noticed, by the way, former Governor Edgar here, who I haven’t seen in a long time, and somehow he has not aged and I have. And it’s great to see you, Governor. I want to thank President Killeen and everybody at the U of I System for making it possible for me to be here today. And I am deeply honored at the Paul Douglas Award that is being given to me. He is somebody who set the path for so much outstanding public service here in Illinois. Now, I want to start by addressing the elephant in the room. I know people are still wondering why I didn’t speak at the commencement."""
   y
   dataset = nltk.sent_tokenize(text)
   for i in range(len(dataset)): 
       dataset[i] = dataset[i].lower() 
       dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
       dataset[i] = re.sub(r'\s+', ' ', dataset[i]) 
   ```
    ### Output :

    ```bash
    beans 
    i was trying to explain to somebody as we were flying in that s corn 
    that s beans 
    and they were very impressed at my agricultural knowledge 
    please give it up for amaury once again for that outstanding introduction 
    i have a bunch of good friends here today including somebody who i served with who is one of the finest senators in the country and we re lucky to have him your senator dick durbin is here 
    i also noticed by the way former governor edgar here who i haven t seen in a long time and somehow he has not aged and i have 
    and it s great to see you governor 
    i want to thank president killeen and everybody at the u of i system for making it possible for me to be here today 
    and i am deeply honored at the paul douglas award that is being given to me 
    he is somebody who set the path for so much outstanding public service here in illinois 
    now i want to start by addressing the elephant in the room 
    i know people are still wondering why i didn t speak at the commencement 
    ```
2. **Obtaining the Most Frequent Words**:
   - **Create a dictionary**: Holds the count of each word's occurrence.
   - **Tokenize the sentences**: Breaks the text into words.
   - **Count word frequency**: For each word, check if it exists in the dictionary; if yes, increment its count; if not, add it with a count of 1.

   Example Python code for generating word counts:
   ```python
   word2count = {} 
   for data in dataset: 
       words = nltk.word_tokenize(data) 
       for word in words: 
           if word not in word2count.keys(): 
               word2count[word] = 1
           else: 
               word2count[word] += 1
   ```
   ### Output:
    ```bash
    Word2Count------>
    {'beans': 2, 'i': 12, 'was': 1, 'trying': 1, 'to': 8, 'explain': 1, 'somebody': 3, 'as': 1, 'we': 2, 'were': 2, 'flying': 1, 'in': 5, 'that': 4, 's': 3, 'corn': 1, 'and': 7, 'they': 1, 'very': 1, 'impressed': 1, 'at': 4, 'my': 1, 'agricultural': 1, 'knowledge': 1, 'please': 1, 'give': 1, 'it': 3, 'up': 1, 'for': 5, 'amaury': 1, 'once': 1, 'again': 1, 'outstanding': 2, 'introduction': 1, 'have': 3, 'a': 2, 'bunch': 1, 'of': 3, 'good': 1, 'friends': 1, 'here': 5, 'today': 2, 'including': 1, 'who': 4, 'served': 1, 'with': 1, 'is': 4, 'one': 1, 'the': 9, 'finest': 1, 'senators': 1, 'country': 1, 're': 1, 'lucky': 1, 'him': 1, 'your': 1, 'senator': 1, 'dick': 1, 'durbin': 1, 'also': 1, 'noticed': 1, 'by': 2, 'way': 1, 'former': 1, 'governor': 2, 'edgar': 1, 'haven': 1, 't': 2, 'seen': 1, 'long': 1, 'time': 1, 'somehow': 1, 'he': 2, 'has': 1, 'not': 1, 'aged': 1, 'great': 1, 'see': 1, 'you': 1, 'want': 2, 'thank': 1, 'president': 1, 'killeen': 1, 'everybody': 1, 'u': 1, 'system': 1, 'making': 1, 'possible': 1, 'me': 2, 'be': 1, 'am': 1, 'deeply': 1, 'honored': 1, 'paul': 1, 'douglas': 1, 'award': 1, 'being': 1, 'given': 1, 'set': 1, 'path': 1, 'so': 1, 'much': 1, 'public': 1, 'service': 1, 'illinois': 1, 'now': 1, 'start': 1, 'addressing': 1, 'elephant': 1, 'room': 1, 'know': 1, 'people': 1, 'are': 1, 'still': 1, 'wondering': 1, 'why': 1, 'didn': 1, 'speak': 1, 'commencement': 1}
    ```
3. **Selecting the Most Frequent Words**:
   - **Limit the number of words**: In large texts, select only the most frequent words to avoid dealing with a vast number of words.
   
   Python code for selecting the top 100 words:
   ```python
   import heapq 
   freq_words = heapq.nlargest(100, word2count, key=word2count.get)
   ```

   ### Output:
   ```bash
   Most frequent words ----
    ['i', 'the', 'to', 'and', 'in', 'for', 'here', 'that', 'at', 'who', 'is', 'somebody', 's', 'it', 'have', 'of', 'beans', 'we', 'were', 'outstanding', 'a', 'today', 'by', 'governor', 't', 'he', 'want', 'me', 'was', 'trying', 'explain', 'as', 'flying', 'corn', 'they', 'very', 'impressed', 'my', 'agricultural', 'knowledge', 'please', 'give', 'up', 'amaury', 'once', 'again', 'introduction', 'bunch', 'good', 'friends', 'including', 'served', 'with', 'one', 'finest', 'senators', 'country', 're', 'lucky', 'him', 'your', 'senator', 'dick', 'durbin', 'also', 'noticed', 'way', 'former', 'edgar', 'haven', 'seen', 'long', 'time', 'somehow', 'has', 'not', 'aged', 'great', 'see', 'you', 'thank', 'president', 'killeen', 'everybody', 'u', 'system', 'making', 'possible', 'be', 'am', 'deeply', 'honored', 'paul', 'douglas', 'award', 'being', 'given', 'set', 'path', 'so']
   ```

4. **Building the BoW Model**:
   - **Construct vectors**: Create a vector for each sentence indicating the presence of frequent words. If a frequent word is in the sentence, it is represented by 1; otherwise, by 0.

   Python code for building the model:
   ```python
   X = [] 
   for data in dataset: 
       vector = [] 
       for word in freq_words: 
           if word in nltk.word_tokenize(data): 
               vector.append(1) 
           else: 
               vector.append(0) 
       X.append(vector) 
   X = np.asarray(X)
   ```

    ### Output:
    ```bash
        Construct vectors----->
        [[0 0 0 ... 0 0 0]
        [1 0 1 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        ...
        [0 1 0 ... 1 1 1]
        [1 1 1 ... 0 0 0]
        [1 1 0 ... 0 0 0]]
    ```



## Limitations of the Bag of Words (BoW) model's with examples:

### 1. **Ignoring Word Order**:
- In BoW, the order of words is not considered, which can lead to a loss of context. For example, consider two sentences:

    - **Sentence 1**: "The dog bit the man."
    - **Sentence 2**: "The man bit the dog."

    In the BoW model, both sentences would produce the same word vector because they contain the same words, even though the meaning is completely different. BoW simply counts word occurrences, ignoring the sequence. This limitation makes it ineffective for tasks that depend on word order, such as sentiment analysis or text understanding.

    #### For example, the BoW vector for both sentences might look like this (assuming a vocabulary of 5 unique words):

    | Word     | The | dog | bit | man | the |
    |----------|-----|-----|-----|-----|-----|
    | Sentence 1 | 1   | 1   | 1   | 1   | 1   |
    | Sentence 2 | 1   | 1   | 1   | 1   | 1   |

    Even though the sentences mean different things, the vectors are identical, causing BoW to lose crucial contextual meaning.

### 2. **Sparsity**:
BoW generates large, sparse vectors, especially when working with large vocabularies. For example, if you have a corpus with 10,000 unique words and a document that uses only 50 of them, the resulting vector would contain 9,950 zero values and only 50 non-zero values.

- Let’s consider these three short documents:

    - **Document 1**: "I love programming."
    - **Document 2**: "Programming is fun."
    - **Document 3**: "I enjoy machine learning."

    Assume the corpus has a vocabulary of 7 words: ["I", "love", "programming", "is", "fun", "enjoy", "machine", "learning"]. Each document will be represented by an 8-dimensional vector, but most elements will be zero:

    | Word           | I  | love | programming | is | fun | enjoy | machine | learning |
    |----------------|----|------|-------------|----|-----|-------|---------|----------|
    | **Document 1** | 1  | 1    | 1           | 0  | 0   | 0     | 0       | 0        |
    | **Document 2** | 0  | 0    | 1           | 1  | 1   | 0     | 0       | 0        |
    | **Document 3** | 1  | 0    | 0           | 0  | 0   | 1     | 1       | 1        |

    As shown, the vectors are sparse, with many zeros, especially in large datasets. 
    #### This makes the model computationally inefficient and requires more memory to store these vectors.


## 1.3. Term frequency-inverse document frequency (TF-IDF)

### Term Frequency-Inverse Document Frequency (TF-IDF) is a statistic used to evaluate the importance of a word in a document relative to a collection of documents (corpus). 
### It consists of two components:

- **Term Frequency (TF)**: Measures how often a term appears in a document.
- **Inverse Document Frequency (IDF)**: Measures the importance of a term across the entire corpus, with less common terms receiving higher weights.

    The TF-IDF score is calculated by multiplying TF and IDF, where a higher score indicates greater importance of a word in a document compared to the entire corpus. TF-IDF is commonly used in text mining, information retrieval, and document clustering.

    To implement TF-IDF in Python, the `TfidfVectorizer` from the scikit-learn library can transform sample documents into a TF-IDF matrix, helping identify significant words for text analysis tasks.


    ```python
        from sklearn.feature_extraction.text import TfidfVectorizer
        documents = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
        ]

        vectorizer = TfidfVectorizer() # Create the TF-IDF vectorizer
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_values = {}

        for doc_index, doc in enumerate(documents):
            feature_index = tfidf_matrix[doc_index, :].nonzero()[1]
            tfidf_doc_values = zip(feature_index, [tfidf_matrix[doc_index, x] for x in feature_index])
            tfidf_values[doc_index] = {feature_names[i]: value for i, value in tfidf_doc_values}
        #let's print
        for doc_index, values in tfidf_values.items():
            print(f"Document {doc_index + 1}:")
            for word, tfidf_value in values.items():
                print(f"{word}: {tfidf_value}")
            print("\n")
    ```
 ## Limitations of TF-IDF:

1. **Lack of Semantic Understanding**: TF-IDF treats words independently and doesn't account for synonyms or contextual meaning.
  
2. **Sensitivity to Document Length**: Longer documents tend to have higher term frequencies, which can bias TF-IDF scores.

3. **Inability to Capture Word Order**: TF-IDF ignores word order and syntactic structure, missing complex language patterns.

4. **Static Weights**: TF-IDF weights do not adapt or evolve as new documents are added to the corpus.


1. **Lack of Semantic Understanding**:
   - **Example**: In two documents:
     - Document 1: "I want to **buy** a new car."
     - Document 2: "I plan to **purchase** a new vehicle."
   - TF-IDF treats "**buy**" and "**purchase**" as different words, even though they have the same meaning. As a result, it misses the semantic similarity between the two documents.

2. **Sensitivity to Document Length**:
   - **Example**:
     - Document 1: "Python is a programming language."
     - Document 2: "Python is a widely used high-level programming language created by Guido van Rossum in 1991. Python emphasizes readability, with its use of significant indentation."
   - Document 2 is longer and will likely have higher term frequencies, causing TF-IDF to give more weight to terms in the longer document, even if Document 1 is just as relevant.

3. **Inability to Capture Word Order**:
   - **Example**:
     - Document 1: "The cat chased the dog."
     - Document 2: "The dog chased the cat."
   - TF-IDF assigns the same weights to both documents because it doesn’t capture word order, even though their meanings are opposite.

4. **Static Weights**:
   - **Example**: If you compute TF-IDF for a set of documents and later add more documents, the term weights remain unchanged unless recalculated for the entire corpus. This makes TF-IDF less dynamic for continuously growing datasets.