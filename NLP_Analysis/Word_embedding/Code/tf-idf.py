
# from sklearn.feature_extraction.text import TfidfVectorizer

# documents = [
#     "The quick brown fox jumps over the lazy dog.",
#     "A journey of a thousand miles begins with a single step.",
# ]
# vectorizer = TfidfVectorizer() # Create the TF-IDF vectorizer

# tfidf_matrix = vectorizer.fit_transform(documents)
# print('tfidf matrix---',tfidf_matrix)

# feature_names = vectorizer.get_feature_names_out()
# print('Feature names', feature_names)

# tfidf_values = {}



# for doc_index, doc in enumerate(documents):
#     feature_index = tfidf_matrix[doc_index, :].nonzero()[1]
#     print('Doc index', doc_index)
#     print('feature index', feature_index)

#     tfidf_doc_values = zip(feature_index, [tfidf_matrix[doc_index, x] for x in feature_index])
#     tfidf_values[doc_index] = {feature_names[i]: value for i, value in tfidf_doc_values}

# #let's print
# for doc_index, values in tfidf_values.items():
#     print(f"Document {doc_index + 1}:")
#     for word, tfidf_value in values.items():
#         print(f"{word}: {tfidf_value}")
#     print("\n")



from sklearn.feature_extraction.text import TfidfVectorizer

# Define the documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
]

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
print('tfidf matrix---',tfidf_matrix)

# Get the feature (word) names
feature_names = vectorizer.get_feature_names_out()
print('Feature names', feature_names)

# Convert the TF-IDF matrix to a dense (non-sparse) array for easy viewing
dense_matrix = tfidf_matrix.toarray()
print('dense matrix', dense_matrix)

# Loop over the documents and display TF-IDF scores
for doc_index, doc in enumerate(documents):
    print(f"Document {doc_index + 1}:")
    for word_index, word in enumerate(feature_names):
        tfidf_value = dense_matrix[doc_index][word_index]
        if tfidf_value > 0:  # Only print non-zero values
            print(f"{word}: {tfidf_value}")
    print("\n")
