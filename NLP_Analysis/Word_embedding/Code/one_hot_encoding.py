def one_hot_encode(text):
    words = text.split()
    vocabulary = set(words)
    print('vocabulary---',vocabulary)
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