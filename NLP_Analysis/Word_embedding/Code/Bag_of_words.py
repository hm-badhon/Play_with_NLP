import nltk 
# nltk.download('punkt_tab')
import re 


# Preprocessing the Text:
text = """Beans. I was trying to explain to somebody as we were flying in, that’s corn. That’s beans. And they were very impressed at my agricultural knowledge. Please give it up for Amaury once again for that outstanding introduction. I have a bunch of good friends here today, including somebody who I served with, who is one of the finest senators in the country, and we’re lucky to have him, your Senator, Dick Durbin is here. I also noticed, by the way, former Governor Edgar here, who I haven’t seen in a long time, and somehow he has not aged and I have. And it’s great to see you, Governor. I want to thank President Killeen and everybody at the U of I System for making it possible for me to be here today. And I am deeply honored at the Paul Douglas Award that is being given to me. He is somebody who set the path for so much outstanding public service here in Illinois. Now, I want to start by addressing the elephant in the room. I know people are still wondering why I didn’t speak at the commencement.
"""
dataset = nltk.sent_tokenize(text)
for i in range(len(dataset)): 
    dataset[i] = dataset[i].lower() 
    dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
    dataset[i] = re.sub(r'\s+', ' ', dataset[i])
    print(dataset[i])

# Obtaining the Most Frequent Words:
word2count = {} 
for data in dataset: 
    words = nltk.word_tokenize(data) 
    for word in words: 
        if word not in word2count.keys(): 
            word2count[word] = 1
        else: 
            word2count[word] += 1

print('\nWord2Count------>\n', word2count)

# Selecting the Most Frequent Words:
import heapq
freq_words = heapq.nlargest(100, word2count,key=word2count.get)

print('\nMost frequent words ----\n',freq_words)

# Create a vector for each sentence indicating the presence of frequent words
import numpy as np
X =[]
for data in dataset:
    # print(data)
    vector=[]
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)
# X = np.asarray(X[0])
X = np.asarray(X)

print('\nConstruct vectors----->\n',X)

