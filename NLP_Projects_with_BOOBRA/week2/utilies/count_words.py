import re

def count_words(text):
    counts = dict()
    print('coouts in dict', counts)
    # convert to lowercase
    text = text.lower()
    print('text------>',text)
    words = re.findall(r'\b\w+\b',text)
    print('words----->',words)
     # Aggregate word counts using a dictionary
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


def test_run():
    with open("dataset/input.txt", "r") as f:
        text = f.read()
        counts = count_words(text)
        sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
        
        print("10 most common words----------:\nWord\tCount")
        for word, count in sorted_counts[:10]:
            print("{}\t{}".format(word, count))
        
        print("\n10 least common words----------:\nWord\tCount")
        for word, count in sorted_counts[-10:]:
            print("{}\t{}".format(word, count))

        
if __name__ == "__main__":
    test_run()