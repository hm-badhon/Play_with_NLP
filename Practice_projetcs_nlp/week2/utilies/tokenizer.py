import re

def sent_tokenize(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences = [s.strip() for s in sentences if s]
    return sentences

def word_tokenize(sent):
    """Split a sentence into words."""
    words = re.findall(r'\b\w+\b', sent)
    words = [word.strip() for word in words if word]
    return words



def test_run():
    """Called on Test Run."""

    text = "The first time you see The Second Renaissance it may look boring. Look at it at least twice and definitely watch part 2. It will change your view of the matrix. Are the human people the ones who started the war? Is AI a bad thing?"
    print("--- Sample text ---", text, sep="\n")
    
    sentences = sent_tokenize(text)
    print("\n--- Sentences ---")
    print(sentences)
    
    print("\n--- Words ---")
    for sent in sentences:
        print(sent)
        print(word_tokenize(sent))
        print()  # blank line for readability

if __name__ == "__main__":
    test_run()


