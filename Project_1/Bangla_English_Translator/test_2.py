import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

# Load dataset
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pairs = [line.strip().split('\t')[:2] for line in lines if '\t' in line]
    bn_sentences, en_sentences = zip(*pairs)
    return list(bn_sentences), list(en_sentences)

# Preprocessing
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^\u0980-\u09FFa-zA-Z!?.,' ]+", "", sentence)
    return sentence

def preprocess_data(bn_sentences, en_sentences):
    bn_sentences = [preprocess_sentence(sent) for sent in bn_sentences]
    en_sentences = [preprocess_sentence(sent) for sent in en_sentences]
    return bn_sentences, en_sentences

# Load & preprocess data
file_path = 'data/dataset.txt'  # Change this if needed
bn_sentences, en_sentences = load_data(file_path)
bn_sentences, en_sentences = preprocess_data(bn_sentences, en_sentences)

# Split dataset
train_bn, val_bn, train_en, val_en = train_test_split(bn_sentences, en_sentences, test_size=0.2, random_state=42)

# Define Tokenizer
class Tokenizer:
    def __init__(self, sentences, max_vocab_size=10000):
        words = set()
        for sent in sentences:
            words.update(sent.split())
        self.vocab = {word: idx for idx, word in enumerate(words, start=1)}
        self.vocab['<PAD>'] = 0
    
    def encode(self, sentence, max_length):
        return [self.vocab.get(word, 0) for word in sentence.split()][:max_length] + [0] * (max_length - len(sentence.split()))
    
    def decode(self, indices):
        inv_vocab = {idx: word for word, idx in self.vocab.items()}
        return " ".join([inv_vocab[idx] for idx in indices if idx > 0])

bn_tokenizer = Tokenizer(train_bn)
en_tokenizer = Tokenizer(train_en)

max_length = 20
train_bn_tokens = [bn_tokenizer.encode(sent, max_length) for sent in train_bn]
train_en_tokens = [en_tokenizer.encode(sent, max_length) for sent in train_en]
val_bn_tokens = [bn_tokenizer.encode(sent, max_length) for sent in val_bn]
val_en_tokens = [en_tokenizer.encode(sent, max_length) for sent in val_en]

# Dataset & DataLoader
class TranslationDataset(Dataset):
    def __init__(self, source, target):
        self.source = torch.tensor(source, dtype=torch.long)
        self.target = torch.tensor(target, dtype=torch.long)
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]

train_dataset = TranslationDataset(train_bn_tokens, train_en_tokens)
val_dataset = TranslationDataset(val_bn_tokens, val_en_tokens)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define Model
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim=256, hidden_dim=512):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.Embedding(input_dim, embed_dim)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Embedding(output_dim, embed_dim)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src, trg):
        _, (hidden, cell) = self.encoder_lstm(self.encoder(src))
        outputs, _ = self.decoder_lstm(self.decoder(trg), (hidden, cell))
        return self.fc(outputs)

# Instantiate Model
input_dim = len(bn_tokenizer.vocab)
output_dim = len(en_tokenizer.vocab)
model = Seq2Seq(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, trg in train_loader:
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            loss = criterion(output.reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')
    torch.save(model.state_dict(), 'bangla_nmt_pytorch.pth')

def translate_sentence(model, sentence):
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor([bn_tokenizer.encode(sentence, max_length)], dtype=torch.long)
        trg_tensor = torch.zeros((1, max_length), dtype=torch.long)
        for i in range(max_length):
            output = model(src_tensor, trg_tensor)
            next_word = torch.argmax(output[0, i, :]).item()
            if next_word == 0:
                break
            trg_tensor[0, i] = next_word
        return en_tokenizer.decode(trg_tensor[0].tolist())

train_model(model, train_loader, val_loader)
translator = lambda x: translate_sentence(model, x)

# Evaluate BLEU Score
def evaluate_bleu_score(test_bn, test_en, translator):
    references = [[ref.split()] for ref in test_en]
    candidates = [translator(sent).split() for sent in test_bn]
    return corpus_bleu(references, candidates)

bleu_score = evaluate_bleu_score(val_bn, val_en, translator)
print("BLEU Score:", bleu_score)