import torch
import torch.nn as nn
import torch.optim as optim

# Define CBOW model
class CBOWModel(nn.Module):
    def __init__(self, vocal_size , embed_size):
        super(CBOWModel,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, context):
        context_embeds = self.embeddings(context).sum(dim=1)
        return output
# Sample data

context_size = 2
raw_text = "word embedding are awesome"
tokens = raw_text.split()
vocab = set(tokens)
word_to_index = {word:i for i , word in enumerate(vocab)}
data = []

for i in range
