import torch
import torch.nn as nn
import torch.optim as optim

# Define CBOW model
class CBOWModel(nn.Module):
	def __init__(self, vocab_size, embed_size):
		super(CBOWModel, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embed_size)
		self.linear = nn.Linear(embed_size, vocab_size)

	def forward(self, context):
		context_embeds = self.embeddings(context).sum(dim=1)
		output = self.linear(context_embeds)
		return output

context_size = 2
raw_text = "word embeddings are awesome"
tokens = raw_text.split()
print('tokens',tokens)
vocab = set(tokens)
print('vocab', vocab)
word_to_index = {word: i for i, word in enumerate(vocab)}
print('word_to_index', word_to_index)

data =['my name is badhon']
print("Token:", tokens)

for i in range(2, len(tokens)-2):
    print('i',i)
    context = [word_to_index[word] for word in tokens[i - 2:i] + tokens[i + 1:i +3]]
    print('context------', context)
    target = word_to_index[tokens[i]]
    print('target-------', target)
    data.append((torch.tensor(context), torch.tensor(target)))
    print('data', data)
    
    
# Hyperparameters
vocab_size = len(vocab)
embed_size = 10
learning_rate = 0.01
epochs = 100

# Initialize CBOW model
cbow_model = CBOWModel(vocab_size, embed_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cbow_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    total_loss = 0
#     print("#####Data:", data)

    for context, target in data:
        print('context', context)
        print('target',target)
        print('data', data)
        optimizer.zero_grad()
        output = cbow_model(context)
        loss = criterion(output.unsqueeze(0), target.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# Example usage: Get embedding for a specific word
word_to_lookup = "embeddings"
word_index = word_to_index[word_to_lookup]
embedding = cbow_model.embeddings(torch.tensor([word_index]))
print(f"Embedding for '{word_to_lookup}': {embedding.detach().numpy()}")
