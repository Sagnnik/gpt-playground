import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

# Hyperparams
batch_size = 32
block_size = 8
epochs = 30000
eval_interval=3000
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_epochs = 500

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()


chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: [itos[t] for t in s]

data = torch.tensor(encode(data), dtype=torch.long)
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

def get_batch(batch_size:int=4, split:str='train'):
    data = train_data if split=="train" else val_data
    # Creating 4 random offsets on the dataset
    # data would be batched from those offsets
    # This creates a random sampling of batches
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    # tensor of batch_size, block_size by stacking rows
    x = torch.stack([data[i: i+block_size] for i in ix])
    # Targets tensor offset by 1
    # if input tensor = [42, 37, 65, 80]
    # target tensor = [37, 65, 80, 28]
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class BiGramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, y=None):
        logits = self.embedding(x)
        # Slight problem with pytorch cross-entropy
        # They expect the input shape to be (B*T, C) and output shape to be (B*T) or flatten the first 2 dims
        if y==None:
            return logits, None
        
        else:
            logits = logits.flatten(0, 1)
            y = y.flatten(0, 1)
            loss = F.cross_entropy(logits, y)

        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # Forward pass with X (B, T) gives output -> (B, T+1)
            logits, loss = self(x) # (B, T, C)
            # need to slice the predicted token
            logits = logits[:, -1, :] # (B, C)
            # apply softmax on the channel
            probs = F.softmax(logits, dim=-1)
            # sample from the dist
            # say the probs are 0.06, 0.08, 0.4 ... (65 values)
            # say 0.1 is the higest value and it's corresponding index is 2.
            # so for B=0, x_next = [[2]]
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat((x, x_next), dim=1) # (B, T+1)

        return x

@torch.no_grad()
def estimate_loss(model):
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_epochs)
        for k in range(eval_epochs):
            X, Y = get_batch(split=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

m = BiGramModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

for epoch in trange(epochs, desc="Training"):

    if epoch % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"Step: {epoch}: train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

    xb, yb = get_batch(batch_size=batch_size, split='train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
out_decoded = decode(m.generate(context, 500)[0].tolist())
print(''.join(out_decoded))