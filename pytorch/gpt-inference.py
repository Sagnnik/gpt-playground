import torch
import torch.nn as nn
import torch.nn.functional as F
import os
torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: ''.join([itos[t] for t in s])

# Load checkpoint
checkpoint_path = "checkpoints/gpt-pytorch/checkpoint_final.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Set hyperparams from checkpoint
batch_size = checkpoint['hyperparams']['batch_size']
block_size = checkpoint['hyperparams']['block_size']
n_embed = checkpoint['hyperparams']['n_embed']
num_heads = checkpoint['hyperparams']['num_heads']
num_layers = checkpoint['hyperparams']['num_layers']
dropout = checkpoint['hyperparams']['dropout']

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed//n_head  # 4 heads so each of size 8. when concatenated will give C=32
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x+ self.sa(self.ln1(x))
        x = x+ self.ffw(self.ln2(x))
        return x

class BiGramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # self.sa_head = MultiHeadAttention(num_heads, n_embed//num_heads) 
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)

    def forward(self, x, y=None):
        B, T = x.shape
        
        pos = torch.arange(T, device=device)
        pos_embed = self.position_embedding(pos)
        pos_embed = pos_embed.unsqueeze(0)

        x_embed = self.token_embedding(x)
        x = x_embed + pos_embed

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if y is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.view(B*T)
        loss = F.cross_entropy(logits, y)

        return logits, loss

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # Forward pass with X (B, T) gives output -> (B, T+1)
            # need to crop the x since the positional enbedding is upto block size
            x = x[:, -block_size:]
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

model = BiGramModel().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
context = "ELIOT"
context = torch.tensor(encode(context), dtype=torch.long, device=device)
context = context.unsqueeze(0)
max_new_tokens = 500

with torch.no_grad():
    generated = model.generate(context, max_new_tokens)
    text = decode(generated[0].tolist())
    print(text)