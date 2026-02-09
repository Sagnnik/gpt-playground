import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import time
import wandb
import os
from datetime import datetime

# --------------- Hyperparams ---------------
batch_size = 64
block_size = 256
epochs = 5000
eval_interval=500
lr = 3e-4
n_embed = 384
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_epochs = 200
num_heads = 6
num_layers = 6
dropout = 0.2
# --------------------------------------------

# Checkpoint directory
checkpoint_dir = "checkpoints/gpt-pytorch"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_interval = 1000

# Initialize wandb
wandb.init(
    project="gpt-run",  # Change to your project name if desired
    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # Optional: name this run
    config={
        'batch_size': batch_size,
        'block_size': block_size,
        'epochs': epochs,
        'eval_interval': eval_interval,
        'lr': lr,
        'n_embed': n_embed,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout': dropout,
    }
)
print(f"Wandb initialized. View at: {wandb.run.url}")

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()


chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: ''.join([itos[t] for t in s])

data = torch.tensor(encode(data), dtype=torch.long)
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]

def get_batch(split:str='train'):
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

# This LayerNorm does work just not used in this code
# The custom layernorm params are not tracked by the optimizer
# To track them we need to inherit from nn.Module and register the params as requires_grad=True. At that point just use nn.LayerNorm instead.
# class LayerNorm:
#     def __init__(self, dim, eps=1e-5, momentum=0.1):
#         self.eps = eps
#         self.gamma = torch.ones(dim).to(device)
#         self.beta = torch.zeros(dim).to(device)

#     def __call__(self, x):
#         xmean = x.mean(dim=1, keepdim=True) # layer mean (B, 1, C)
#         xvar = x.var(dim=1, keepdim=True) # layer variance (B, 1, C)
#         xhat = (x-xmean)/torch.sqrt(xvar+self.eps)
#         self.out = self.gamma*xhat + self.beta
#         return self.out

#     def parameters(self):
#         return [self.gamma, self.beta]

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
            nn.GELU(),
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

m = BiGramModel().to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
num_params = sum(p.numel() for p in m.parameters())
print("Number of parameters:", num_params/1e6, "M")
wandb.config.update({'num_parameters': num_params})

def save_checkpoint(model, optimizer, epoch, losses, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': losses['train'],
        'val_loss': losses['val'],
        'hyperparams': {
            'batch_size': batch_size,
            'block_size': block_size,
            'epochs': epochs,
            'eval_interval': eval_interval,
            'lr': lr,
            'n_embed': n_embed,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']+1
    print(f"Loaded checkpoint from epoch {start_epoch}")
    print(f"Training loss: {checkpoint['train_loss']:.4f} | Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Hyperparams: {checkpoint['hyperparams']}")
    return start_epoch

time_start = time.time()
for epoch in trange(epochs, desc="Training"):

    if epoch % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"{epoch}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        wandb.log(losses | {"step": epoch})

        # Save checkpoint
        # if epoch>0 and epoch % checkpoint_interval == 0:
        #     checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        #     save_checkpoint(m, optimizer, epoch, losses, checkpoint_path)

    xb, yb = get_batch(split='train')
    logits, loss = m(xb, yb)

    # Log training loss more frequently
    # if epoch % 100 == 0:
    #     wandb.log({'train_loss_step': loss.item(), 'epoch': epoch})
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

time_end = time.time()
print(f"Training Time: {time_end - time_start} seconds")
final_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_final.pth")
losses = estimate_loss(m)
save_checkpoint(m, optimizer, epochs, losses, final_checkpoint_path)
wandb.finish()
print(f"Final checkpoint saved to {final_checkpoint_path}")
print(f"Training loss: {losses['train']:.4f} | Validation loss: {losses['val']:.4f}")

context = torch.zeros((1,1), dtype=torch.long, device=device)
out_decoded = decode(m.generate(context, 500)[0].tolist())
print(out_decoded)