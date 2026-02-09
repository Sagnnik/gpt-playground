import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Setup 
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = "checkpoints/gpt-pytorch-improved/checkpoint_final.pth"

def strip_compile_prefix(state_dict):
    return {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }


# Load data / vocab
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return "".join([itos[i] for i in t])

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
hp = checkpoint["hyperparams"]

batch_size   = hp["batch_size"]
block_size   = hp["block_size"]
n_embed      = hp["n_embed"]
num_heads    = hp["num_heads"]
num_layers   = hp["num_layers"]
dropout      = hp["dropout"]

# Model
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embed // num_heads

        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_size).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)

        # Determine if we should use causal masking
        # Only use is_causal when cache is empty (first forward pass)
        use_causal = kv_cache is None or kv_cache["k"] is None

        if kv_cache is not None:
            if kv_cache["k"] is None:
                kv_cache["k"], kv_cache["v"] = k, v
            else:
                kv_cache["k"] = torch.cat([kv_cache["k"], k], dim=2)
                kv_cache["v"] = torch.cat([kv_cache["v"], v], dim=2)
            k, v = kv_cache["k"], kv_cache["v"]

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=use_causal,
            dropout_p=0.0
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed, bias=False),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mha = MultiHeadAttention()
        self.ffn = FeedForward()

    def forward(self, x, kv_cache=None):
        x = x + self.mha(self.ln1(x), kv_cache)
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, x, kv_cache=None, start_pos=0):
        B, T = x.shape
        pos = torch.arange(start_pos, start_pos + T, device=x.device)
        x = self.token_embed(x) + self.pos_embed(pos)

        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache[i] if kv_cache else None)

        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()

        kv_cache = [{"k": None, "v": None} for _ in range(num_layers)]

        for i in range(max_new_tokens):
            # For the first iteration, pass the full prompt; afterwards, only the last token
            if i == 0:
                idx_cond = idx
                start_pos = 0
            else:
                idx_cond = idx[:, -1:]
                # Clamp position to stay within block_size to avoid out-of-bounds
                start_pos = min(idx.shape[1] - 1, block_size - 1)

            logits = self(idx_cond, kv_cache=kv_cache, start_pos=start_pos)
            logits = logits[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)

        return idx

# Run inference
model = GPT().to(device)
raw_state_dict = checkpoint["model_state_dict"]
clean_state_dict = strip_compile_prefix(raw_state_dict)
model.load_state_dict(clean_state_dict)
model.eval()

prompt = "ELIOT"
context = encode(prompt).unsqueeze(0).to(device)

with torch.no_grad():
    output = model.generate(context, max_new_tokens=500, temperature=1.0)
    print(decode(output[0].tolist()))
