import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import os
import time
import wandb
from datetime import datetime
import math

# Performance Flags
# TF32 takes the exponent of FP32(8 Bits) but Mantissa of FP16(10bits). Eg: 6.23 (Mantissa/precision) * 10^8 (exponent)
# This reduces the precision of the calculation but increases the speed.
torch.backends.cuda.matmul.allow_tf32 = True # Allows the TF32 format matmul.
torch.backends.cudnn.benchmark = True # Gives better kernel selection
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint directory
checkpoint_dir = "checkpoints/gpt-pytorch-improved"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_interval = 1000

# Hyperparams
batch_size = 64
block_size = 256
epochs = 5000
eval_interval=1000
#lr = 3e-4
max_lr = 3e-4
min_lr = 3e-5 
warmup_epochs = int(0.05 * epochs)   # 5% warmup
lr_decay_epochs = epochs


n_embed = 384
eval_epochs = 200
num_heads = 6
num_layers = 6
dropout = 0.2

wandb.init(
    project="gpt-run",
    name=f"pytorch_improved_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        'batch_size': batch_size,
        'block_size': block_size,
        'epochs': epochs,
        'eval_interval': eval_interval,
        'max_lr': max_lr,
        'min_lr': min_lr,
        'warmup_epochs': warmup_epochs,
        'lr_decay_epochs': lr_decay_epochs,
        'n_embed': n_embed,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout': dropout,
    }
)
print(f"Wandb initialized. View at: {wandb.run.url}")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join([itos[i] for i in t])

data = encode(text)
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]
print(f"Total Training tokens: {len(train_data) // 1000000:.2f}M")
print(f"Total Validation tokens: {len(val_data) // 1000:.2f}K")

def get_batch(split:str='train'):
    src = train_data if split=="train" else val_data
    # Creating random indices to batch the data eg: [100, 500, 50, ... (batch_size)]
    ix = torch.randint(0, len(src) - block_size -1, (batch_size, ))
    # Creating a tensor of offsets eg: [0, 1, 2, 3, ... (block_size)]
    offs = torch.arange(block_size)

    # Broadcasting operation
    # 1st increasing the dimension of ix to (batch_size, 1) by ix[:, None]
    # ix = [[100, 100, 100], [500, 500, 500], [50, 50, 50], ... (batch_size)]
    # 2nd step is adding the offsets to the ix tensor by ix[:, None] + offs
    # ix + offs = [[100, 101, 102, ... (block_size)], [500, 501, 502, ... (block_size)], [50, 51, 52, ... (block_size)], ... (batch_size)]
    # Just need to grab the values from the src tensor at the new indices. Eg 1 data point is [100, 101, 102, ... (block_size)]
    x = src[ix[:, None] + offs] # (batch_size, block_size)
    y = src[ix[:, None] + offs + 1] # (batch_size, block_size)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embed // num_heads
        
        self.qkv = nn.Linear(n_embed, 3*n_embed, bias=False) # Because of layer norm the effect of bias is cancelled out
        self.proj = nn.Linear(n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        qkv = self.qkv(x) # (B, T, 3*n_embed) and n_embed = head_size * num_heads
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_size).transpose(1,3) # (B, T, 3, nh, hs) --> (B, nh, 3, T, hs)
        q, k, v = qkv.unbind(dim=2) # (B, nh, T, hs)

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

        # use the scaled dot product attention for Flash Attention Kernels
        out = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=use_causal, 
            dropout_p=dropout if self.training else 0.0
            ) # (B, nh, T, hs)

        # .transpose does not move the numbers around in RAM which will give an error in .view() operation
        # it will throw-> RuntimeError: input is not contiguous
        # .contiguous() makes the tensor contiguous in memory.
        out = out.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) --> (B, T, C)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # There is a 4x expansion in the hidden dimension.
        # Using GELU instead of ReLU for better performance.
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed, bias=False),
            nn.GELU(),
            nn.Linear(4*n_embed, n_embed, bias=False),
            nn.Dropout(dropout)
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
        # 1. Layer Norm
        # 2. Multi Head Attention
        # 3. Skip Connection
        # 4. Layer Norm
        # 5. Feed Forward
        # 6. Skip Connection
        x = x+ self.mha(self.ln1(x), kv_cache)
        x = x+ self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)
        # Tying the head weights to the token embeddings to avoid the need for a separate weight matrix
        self.head.weight = self.token_embed.weight

        # GPT-2 style initialization of Linear and embedding layers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None, kv_cache=None, start_pos=0):
        B, T = x.shape
        pos = torch.arange(start_pos, start_pos + T, device=x.device)
        pos = pos.clamp(max=block_size - 1) # To prevent out of bounds error
        x = self.token_embed(x) + self.pos_embed(pos)

        for i, block in enumerate(self.blocks):
            cache = None if kv_cache is None else kv_cache[i]
            x = block(x, cache)

        x = self.ln_f(x)
        logits = self.head(x)

        if y is None:
            return logits, None

        loss = F.cross_entropy(logits.view(B*T, vocab_size), y.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()

        kv_cache = [
            {"k": None, "v": None} for _ in range(num_layers)
        ]

        for i in range(max_new_tokens):
            # For the first iteration, pass the full prompt; afterwards, only the last token
            if i == 0:
                idx_cond = idx
                start_pos = 0
            else:
                idx_cond = idx[:, -1:] # (B, 1) only the last token is needed because of kv cache
                # Clamp position to stay within block_size to avoid out-of-bounds
                start_pos = min(idx.shape[1] - 1, block_size - 1)

            logits, _ = self(idx_cond, kv_cache=kv_cache, start_pos=start_pos)
            logits = logits[:, -1]/ temperature # (B, C) 
            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)

        return idx
    
    @torch.no_grad()
    def generate2(self, idx, max_new_tokens, temperature=1.0):
        """Sliding window attention for inference"""
        self.eval()
        B = idx.shape[0]
        for _ in range(max_new_tokens):
            # Sliding window
            idx_cond = idx[:, -block_size:]  # crop to context window
            # Forward pass (no kv cache)
            logits, _ = self(idx_cond)
            # Take last token logits
            logits = logits[:, -1, :] / temperature
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample next token
            next_idx = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, next_idx), dim=1)

        return idx

# Training utils
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_epochs, device=device)
        for k in range(eval_epochs):
            X, Y = get_batch(split=split)
            _, loss = model(X, Y)
            losses[k] = loss.detach() # .item() forces gpu->cpu sync every step so use .detach() instead
        out[split] = losses.mean().item()
    model.train()
    return out

def save_checkpoint(model, optimizer, scaler, epoch, losses, filepath):
    if scaler is not None:
        scaler_state_dict = scaler.state_dict()
    else:
        scaler_state_dict = None

    if scheduler is not None:
        scheduler_state_dict = scheduler.state_dict()
    else:
        scheduler_state_dict = None

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler_state_dict,
        'scheduler_state_dict': scheduler_state_dict, 
        'train_loss': losses['train'],
        'val_loss': losses['val'],
        'hyperparams': {
            'batch_size': batch_size,
            'block_size': block_size,
            'epochs': epochs,
            'eval_interval': eval_interval,
            'max_lr': max_lr,
            'min_lr': min_lr,   
            'warmup_epochs': warmup_epochs,
            'lr_decay_epochs': lr_decay_epochs,
            'n_embed': n_embed,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scaler=None, scheduler=None):
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    print(f"Loaded checkpoint from epoch {start_epoch}")
    print(f"Train loss: {checkpoint['train_loss']:.4f} | "
          f"Val loss: {checkpoint['val_loss']:.4f}")
    print(f"Hyperparams: {checkpoint['hyperparams']}")

    return start_epoch

def get_lr(step):
    # 1. linear warmup
    if step < warmup_epochs:
        return step / max(1, warmup_epochs)

    # 2. cosine decay
    if step > lr_decay_epochs:
        return min_lr / max_lr

    # Formula already available in pytorch cosine annealing scheduler
    decay_ratio = (step - warmup_epochs) / (lr_decay_epochs - warmup_epochs)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (min_lr + coeff * (max_lr - min_lr)) / max_lr


# Training 
model = GPT().to(device)
print(f"Model parameters:  {sum(p.numel() for p in model.parameters())/1e6:.4f}M")
# torch.compile enables kernel fusion; multiple operations are fused into a single kernel for better performance.
# max-autotune is the best but very slow for the first few batches. creates custom triton kernels. This would be the fastest but RTX 3060 doesn't have enough SMs to run it.
# max-autotune-no-cudagraphs is useful if you hit OOMs
# default is balance of speed and memory usage. Using default for now.
# reduce-overhead is useful if CPU is the bottleneck
model = torch.compile(model, mode="default")

# fused=True is used to use the fused AdamW optimizer which is faster and uses less memory. Only available for certain GPUs.
# It reads the weights, gradients m, v, updates the weights in one go
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, fused=True, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)
# GradScaler is used to scale the gradients before backprop to avoid vanishing gradients with FP16 training
# Only needed for FP16 and not for BF16 training
training_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("Training dtype: ", training_dtype)
scaler = torch.amp.GradScaler("cuda") if training_dtype == torch.float16 else None

# start_epoch = load_checkpoint(path, model, optimizer, scaler, scheduler)
start = time.time()

for step in trange(epochs, desc="Training"):
#for step in range(epochs):
    t0 = time.time()
    if step% eval_interval == 0:
        losses = estimate_loss(model)
        #print(f"{step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        wandb.log(losses | {"step": step})

    x, y = get_batch('train')

    # bfloat16 is good for training this model
    with torch.amp.autocast("cuda", dtype=training_dtype):
        _, loss = model(x, y)

    # Log training loss more frequently
    # if step % 100 == 0:
    #     wandb.log({'train_loss_step': loss.item(), 'epoch': step})

    # there is a read operation before backprop to read the gradient
    # with set_to_none=True, it will set the gradient to None after adding the gradient
    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        norm =torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()
    
    t1 = time.time()
    dt = (t1-t0)*1000
    tokens_per_second = batch_size * block_size / (t1-t0)
    #print(f"Step: {step} | loss: {loss.item():.4f} | norm: {norm:.4f} | Time taken: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f}")
    wandb.log({"time_per_step_in_ms": dt, "tokens_per_second": tokens_per_second, "norm": norm.item()})

end = time.time()
print(f"Training time: {end - start:.2f}s")

final_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_final.pth")
losses = estimate_loss(model)
save_checkpoint(model, optimizer, scaler, epochs, losses, final_checkpoint_path)
wandb.finish()
print(f"Final checkpoint saved to {final_checkpoint_path}")
print(f"Training loss: {losses['train']:.4f} | Validation loss: {losses['val']:.4f}")

context = torch.zeros((1,1), dtype=torch.long, device=device)
out_decoded = decode(model.generate2(context, 500)[0].tolist())
print(out_decoded)
