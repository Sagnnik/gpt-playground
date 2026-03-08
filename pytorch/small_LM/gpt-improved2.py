import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from dataclasses import dataclass
import os
import time
import wandb
from datetime import datetime

# Performance Flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.cuda.manual_seed_all(1337)




# Probable config for small LM (based on LFM2.5):
# {
# "total_params": 285M,
# "layers": 16 Total (10 Gated Conv + 6 GQA),
# "d_model": 1024,
# "intermediate_dim": 3584 (3.5x expansion for SwiGLU. Optimized for 256-bit alignment) 
# "attention_heads": 16Q / 4KV
# "vocab_size": 50304(gpt-2) or 68k (Sarvam tokenizer)
# "weight_tying": true,
# "max_context": 4096
# }
# use xavier initialization for internal gates of conv
# residual scaling for final linear layers in every block. Initialize weights to a std of 0.02/sqrt(2 * layers)

# Hyperparams
@dataclass
class ModelConfig:
    batch_size: int = 24
    seq_length: int = 256
    vocab_size: int = 50304
    n_embed: int = 384
    num_heads: int = 8
    n_layer: int = 4
    num_kv_heads: int = 4 # For Grouped Query Attention
    sliding_window: int = 4096
    intermediate_dim: 3584 # SwiGLU expansion
    qk_norm: bool = True
    query_pre_attn_scalar: float = 1.0

    # Training configs
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01
    weight_decay: float = 0.1
    dropout: float = 0.2
    grad_clip: float = 1.0
    use_amp: bool = True

    # Eval Configs
    eval_interval: int = 1000
    eval_epochs: int = 200

config = ModelConfig()
# Loading dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt-2")
tokens = torch.tensor(enc.encode(text), dtype=torch.long)

# enc = tiktoken.get_encoding("gpt-2")
# vocab_size = enc.n_vocab
# print(f"Vocab Size: {vocab_size}")
# import sys; sys.exit()

# def encode(s): return torch.tensor(enc.encode(s), dtype=torch.long)
# def decode(t): return enc.decode(t.tolist())

# data = encode(text)
# split = int(0.9*len(data))
# train_data = data[:split]
# val_data = data[split:]
# print(f"Total Training tokens: {len(train_data) // 1000:.2f}K")
# print(f"Total Validation tokens: {len(val_data) // 1000:.2f}K")

class TinyShakespeareDataset(Dataset):
    def __init__(self, tokens, split='train', block_size=128, train_ratio=0.9):
        self.text = text
        self.tokens = tokens

        n = int(train_ratio * len(self.tokens))
        if split == 'train':
            self.data = self.tokens[:n]
        else:
            self.data = self.tokens[n:]
            
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, index):
        chunk = self.data[index:index+self.block_size+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
    

# Initialize Datasets
train_ds = TinyShakespeareDataset(tokens, split='train', block_size=config.seq_length)
val_ds = TinyShakespeareDataset(tokens, split='val', block_size=config.seq_length)

# Initialize DataLoaders
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

"""
Functional implementation of RoPE:

STEP 1: Angle definition
theta(p, i) = p / 10000^(2i/d)

STEP 2: Pairwise Rotation
q = [x1 x2 x3 x4]
we group, 
i=0 -> (x1 x2) and i=1 -> (x3 x4)

for pair i,
[x1'] = Ri . [x1]  
[x2']        [x2]

=> [x1'] = [x1cos(theta) - x2sin(theta)]  
   [x2']   [x1sin(theta) + x2cos(theta)]

def get_inv_freq(head_dim, base=10000):
    # Calculating the w(i) value
    n_pairs = head_dim // 2 # Number of pairs
    i = torch.arange(n_pairs, dtype=torch.float32) # vector of pairs
    return 1.0/(base ** (i/n_pairs))

def get_angles(seq_len, inv_freq):
    # Calculating the angles
    pos = torch.arange(seq_len, dtype=torch.float32)
    return torch.outer(pos, inv_freq)

def apply_rope(x, inv_freq):
    # x shape: (batch, num_heads, seq_len, dim)
    B, H, L, D = x.shape
    n_pairs = D // 2
    inv_freq = get_inv_freq(D, 10000)
    angles = get_angles(L, inv_freq)

    sin = torch.sin(angles)[None, None, :, :] # (1, 1, seq_len, head_dim//2)
    cos = torch.cos(angles)[None, None, :, :] # (1, 1, seq_len, head_dim//2)

    # Split the X into two parts
    x1 = x[..., :n_pairs]
    x2 = x[..., n_pairs:]

    # Calculating the xRi matrix
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos

    return torch.cat([rot1, rot2], dim=-1).to(dtype=x.dtype)
"""
# RoPE class
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000, max_positions=2048):
        super().__init__()
        assert head_dim % 2 == 0, "Head dimension must be even"
        self.head_dim = head_dim
        self.base = base
        self.max_positions = max_positions

        # Precompute Inverse Frequencies
        half_dim = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(half_dim, dtype=torch.float32) / half_dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute sin/cos cache
        self.build_cache(max_positions)

    def build_cache(self, seq_len):
        positions = torch.arange(seq_len, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x, position_offset=0):
        # x shape: (batch, num_heads, seq_len, head_dim)
        B, H, L, D = x.shape
        assert D == self.head_dim

        # Extending the cache if necessary
        if position_offset + L > self.cos_cached.shape[0]:
            self.build_cache(position_offset + L)

        # This selects exactly the angles corresponding to the true token positions (useful for KV-cache)
        cos = self.cos_cached[position_offset:position_offset+L]
        sin = self.sin_cached[position_offset:position_offset+L]

        cos = cos[None, None, :, :] # (1, 1, seq_len, head_dim//2)
        sin = sin[None, None, :, :] # (1, 1, seq_len, head_dim//2)

        # Split the X into two parts
        x1 = x[..., :D//2]
        x2 = x[..., D//2:]

        # Calculating the xRi matrix
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos

        return torch.cat([rot1, rot2], dim=-1).to(dtype=x.dtype)

"""
    RMSNorm

    From pytorch docs,
    
    y(i) = x(i) * gamma(i) / RMS(x)

    where, 
    RMS(x) = sqrt(1/n*sum(x^2) + eps)
=>  RMS(x) = sqrt(mean(x^2) + eps)

    BatchNorm -> normalizes across the batch dimension
    LayerNorm -> normalizes across the feature dimension
    RMSNorm -> like LayerNorm, but without subtracting the mean

    Mean shift is unecessary as the model can learn to adapt to the mean shift
"""   
class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps

        # Pytorch initializes gamma with ones. Google's models use zero initialization and later uses (1.0 + self.gamma)
        self.gamma = nn.Parameter(torch.zeros(n_embed))
        # This is optional as LLMs generally do not use bias
        self.bias = nn.Parameter(torch.zeros(n_embed)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        x_f = x.float() # float32
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps) # rsqrt = 1/sqrt
        
        out = x_norm * (1.0 + self.gamma.float())

        if self.bias is not None:
            out += self.bias.float()

        return out.to(input_dtype)

# Qwen3ForCausalLM(
#   (model): Qwen3Model(
#     (embed_tokens): Embedding(151936, 1024)
#     (layers): ModuleList(
#       (0-27): 28 x Qwen3DecoderLayer(
#         (self_attn): Qwen3Attention(
#           (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
#           (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
#           (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
#           (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
#         )
#         (mlp): Qwen3MLP(
#           (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)
#           (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
#           (down_proj): Linear(in_features=3072, out_features=1024, bias=False)
#           (act_fn): SiLUActivation()
#         )
#         (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
#         (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
#       )
#     )
#     (norm): Qwen3RMSNorm((1024,), eps=1e-06)
#     (rotary_emb): Qwen3RotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
# )
# Lfm2DecoderLayer(
#   (self_attn): Lfm2Attention(
#     (q_proj): Linear(in_features=1024, out_features=1024, bias=False)
#     (k_proj): Linear(in_features=1024, out_features=512, bias=False)
#     (v_proj): Linear(in_features=1024, out_features=512, bias=False)
#     (out_proj): Linear(in_features=1024, out_features=1024, bias=False)
#     (q_layernorm): Lfm2RMSNorm((64,), eps=1e-05)
#     (k_layernorm): Lfm2RMSNorm((64,), eps=1e-05)
#   )
#   (feed_forward): Lfm2MLP(
#     (w1): Linear(in_features=1024, out_features=4608, bias=False)
#     (w3): Linear(in_features=1024, out_features=4608, bias=False)
#     (w2): Linear(in_features=4608, out_features=1024, bias=False)
#   )
#   (operator_norm): Lfm2RMSNorm((1024,), eps=1e-05)
#   (ffn_norm): Lfm2RMSNorm((1024,), eps=1e-05)
# )

class GroupedQueryAttention(nn.Module):
    """
    GQA reduces memory by sharing the key-value heads across multiple query heads
    This has similar performace with MHA but more efficient

    1. QK-Norm
        Q' = RMSNorm(Q)
        K' = RMSNorm(K)
        Attention = Q'K'T/sqrt(d_in)

        Even with scale factor RMSNorm is required
        e.g. head_dim = 64
        scaling = 1/8
        ||Q|| = 30, ||K|| = 30
        dot(Q, K) = 900
        scaled ~ 112
        with RMSNorm magnitude remains stable
    """

    def __init__(self, d_in, num_heads, num_kv_heads, head_dim=None, qk_norm=False, query_pre_attn_scalar=None):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_groups"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.rope = RotaryEmbedding(head_dim)

        if head_dim is None:
            assert d_in % num_heads == 0, "d_in must be divisible by num_heads"
            head_dim = d_in // num_heads
        
        self.head_dim = head_dim

        # Q => num_heads, K => num_kv_heads, V => num_kv_groups
        # k_proj and v_proj can be done in 1 linear layer
        self.q_proj = nn.Linear(d_in, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_in, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_in, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, d_in, bias=False)

        if qk_norm:
            
            self.q_norm = RMSNorm(head_dim) # Q' = RMSNorm(Q)
            self.k_norm = RMSNorm(head_dim) # K' = RMSNorm(K)
        else:
            self.q_norm = self.k_norm = None

        # This controls the attention scaling factor (variance control) or 1/sqrt(d)
        # Some models experiment with this but small models like this does not need it
        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar ** -0.5
        else:
            self.scaling = head_dim ** -0.5

    def forward(self, x, mask):
        # x shape: (B, T, d_in)

        b, num_tokens, _ = x.shape

        queries = self.q_proj(x) # (B, T, num_heads * head_dim)
        keys = self.k_proj(x) # (B, T, num_kv_heads * head_dim)
        values = self.v_proj(x) # (B, T, num_kv_heads * head_dim)

        # Reshape to (B, num_heads/num_kv_heads, T, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        # Apply RoPE to queries and keys
        queries = self.rope(queries)
        keys = self.rope(keys)

        # Repeat keys and values for each group(GQA)
        keys = keys.repeat_interleave(self.group_size, dim=1) # (B, num_kv_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1) # (B, num_kv_heads, T, head_dim) -> (B, num_heads, T, head_dim)

        # Scaling queries (1/sqrt(d))
        queries = queries * self.scaling

        out = F.scaled_dot_product_attention(queries, keys, values, is_causal=True) # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, num_tokens, -1) # (B, T, num_heads * head_dim)
        out = self.o_proj(out)
        return out


class FeedForward(nn.Module):
    """
    Standard Transformer MLP:
    FFN(x) = Linear(GELU(Linear(x)))

    Modern LLMs replace this with a gated unit:
    FFN(x) = W2 ( SiLU(W1x) * (W3x) )
    
    swish(x) = x * sigmoid(beta*x) with beta=1 = x*sigmoid(x)
    sigmoid(x) = 1/(1+e^-x)

    Process:
    gate -> w1(x)
    value -> w2(x)
    swiglu = w3(silu(gate) * value)

    As the authors described: "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."
    but the intuition could be that instead of a single activation pathway SwiGLU creates a multiplicative gate similar to LSTM gating
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden_dim = cfg.intermediate_dim
        # w1 and w3 can be done in 1 linear layer
        self.w1 = nn.Linear(cfg.n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, cfg.n_embed, bias=False)
        self.w3 = nn.Linear(cfg.n_embed, hidden_dim, bias=False)

    def forward(self, x):
        gate = self.w1(x)
        value = self.w3(x)
        swiglu = self.w2(F.silu(gate) * value)
        return swiglu
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg:ModelConfig, attn_type:str):
        super().__init__()
        self.attn_type = attn_type # 'attention' or 'conv'
        self.attention = GroupedQueryAttention(
            d_in=cfg.n_embed,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            qk_norm=cfg.qk_norm,
            query_pre_attn_scalar=cfg.query_pre_attn_scalar,
        )

        self.ffn = FeedForward(cfg)
        self.input_layernorm = RMSNorm(n_embed=cfg.n_embed)
        self.post_attention_layernorm = RMSNorm(n_embed=cfg.n_embed)
        self.pre_feedforward_layernorm = RMSNorm(n_embed=cfg.n_embed)
        self.post_feedforward_layernorm = RMSNorm(n_embed=cfg.n_embed)

    def forward(self, x):
        pass


