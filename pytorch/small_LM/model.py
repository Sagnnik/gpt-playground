import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import tiktoken
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from dataclasses import dataclass, field
import os
import time
import wandb
from datetime import datetime
from copy import deepcopy


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
# "vocab_size": 50304(gpt-2) or 65536 (LFM-2.5 tok) or 68k (Sarvam tok)
# "weight_tying": true,
# "max_context": 4096
# }
# use xavier initialization for internal gates of conv
# residual scaling for final linear layers in every block. Initialize weights to a std of 0.02/sqrt(2 * layers)

# Hyperparams
@dataclass
class ModelConfig:
    vocab_size: int = 50304
    seq_length: int = 512
    n_embed: int = 1024
    intermediate_dim: int = 3584      # ~3.5x expansion, 256-bit aligned
    num_heads: int = 16
    num_kv_heads: int = 8
    num_layers: int = 16
    head_dim: int = 64
 
    # Architecture
    sliding_window: int = 512
    max_position_embeddings: int = 4096
    norm_eps: float = 1e-5
    qk_norm: bool = True
    query_pre_attn_scalar: float = 1.0
 
    # RoPE
    rope_theta_local: float = 10000.0
    rope_theta_global: float = 1000000.0
 
    # FFN
    use_swiglu: bool = True

    
    layer_types: list = field(default_factory=lambda: [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ])
 
    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    dropout: float = 0.0
 
 
config = ModelConfig()

# Loading dataset
if not os.path.exists('input.txt'):
    print("Downloading tiny shakespeare dataset...")
    import requests
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open('input.txt', 'w', encoding='utf-8') as f:
        f.write(requests.get(url).text)
 
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
 
enc = tiktoken.get_encoding("gpt2")
tokens = torch.tensor(enc.encode(text), dtype=torch.long)
 
split = int(0.9 * len(tokens))
print(f"Total Training tokens: {split // 1000:.2f}K")
print(f"Total Validation tokens: {(len(tokens) - split) // 1000:.2f}K")

class TinyShakespeareDataset(Dataset):
    def __init__(self, tokens, split='train', train_ratio=0.9, block_size=512):
        self.tokens = tokens
        self.block_size = block_size

        n = int(train_ratio * len(self.tokens))
        if split == 'train':
            self.data = self.tokens[:n]
        else:
            self.data = self.tokens[n:]

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
    def __init__(self, head_dim, base=10000, max_positions=4096):
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
        self._build_cache(max_positions)

    def _build_cache(self, seq_len):
        positions = torch.arange(seq_len, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x:torch.Tensor, position_offset=0):
        # x shape: (batch, num_heads, seq_len, head_dim)
        B, H, L, D = x.shape
        assert D == self.head_dim

        # Extending the cache if necessary
        if position_offset + L > self.cos_cached.shape[0]:
            self._build_cache(position_offset + L)

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

        return torch.cat([rot1, rot2], dim=-1).to(x.dtype)

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
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.gamma.float())
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(input_dtype)

class GroupedQueryAttention(nn.Module):
    """
    GQA reduces memory by sharing the key-value heads across multiple query heads
    This has similar performace with MHA but more efficient

    -> QK-Norm
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

    def __init__(self, 
        d_in, 
        num_heads, 
        num_kv_heads, 
        head_dim=None,
        qk_norm=False, 
        rope_base=10000.0, 
        query_pre_attn_scalar=None):

        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_groups"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        

        if head_dim is None:
            assert d_in % num_heads == 0, "d_in must be divisible by num_heads"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        # Each GQA head has its own RoPE and it's own buffers
        self.rope = RotaryEmbedding(head_dim, base=rope_base)
        

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

        B, T, _ = x.shape

        queries = self.q_proj(x) # (B, T, num_heads * head_dim)
        keys = self.k_proj(x) # (B, T, num_kv_heads * head_dim)
        values = self.v_proj(x) # (B, T, num_kv_heads * head_dim)

        # Reshape to (B, num_heads/num_kv_heads, T, head_dim)
        queries = queries.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        # Apply RoPE to queries and keys
        queries = self.rope(queries)
        keys = self.rope(keys)

        # Repeat keys and values for each group(GQA)
        keys = keys.repeat_interleave(self.group_size, dim=1) # (B, num_kv_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1) # (B, num_kv_heads, T, head_dim) -> (B, num_heads, T, head_dim)

        # Scaling queries (1/sqrt(d))
        queries = queries * self.scaling

        if mask is not None:
            out = F.scaled_dot_product_attention(queries, keys, values, attn_mask=mask, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(queries, keys, values, is_causal=True) # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1) # (B, T, num_heads * head_dim)
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
        self.w1 = nn.Linear(cfg.n_embed, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim,  cfg.n_embed, bias=False)  # output
        self.w3 = nn.Linear(cfg.n_embed, hidden_dim, bias=False)  # value

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg:ModelConfig, attn_type:str):
        super().__init__()
        self.attn_type = attn_type #  'sliding_attention' or 'full_attention' or 'conv'

        rope_base = cfg.rope_theta_local if attn_type == "sliding_attention" else cfg.rope_theta_global

        self.attention = GroupedQueryAttention(
            d_in=cfg.n_embed,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            qk_norm=cfg.qk_norm,
            query_pre_attn_scalar=cfg.query_pre_attn_scalar,
            head_dim=cfg.head_dim,
            rope_base=rope_base
        )

        self.ffn = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg.n_embed, eps=cfg.norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.n_embed, eps=cfg.norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(cfg.n_embed, eps=cfg.norm_eps)
        self.post_feedforward_layernorm = RMSNorm(cfg.n_embed, eps=cfg.norm_eps)

        # LFM uses Residual scaling for every last linear layer
        # Residual scaling: std = 0.02 / sqrt(2 * num_layers)
        residual_std = 0.02 / (2 * cfg.num_layers) ** 0.5
        nn.init.normal_(self.attention.o_proj.weight, std=residual_std)
        nn.init.normal_(self.ffn.w2.weight, std=residual_std)

    def forward(self, x, mask_global, mask_local):
        attn_mask = mask_local if self.attn_type == "sliding_attention" else mask_global
        
        res = x
        x   = self.input_layernorm(x)
        x   = self.attention(x, attn_mask)
        x   = self.post_attention_layernorm(x)
        x   = res + x

        res = x
        x   = self.pre_feedforward_layernorm(x)
        x   = self.ffn(x)
        x   = self.post_feedforward_layernorm(x)
        x   = res + x
        return x

class GPT(nn.Module):
    def __init__(self, cfg:ModelConfig):
        super().__init__()
        assert cfg.layer_types is not None
        assert len(cfg.layer_types) == cfg.num_layers
        self.cfg = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embed)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, layer_type) for layer_type in cfg.layer_types
        ])

        self.final_norm = RMSNorm(cfg.n_embed, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_embeddings.weight
    #     self.apply(self._init_weights())

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_mask(self, seq_len, device):
        """
        Returns additive float masks (0.0 = attend, -inf = block) for SDPA.

        mask_global: causal (can't see future)
            j:  0 1 2 3 4 5 6 7
         i
            0:  0 1 1 1 1 1 1 1
            1:  0 0 1 1 1 1 1 1
            2:  0 0 0 1 1 1 1 1
            3:  0 0 0 0 1 1 1 1
            4:  0 0 0 0 0 1 1 1
            5:  0 0 0 0 0 0 1 1
            6:  0 0 0 0 0 0 0 1
            7:  0 0 0 0 0 0 0 0

        far_past (too far back is masked: i - j >= sliding_window)
        where sliding_window = 4
            j:  0 1 2 3 4 5 6 7
         i
            0:  0 0 0 0 0 0 0 0
            1:  0 0 0 0 0 0 0 0
            2:  0 0 0 0 0 0 0 0
            3:  0 0 0 0 0 0 0 0
            4:  1 0 0 0 0 0 0 0
            5:  1 1 0 0 0 0 0 0
            6:  1 1 1 0 0 0 0 0
            7:  1 1 1 1 0 0 0 0

        Local (sliding_window): causal + sliding window (can't see too-far past either)
        mask_local
            j:  0 1 2 3 4 5 6 7
        i
        0:      0 1 1 1 1 1 1 1
        1:      0 0 1 1 1 1 1 1
        2:      0 0 0 1 1 1 1 1
        3:      0 0 0 0 1 1 1 1
        4:      1 0 0 0 0 1 1 1
        5:      1 1 0 0 0 0 1 1
        6:      1 1 1 0 0 0 0 1
        7:      1 1 1 1 0 0 0 0
        """

        # Create a ones matrix
        ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        # Creat causal attention mask or global mask
        causal = torch.triu(ones, diagonal=1) # future blocked
        # Create a upper triangular matrix with a sliding window size then transpose it to get the mask for past tokens not inside sliding window
        far_past = torch.triu(ones, diagonal=self.cfg.sliding_window).T  # too-far past blocked
        # Combine the two masks to get the final local mask
        local_bool = causal | far_past

        # Need to make it suitable for torch SDPA
        NEG_INF = float('-inf')
        mask_global = torch.zeros(seq_len, seq_len, device=device)
        mask_global.masked_fill_(causal, NEG_INF)

        mask_local = torch.zeros(seq_len, seq_len, device=device)
        mask_local.masked_fill_(local_bool, NEG_INF)

        return mask_global, mask_local
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape

        # Gemma uses embedding scaling: x * sqrt(d_model)
        # LFM/Qwen doesn't
        x = self.tok_embeddings(x) * (self.cfg.n_embed ** 0.5)
        #x = self.tok_embeddings(x)
        mask_global, mask_local = self._create_mask(T, x.device)

        for block in self.blocks:
            x = block(x, mask_global=mask_global, mask_local=mask_local)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# Test
if __name__ == "__main__":

    test_config = ModelConfig(
        vocab_size=50304,
        seq_length=64,
        n_embed=128,
        intermediate_dim=256,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        head_dim=32,
        batch_size=2,
        layer_types= ["sliding_attention","full_attention"]
    )

    print("Initialising model...")
    model = GPT(test_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    dummy_input = torch.randint(
        0,
        test_config.vocab_size,
        (test_config.batch_size, test_config.seq_length),
        device=device
    )

    print("Testing forward pass...")
    logits, loss = model(dummy_input, targets=dummy_input)

    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("All checks passed!")