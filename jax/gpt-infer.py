import os
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.linen as nn
import orbax.checkpoint as ocp
from dataclasses import dataclass
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_gemm=true "
    "--xla_gpu_enable_cudnn_fmha=true "
)

jax.config.update("jax_default_matmul_precision", "bfloat16")

PARAM_DTYPE = jnp.float32
COMPUTE_DTYPE = jnp.bfloat16

@dataclass
class GPTConfig:
    block_size: int = 256
    batch_size: int = 64
    n_embed: int = 384
    num_heads: int = 6
    num_layers: int = 6
    dropout: float = 0.2



# MODEL
class FeedForward(nn.Module):
    n_embed: int
    dropout: float

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Dense(4 * self.n_embed, use_bias=False,
                     dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.n_embed, use_bias=False,
                     dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x)
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
        return x


class MultiHeadAttention(nn.Module):
    n_embed: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x, train=True):
        B, T, C = x.shape
        hs = C // self.num_heads

        qkv = nn.Dense(3 * C, use_bias=False,
                       dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, hs)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        mask = jnp.tril(jnp.ones((1, 1, T, T), dtype=bool))

        out = nn.dot_product_attention(
            q, k, v,
            mask=mask,
            deterministic=not train,
            dropout_rate=self.dropout if train else 0.0,
            dropout_rng=self.make_rng("dropout"),
        )

        out = out.reshape(B, T, C)
        out = nn.Dense(self.n_embed, use_bias=False,
                       dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(out)
        out = nn.Dropout(rate=self.dropout, deterministic=not train)(out)
        return out


class Block(nn.Module):
    n_embed: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x, train=True):
        x = x + MultiHeadAttention(
            self.n_embed, self.num_heads, self.dropout
        )(nn.LayerNorm()(x), train=train)

        x = x + FeedForward(self.n_embed, self.dropout)(
            nn.LayerNorm()(x), train=train
        )
        return x


class GPT(nn.Module):
    vocab_size: int
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train=True):
        B, T = x.shape

        tok = nn.Embed(self.vocab_size, self.config.n_embed,
                       dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x)

        pos = nn.Embed(self.config.block_size, self.config.n_embed,
                       dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(
            jnp.arange(T)
        )

        x = tok + pos

        for _ in range(self.config.num_layers):
            x = Block(self.config.n_embed, self.config.num_heads,
                      self.config.dropout)(x, train=train)

        x = nn.LayerNorm()(x)
        # Need to refer to the exact embedding matrix
        embed_weights = self.variables["params"]["Embed_0"]["embedding"]
        logits = x @ embed_weights.T
        return logits

def generate(model, params, key, idx, max_new_tokens, config, temperature=1.0):
    for _ in range(max_new_tokens):

        if idx.shape[1] < config.block_size:
            pad = config.block_size - idx.shape[1]
            idx_cond = jnp.pad(idx, ((0, 0), (pad, 0)))
        else:
            idx_cond = idx[:, -config.block_size:]

        key, dkey = jrandom.split(key)
        logits = model.apply(params, idx_cond, train=False,
                             rngs={"dropout": dkey})

        logits = logits[:, -1] / temperature

        key, sk = jrandom.split(key)
        next_token = jrandom.categorical(sk, logits)[:, None]
        idx = jnp.concatenate([idx, next_token], axis=1)

    return idx

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

def encode(s):
    return jnp.array([stoi[c] for c in s], dtype=jnp.int32)

def decode(t):
    return "".join([itos[int(i)] for i in t])


# LOAD CHECKPOINT
ckpt_dir = str(Path("checkpoints/gpt-jax").resolve())

manager = ocp.CheckpointManager(
    ckpt_dir,
    item_handlers={
        "params": ocp.PyTreeCheckpointHandler(),
        "opt_state": ocp.PyTreeCheckpointHandler(),
        "step": ocp.JsonCheckpointHandler(),
    },
)

latest = manager.latest_step()
print("Loading checkpoint step:", latest)

restored = manager.restore(latest)
params = restored["params"]

config = GPTConfig()
vocab_size = len(chars)

model = GPT(vocab_size=vocab_size, config=config)

key = jrandom.PRNGKey(0)

prompt = "ELIOT"
context = encode(prompt)[None, :]

generated = generate(model, params, key, context, 500, config)

print("\n=== GENERATED TEXT ===\n")
print(decode(generated[0]))
