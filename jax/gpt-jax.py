import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.linen as nn
import optax
import orbax.checkpoint as ocp
from dataclasses import dataclass
import time
import wandb
from datetime import datetime
import tqdm
import os
from pathlib import Path

# JAX Performance Flags
# Enable XLA optimizations for GPU
# These flags are useful for nvidia GPUs
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_gemm=true '
    '--xla_gpu_enable_cudnn_fmha=true '
)
# Ensure 32-bit mode (not 64-bit which is slower)
# jax.config.update('jax_enable_x64', False)
# Set default matmul precision to bfloat16 for speed
jax.config.update('jax_default_matmul_precision', 'bfloat16')
# Enable persistent compilation cache (speeds up subsequent runs)
# jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')
# jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
PARAM_DTYPE = jnp.float32 # Weights are stored in float32
COMPUTE_DTYPE = jnp.bfloat16 # Computations are performed in bfloat16
# This matches PyTorch's autocast behavior: matmuls, convs, and LayerNorm in bfloat16
# Reference: https://docs.pytorch.org/docs/stable/amp.html#cpu-ops-that-can-autocast-to-bfloat16

# Hyperparameters
@dataclass
class GPTConfig:
    block_size: int = 256
    batch_size: int = 64
    n_embed: int = 384
    num_heads: int=6
    num_layers: int=6
    dropout: float=0.2

@dataclass
class TrainingConfig:
    epochs: int = 5000
    eval_interval: int = 1000
    eval_epochs: int = 200
    max_lr: float = 3e-4
    min_lr: float = 3e-5 
    lr_decay_epochs: int = epochs
    seed: int = 1337
    warmup_ratio: float = 0.05

    @property
    def warmup_epochs(self):
        return int(self.warmup_ratio * self.epochs)

# Some notes on Modules in Jax
# setup() function can be used to initialize layers like pytorch __init__()
# Better to use setup if you need to refer the same layer
# nn.compact allows lazy loading thus keeping the funtion pure (no hidden states)
# determinitic is used to toggle between model.train() and model.eval() mode. Flax passes a boolean value to the function.

# When you call Module.init() or Module.apply() you are calling the __call__ function
# you also pass a "root" key
# you can use self.make_rng() that pulls the rng key from the modules RNG keys collection
# All you need to do is pass {"dropout": dropout_key} to the apply function and it picks it up automatically. The key should be "dropout"

# In jax whenever you call rand or equivalent functions you need to pass a key to keep track of the rng state. Dropout falls under this category.

# Model
class FeedForward(nn.Module):
    n_embed: int
    dropout: float
    
    @nn.compact
    def __call__(self, x, train: bool=True):
        # Only chaining is allowed no need for early initialization. Similar to Tensorflow
        x = nn.Dense(
            features=4*self.n_embed, 
            use_bias=False, 
            dtype=COMPUTE_DTYPE, 
            param_dtype=PARAM_DTYPE, 
            kernel_init=nn.initializers.normal(stddev=0.02))(x)
        x = nn.gelu(x) # Function not a module
        x = nn.Dense(
            features=self.n_embed, 
            use_bias=False, 
            dtype=COMPUTE_DTYPE, 
            param_dtype=PARAM_DTYPE, 
            kernel_init=nn.initializers.normal(stddev=0.02))(x)
        # Deterministic is used to avoid dropout when in evaluation mode
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x) 
        return x

class MultiHeadAttention(nn.Module):
    n_embed: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x, train: bool = True):
        B, T, C = x.shape
        assert self.n_embed % self.num_heads == 0
        assert C == self.n_embed
        head_size = self.n_embed // self.num_heads

        # QKV projection
        # Using fused qkv for one big matmul instead of three separate matmuls.
        qkv = nn.Dense(
            features= 3*C, 
            use_bias=False, 
            dtype=COMPUTE_DTYPE, 
            param_dtype=PARAM_DTYPE, 
            kernel_init=nn.initializers.normal(stddev=0.02))(x)

        qkv = qkv.reshape(B, T, 3, self.num_heads, head_size) # (B, T, 3, nh, hs)
        # jax.nn.dot_product_attention expects (B, T, nh, hs) — same layout, no transpose.
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2] # each (B, T, nh, hs)

        # Use JAX's native dot_product_attention for Flash Attention via cuDNN.
        # Flax's nn.dot_product_attention is a manual einsum implementation that
        # materializes the full (T, T) attention matrix — O(T^2) memory, no Flash Attention.
        # jax.nn.dot_product_attention with is_causal=True dispatches to cuDNN Flash Attention
        # kernels that run in O(T) memory. This matches PyTorch's F.scaled_dot_product_attention.
        # Note: attention-weight dropout is not supported here — we rely on the projection
        # dropout below (nn.Dropout after the output Dense), which is the modern convention.
        # If you want dropout you could use the jax.linen.dot_product_attention wrapper
        # But it doesn't have the Flash Attention support.
        mask = jnp.tril(jnp.ones((1, 1, T, T), dtype=bool))
        out = nn.dot_product_attention(
            q, k, v, 
            mask = mask,
            deterministic=not train,
            dropout_rate = self.dropout if train else 0.0,
            dropout_rng = self.make_rng("dropout"),
        ) # (B, T, nh, hs) same layout as input

        # out = jax.nn.dot_product_attention(
        #     q, k, v, 
        #     is_causal=True,
        #     implementation='cudnn'
        # )
        out = out.reshape(B, T, C) # (B, T, C)

        out = nn.Dense(
            features=self.n_embed, 
            use_bias=False, 
            dtype=COMPUTE_DTYPE, 
            param_dtype=PARAM_DTYPE, 
            kernel_init=nn.initializers.normal(stddev=0.02))(out)

        out = nn.Dropout(rate=self.dropout, deterministic=not train)(out)

        return out

class Block(nn.Module):
    n_embed: int
    num_heads: int
    dropout: float

    @nn.compact
    def __call__(self, x, train: bool = True):
        norm_x = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x)
        attn_out = MultiHeadAttention(
            n_embed=self.n_embed,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )(norm_x, train=train)
        x = x + attn_out

        norm_x = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x)
        ffn_out = FeedForward(n_embed=self.n_embed, dropout=self.dropout)(norm_x, train=train)
        x = x + ffn_out
        return x

class GPT(nn.Module):
    vocab_size: int
    config: GPTConfig

    @nn.compact
    def __call__(self, x, train:bool=True): 
        B, T = x.shape

        # Token Embedding
        token_embed_table = nn.Embed(
            self.vocab_size, 
            self.config.n_embed, 
            dtype=COMPUTE_DTYPE, 
            param_dtype=PARAM_DTYPE, 
            embedding_init = nn.initializers.normal(stddev=0.02))

        token_embed = token_embed_table(x)

        # Positional Embedding
        pos = jnp.arange(T)
        pos_embed = nn.Embed(
            self.config.block_size, 
            self.config.n_embed, 
            dtype=COMPUTE_DTYPE, 
            param_dtype=PARAM_DTYPE, 
            embedding_init = nn.initializers.normal(stddev=0.02))(pos)
        
        x = token_embed + pos_embed

        # Transformer Blocks
        # With this loop JAX assigns unique name for each block like Block_0, Block_1 etc.
        for i in range(self.config.num_layers):
            x = Block(
                n_embed=self.config.n_embed,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )(x, train=train)

        x = nn.LayerNorm(dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x)
        # logits = nn.Dense(self.vocab_size, use_bias=False, dtype=COMPUTE_DTYPE, param_dtype=PARAM_DTYPE)(x)
        # Weights tying using the same token_embedd_table (transpose) for the output projections
        # .embedding holds the (vocab_size, n_embed) matrix
        # Note: x is in bfloat16, embedding is in float32, result will be float32 (good for numerical stability)
        logits = x @ token_embed_table.embedding.T # (B, T, vocab_size)
        return logits


# DataLoading
def get_batch(key, split, train_data, val_data, config):
    src = train_data if split == "train" else val_data
    ix = jrandom.randint(
        key,
        shape=(config.batch_size,),
        minval=0,
        maxval=len(src) - config.block_size - 1,
    )
    # Use jnp.arange once and reuse
    offs = jnp.arange(config.block_size)
    x = src[ix[:, None] + offs]
    y = src[ix[:, None] + offs + 1]
    return x, y


# Training and Eval Utils

# jax.jit traces a function and compiles it to XLA. It can only trace JAX arrays. 
# Python objects (model and optimizer state) are not traced.
# jax arrays are the params argument that are traced.
# So we need to wrap the actual train_step (can be jit-compiled) with make_train_step
# jax.value_and_grad computes both loss_fn(params) AND d(loss_fn)/d(params) in one fwd+bwd pass.
# This gives loss and gradients in one function call.
# .update() is an optax function that takes the gradients, optimizer state and params and returns the updates and new optimizer state.
# But it does modify the params directly. So we need to use .apply_updates() that takes in the params and updates and return the updated params

def make_train_step(model, tx):
    @jax.jit
    def train_step(params, opt_state, x,y, dropout_key):
        def loss_fn(params):
            logits = model.apply(
                params, x, train=True, rngs={"dropout": dropout_key}
            )  # Supply the PRNG for all dropout operations

            logits = logits.astype(jnp.float32) # Convert back to float32 for loss computation
            # we need to reshape the logits and y similar to pytorch
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1)
            ).mean()
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return loss, params, opt_state

    return train_step

def make_eval_step(model):
    @jax.jit
    def eval_step(params, x, y, dropout_key):
        # train = false, deterministic = true, no dropout & rngs needed
        logits = model.apply(params, x, train=False, rngs={"dropout": dropout_key})
        logits = logits.astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1)
        ).mean()
        return loss
    return eval_step

def estimate_loss(eval_step_fn, params, key, train_data, val_data, config, training_config):
    out={}
    for split_name in ['train', 'val']:
        losses = []
        for _ in range(training_config.eval_epochs):
            key, batch_key, dropout_key = jrandom.split(key, 3)
            x, y = get_batch(batch_key, split_name, train_data, val_data, config)
            loss = eval_step_fn(params, x, y, dropout_key)
            losses.append(loss)
        losses = jnp.stack(losses)
        out[split_name] = losses.mean().item()
    return out, key # return the new key to continue the PRNG sequence

# Generation
# We cannot use jit-compilation since the kv cache keeps growing
# Each new step would trigger a new compilation
# Also needs a fixed block size attention window and slide that window as we generate
# jrandom.categorical(key, logits) is the jax equivalent of pytorch.multinomial(probs, 1)
# It takes logits directly; no need for softmax or sampling. It does that internally.

def generate(model, params, key, idx, max_new_tokens, config, temperature=1.0):
    B = idx.shape[0]

    for _ in range(max_new_tokens):
        # Crop or pad to block size
        T = idx.shape[1]

        if T < config.block_size:
            pad_len = config.block_size - T
            idx_cond = jnp.pad(idx, ((0, 0), (pad_len, 0)))
        else:
            idx_cond = idx[:, -config.block_size:]

        key, dropout_key = jrandom.split(key)
        logits = model.apply(params, idx_cond, train=False, rngs={"dropout": dropout_key})
        logits = logits.astype(jnp.float32)
        logits = logits[:, -1, :] / temperature

        key, subkey = jrandom.split(key)
        next_idx = jrandom.categorical(subkey, logits)[:, None]
        idx = jnp.concatenate([idx, next_idx], axis=1)

    return idx

# Training Loop

# In pytorch we used model.parameters() to get the parameters of the model.
# In Jax we need to use model.init(key, dummy_x) to construct the pytree
# Pytree is a nested dict of all the model layer information (params, buffers, rngs)
# We need to intialize it with dummy data because jax calculates the params dynamically (tied to the concept of pure functions)
# you can do pytree traversal like this: for p in jax.tree_util.tree_leaves(params)

def train():
    training_config = TrainingConfig()
    config = GPTConfig()
    key = jrandom.PRNGKey(training_config.seed)

    wandb.init(
        project="gpt-run",
        name=f"jax_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            'batch_size': config.batch_size,    
            'block_size': config.block_size,
            'epochs': training_config.epochs,
            'eval_interval': training_config.eval_interval,
            'max_lr': training_config.max_lr,
            'min_lr': training_config.min_lr,
            'warmup_epochs': training_config.warmup_epochs,
            'n_embed': config.n_embed,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,    
            'dropout': config.dropout,
        }
    )

    # Data loading
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {c:i for i, c in enumerate(chars)}
    itos = {i:c for i, c in enumerate(chars)}
    def encode(s): return jnp.array([stoi[c] for c in s], dtype=jnp.int32)
    def decode(t): return "".join([itos[i] for i in t])

    data = encode(text)
    split = int(0.9*len(data))
    train_data = data[:split]
    val_data = data[split:]
    print(f"Total Training tokens: {len(train_data) // 1e6:.2f}M")
    print(f"Total Validation tokens: {len(val_data) // 1e3:.2f}K")

    # Checkpointer setup
    ckpt_dir = str(Path("checkpoints/gpt-jax").resolve())
    os.makedirs(ckpt_dir, exist_ok=True)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        create=True,
        enable_async_checkpointing=True,
    )

    ckpt_manager = ocp.CheckpointManager(
        ckpt_dir,
        item_handlers={
            "params": ocp.PyTreeCheckpointHandler(),
            "opt_state": ocp.PyTreeCheckpointHandler(),
            "step": ocp.JsonCheckpointHandler(),
        },
        options=options,
    )

    # Model init
    model = GPT(vocab_size=vocab_size, config=config)
    key, init_key = jrandom.split(key)
    dummy_x = jnp.ones((1, config.block_size), dtype=jnp.int32)
    params = model.init(init_key, dummy_x, train=False)
    print("Layer names: ", params['params'].keys())
    
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Number of parameters: {num_params/1e6:.4f}M")

    # Optimizer and Scheduler
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=training_config.max_lr,
        warmup_steps=training_config.warmup_epochs,
        decay_steps=training_config.lr_decay_epochs,
        end_value=training_config.min_lr,
    )

    # Chain optimizer actions together
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=scheduler,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=0.01,
        ) # From gpt-3 paper
    )

    # initialize the optimizer state
    opt_state = tx.init(params)

    # Restore if checkpoint exists
    start_step = 0

    # if ckpt_manager.latest_step() is not None:
    #     restore_step = ckpt_manager.latest_step()

    #     restored = ckpt_manager.restore(restore_step)

    #     params = restored["params"]
    #     opt_state = restored["opt_state"]
    #     start_step = restored["step"]

    #     print("Restored checkpoint:", restore_step)

    # Compile the training and evaluation steps
    train_step_fn = make_train_step(model, tx)
    eval_step_fn = make_eval_step(model)

    # Train
    start_time = time.time()
    for step in tqdm.trange(start_step, training_config.epochs, desc="Training"):
    #for step in range(start_step, training_config.epochs):
        t0 = time.time()
        
        # Periodic evaluation
        if step % training_config.eval_interval == 0:
            losses, key = estimate_loss(
                eval_step_fn, params, key,
                train_data, val_data, config, training_config,
            )
            wandb.log(losses | {"step": step})
            # print(f"\nStep {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

            # # Save checkpoint
            # ckpt_manager.save(
            #     step,
            #     items={
            #         "params": params,
            #         "opt_state": opt_state,
            #         "step": step,
            #     },
            # )

        # Need to split the key into 3
        key, dropout_key, batch_key = jrandom.split(key, 3)
        x, y = get_batch(batch_key, "train", train_data, val_data, config)

        loss, params, opt_state = train_step_fn(params, opt_state, x, y, dropout_key)
        # loss.block_until_ready() # async requests sync
        
        t1 = time.time()
        dt_ms = (t1 - t0) * 1000
        tokens_per_sec = config.batch_size * config.block_size / (t1 - t0)
        wandb.log({"time_per_step_in_ms": dt_ms, "tokens_per_second": tokens_per_sec})
        # print(f"step {step} | loss {loss:.4f} | dt {dt_ms:.2f}ms | tok/sec {tokens_per_sec:.2f}")

    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.2f}s")
    
    # Final Eval
    losses, key = estimate_loss(
        eval_step_fn, params, key,
        train_data, val_data, config, training_config,
    )
    print(f"Final — train: {losses['train']:.4f}, val: {losses['val']:.4f}")

    # Generate
    key, gen_key = jrandom.split(key)
    context = jnp.zeros((1, 1), dtype=jnp.int32)
    generated = generate(model, params, gen_key, context, 500, config)
    print(decode(generated[0].tolist()))

    # Save the final checkpoint
    ckpt_manager.save(
        training_config.epochs,
        {
            "params": params,
            "opt_state": opt_state,
            "step": training_config.epochs,
        }
    )
    print(f"Final checkpoint saved to {ckpt_dir}")
    ckpt_manager.wait_until_finished()

if __name__ == "__main__":
    train()