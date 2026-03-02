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

# Hyperparams
batch_size = 64
block_size = 256
epochs = 500
eval_interval=1000
#lr = 3e-4
max_lr = 3e-4
min_lr = 3e-5 
warmup_epochs = int(0.05 * epochs)   # 5% warmup
lr_decay_epochs = epochs


n_embed = 50304
eval_epochs = 200
num_heads = 6
num_layers = 6
dropout = 0.2

@dataclass
class ModelConfig:
    batch_size: int = 24
    seq_length: int = 256
    vocab_size: int = 50304
    n_embed: int = 384
    n_head: int = 8
    n_layer: int = 6
    n_kv_heads: int = 4 # For Grouped Query Attention
    sliding_window: int = 2048

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


with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

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
    def __init__(self, text, split='trian', block_size=128, train_ratio=0.9):
        self.text = text

        enc = tiktoken.get_encoding("gpt-2")
        self.tokens = torch.tensor(enc.encode(text), dtype=torch.long)

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
train_ds = TinyShakespeareDataset(split='train', block_size=block_size)
val_ds = TinyShakespeareDataset(split='val', block_size=block_size)

# Initialize DataLoaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    




