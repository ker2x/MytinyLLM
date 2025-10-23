"""
004.py — Tiny arithmetic LLM (single-file) using PyTorch

What it does
- Trains a very small character-level GPT-style model that learns to compute
  addition, subtraction, multiplication, and division from synthetic data.
- Division samples are restricted to exact integer division (no remainders, no divide-by-zero).
- All data is generated on-the-fly; no external datasets required.
- Single-file implementation as requested.

Quick start
- Train (CPU by default):
    python 004.py train --steps 1000 --batch-size 256 --max-digits 3

- Train on GPU if available:
    python 004.py train --device cuda --steps 5000 --batch-size 512 --max-digits 4

- Demo generation (after training):
    python 004.py demo --prompt "12+3="
    python 004.py demo --prompt "144/12="

- Save path (by default): math_llm_model.pt in the project root.

Notes
- The model is intentionally tiny to be trainable at home. For better accuracy,
  increase steps, batch size, and model depth/width parameters.
- Sequences look like: "a op b = result\n" and generation continues after the '='.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Tokenizer (character-level)
# -----------------------------

class CharTokenizer:
    """Simple character-level tokenizer with PAD id 0.

    Vocabulary chars: digits, + - * / = space and newline.
    We append a terminal newline to every sample, so generation knows when to stop.
    """

    def __init__(self):
        self.pad_token = '<PAD>'
        self.pad_id = 0
        chars = list("0123456789+-*/= \n")
        # Reserve 0 for PAD, others start from 1
        self.itos = [self.pad_token] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids if i != self.pad_id)

# -----------------------------
# Synthetic data generation
# -----------------------------

@dataclass
class GenConfig:
    max_digits: int = 3
    ops: str = "+-*/"


def _rand_int(max_digits: int) -> int:
    lo = 0
    hi = 10 ** max_digits - 1
    return random.randint(lo, hi)


def _make_div_pair(max_digits: int) -> Tuple[int, int]:
    # Ensure integer division with no remainder and no division by zero.
    b = 0
    while b == 0:
        b = _rand_int(max_digits)
    # sample quotient and compute a = b * q
    q = _rand_int(max_digits)
    a = b * q
    return a, b


def _make_sub_pair(max_digits: int) -> Tuple[int, int]:
    # To avoid negative results too often, swap if needed.
    a = _rand_int(max_digits)
    b = _rand_int(max_digits)
    if a < b:
        a, b = b, a
    return a, b


def generate_sample(cfg: GenConfig) -> str:
    """Generate one synthetic arithmetic sample as text.

    The returned string always ends with a newline and follows the format:
    "a<op>b=result\n" — e.g., "12+3=15\n". Division is guaranteed to have
    an exact integer result and no divide-by-zero.
    """
    op = random.choice(cfg.ops)
    if op == '+':
        a, b = _rand_int(cfg.max_digits), _rand_int(cfg.max_digits)
        res = a + b
    elif op == '-':
        a, b = _make_sub_pair(cfg.max_digits)
        res = a - b
    elif op == '*':
        a, b = _rand_int(cfg.max_digits), _rand_int(cfg.max_digits)
        res = a * b
    elif op == '/':
        a, b = _make_div_pair(cfg.max_digits)
        # exact integer division by construction (_make_div_pair ensures b != 0)
        res = a // b
    else:
        raise ValueError(f"Unknown operator: {op}")

    text = f"{a}{op}{b}={res}\n"
    return text


def make_batch(tokenizer: CharTokenizer, batch_size: int, cfg: GenConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a training batch of token ids (inputs and next-token targets).

    Each sample is generated on-the-fly. Sequences are padded to the maximum
    length in the batch. Inputs are all tokens except the last; targets are all
    tokens except the first (a standard next-token prediction setup).
    Returns tensors of shape (B, T) on the specified device.
    """
    samples = [generate_sample(cfg) for _ in range(batch_size)]
    encoded = [tokenizer.encode(s) for s in samples]
    # build input and target by shifting
    max_len = max(len(x) for x in encoded)
    # pad
    pad_id = tokenizer.pad_id
    x = torch.full((batch_size, max_len - 1), pad_id, dtype=torch.long)
    y = torch.full((batch_size, max_len - 1), pad_id, dtype=torch.long)
    for i, seq in enumerate(encoded):
        seq = torch.tensor(seq, dtype=torch.long)
        # Input: all but last token; Target: all but first token
        inp = seq[:-1]
        tgt = seq[1:]
        x[i, : inp.size(0)] = inp
        y[i, : tgt.size(0)] = tgt
    return x.to(device), y.to(device)

# -----------------------------
# Model: Tiny GPT-style decoder-only Transformer
# -----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, dropout: float):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.n_heads
        head_dim = C // H

        k = self.key(x).view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, d)
        q = self.query(x).view(B, T, H, head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, H, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, H, T, T)
        # causal mask: disallow attending to future tokens
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float('-inf'))
        # Use dim=-1 and handle potential NaN from all -inf rows
        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        att = self.attn_drop(att)
        y = att @ v  # (B, H, T, d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, dropout: float, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_mult * n_embd),
            nn.GELU(),
            nn.Linear(mlp_mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 128, n_layer: int = 2, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(512, n_embd)  # supports up to 512 tokens
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# -----------------------------
# Training and generation utilities
# -----------------------------

def compute_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Cross-entropy loss for next-token prediction with padding ignored.

    Args:
        logits: Tensor of shape (B, T, V) from the model.
        targets: Tensor of shape (B, T) with token ids, where PAD positions are to be ignored.
        pad_id: The token id used for padding.
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), ignore_index=pad_id)
    return loss

@torch.no_grad()
def generate(model: TinyGPT, tokenizer: CharTokenizer, prompt: str, max_new_tokens: int = 32, device: torch.device | None = None) -> str:
    """Greedy decoding from the model given a string prompt.

    Continues generating characters until a newline is produced or
    max_new_tokens is reached. Returns the decoded text containing the
    original prompt followed by the model's completion.
    """
    device = device or next(model.parameters()).device
    model.eval()
    # ensure prompt ends exactly at '=' or includes any context
    text = prompt
    if not text.endswith('=') and not text.endswith('= '):
        # allow raw prompts like "12+3=" or full sample; if full sample given, we'll continue anyway
        pass
    ids = tokenizer.encode(text)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        if x.size(1) >= 512:  # keep within positional embedding range (0-511)
            break
        logits = model(x)
        next_logits = logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # greedy
        x = torch.cat([x, next_id], dim=1)
        ch = tokenizer.itos[next_id.item()]
        if ch == '\n':
            break
    out = tokenizer.decode(x[0].tolist())
    return out

# -----------------------------
# Device selection helper
# -----------------------------

def pick_device(device: str) -> torch.device:
    """Pick a torch.device according to preference order and availability.

    Rules:
    - device == 'auto': prefer CUDA, then Metal (MPS), else CPU.
    - device == 'cuda' or 'mps': honor the request if available, else warn and fall back to CPU.
    - any other string is passed to torch.device, but if it fails at runtime it will raise.
    """
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        # Some builds may lack torch.backends.mps
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    # explicit requests
    if device == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print("Requested device 'cuda' is not available. Falling back to CPU.")
        return torch.device('cpu')
    if device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        print("Requested device 'mps' is not available. Falling back to CPU.")
        return torch.device('cpu')
    # passthrough
    try:
        return torch.device(device)
    except Exception:
        print(f"Requested device '{device}' is invalid or unavailable. Falling back to CPU.")
        return torch.device('cpu')

# -----------------------------
# Startup info helper
# -----------------------------

def print_startup_info(cmd: str, requested_device: str, picked_device: torch.device, checkpoint: str | None = None):
    """Print concise startup information: datetime, versions, device, checkpoint.

    Args:
        cmd: 'train' or 'demo'.
        requested_device: CLI device string user requested.
        picked_device: torch.device actually used.
        checkpoint: Path of checkpoint loaded (for demo) or target save path (for train).
    """
    import sys as _sys
    from datetime import datetime as _dt

    now = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
    pyver = _sys.version.split()[0]
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    dev_desc = picked_device.type
    extra = ''
    if dev_desc == 'cuda' and cuda_avail:
        try:
            extra = f" ({torch.cuda.get_device_name(0)})"
        except Exception:
            extra = ''
    print("=== Startup Info ===")
    print(f"time:       {now}")
    print(f"command:    {cmd}")
    print(f"python:     {pyver}")
    print(f"torch:      {torch_ver}")
    print(f"requested:  {requested_device}")
    print(f"device:     {dev_desc}{extra}")
    print(f"cuda:       {'yes' if cuda_avail else 'no'} | mps: {'yes' if mps_avail else 'no'}")
    if checkpoint is not None:
        print(f"checkpoint: {checkpoint}")
    print("====================")

# -----------------------------
# Orchestration (train loop, CLI)
# -----------------------------

def train(
    steps: int = 1000,
    batch_size: int = 256,
    max_digits: int = 3,
    lr: float = 3e-4,
    n_embd: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    dropout: float = 0.1,
    device: str = 'auto',
    ckpt_path: str = 'math_llm_model.pt',
    log_every: int = 100,
    eval_samples: int = 10000,
):
    """Train the model on synthetic arithmetic data and optionally evaluate.

    Prints running losses during training and, at the end, runs a small test on
    freshly generated problems to summarize accuracy per operator and overall.
    """
    #random.seed(1337)
    #torch.manual_seed(1337)

    dev = pick_device(device)
    tokenizer = CharTokenizer()

    # Try to load existing checkpoint, otherwise create new model
    if os.path.exists(ckpt_path):
        print(f"Loading existing checkpoint from {ckpt_path}")
        try:
            model, tokenizer, dev = load_model(ckpt_path, device=device)
            # Override device from load_model if needed
            dev = pick_device(device)
            model = model.to(dev)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Creating new model instead.")
            model = TinyGPT(tokenizer.vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=dropout).to(dev)
    else:
        print(f"No checkpoint found at {ckpt_path}, creating new model.")
        model = TinyGPT(tokenizer.vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=dropout).to(dev)

    # Startup information for training
    print_startup_info(cmd='train', requested_device=device, picked_device=dev, checkpoint=ckpt_path)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    cfg = GenConfig(max_digits=max_digits)

    t0 = time.time()
    ema_loss = None
    for step in range(1, steps + 1):
        model.train()
        x, y = make_batch(tokenizer, batch_size, cfg, dev)
        logits = model(x)
        loss = compute_loss(logits, y, tokenizer.pad_id)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        loss_val = loss.item()
        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * loss_val

        if step % log_every == 0 or step == 1 or step == steps:
            dt = time.time() - t0
            print(f"step {step:5d}/{steps} | loss {loss_val:.4f} | ema {ema_loss:.4f} | {(step/dt):.2f} it/s")
            # quick qualitative check
            with torch.no_grad():
                prompt = generate_sample_prompt(cfg)
                out = generate(model, tokenizer, prompt, max_new_tokens=24, device=dev)
                print(f"  prompt: {prompt!r}\n  model:  {out!r}")

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'dropout': dropout,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # End-of-training evaluation on synthetic test set
    if eval_samples and eval_samples > 0:
        print(f"Running evaluation on {eval_samples} samples...")
        eval_cfg = GenConfig(max_digits=max_digits, ops=cfg.ops)
        stats = evaluate(model, tokenizer, eval_cfg, n_samples=eval_samples, device=dev)
        overall = stats['overall']
        overall_acc = 100.0 * overall['correct'] / max(1, overall['total'])
        print("Evaluation summary:")
        for op in cfg.ops:
            c = stats['per_op'][op]
            acc = 100.0 * c['correct'] / max(1, c['total'])
            print(f"  {op}: {c['correct']}/{c['total']} = {acc:.2f}%")
        print(f"  Overall: {overall['correct']}/{overall['total']} = {overall_acc:.2f}%")


def generate_sample_prompt(cfg: GenConfig) -> str:
    # Create a prompt without the result, e.g., "12+3="
    op = random.choice(cfg.ops)
    if op == '+':
        a, b = _rand_int(cfg.max_digits), _rand_int(cfg.max_digits)
    elif op == '-':
        a, b = _make_sub_pair(cfg.max_digits)
    elif op == '*':
        a, b = _rand_int(cfg.max_digits), _rand_int(cfg.max_digits)
    elif op == '/':
        a, b = _make_div_pair(cfg.max_digits)
    else:
        a, b = 1, 1
    return f"{a}{op}{b}="


def load_model(ckpt_path: str, device: str = 'cpu') -> Tuple[TinyGPT, CharTokenizer, torch.device]:
    """Load a saved checkpoint and return (model, tokenizer, device)."""
    dev = pick_device(device)
    tokenizer = CharTokenizer()
    ckpt = torch.load(ckpt_path, map_location=dev)
    model = TinyGPT(
        vocab_size=ckpt.get('vocab_size', tokenizer.vocab_size),
        n_embd=ckpt.get('n_embd', 128),
        n_layer=ckpt.get('n_layer', 2),
        n_head=ckpt.get('n_head', 4),
        dropout=ckpt.get('dropout', 0.1),
    ).to(dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, tokenizer, dev


# -----------------------------
# Evaluation utilities
# -----------------------------

def _sample_operands_for_op(op: str, max_digits: int) -> Tuple[int, int]:
    if op == '+':
        return _rand_int(max_digits), _rand_int(max_digits)
    if op == '-':
        return _make_sub_pair(max_digits)
    if op == '*':
        return _rand_int(max_digits), _rand_int(max_digits)
    if op == '/':
        return _make_div_pair(max_digits)
    raise ValueError(f"Unknown operator: {op}")


def _ground_truth(a: int, b: int, op: str) -> int:
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        return a // b if b != 0 else 0
    raise ValueError(f"Unknown operator: {op}")


def _extract_result_from_text(text: str) -> str:
    """Extract the model-predicted result substring after '=' up to newline.

    Returns the stripped string of digits (and optional leading '-') until the
    first newline. If '=' is not found, returns empty string.
    """
    if '=' not in text:
        return ''
    tail = text.split('=', 1)[1]
    # read until newline
    if '\n' in tail:
        tail = tail.split('\n', 1)[0]
    # strip spaces; keep digits and leading '-'
    tail = tail.strip()
    # allow only optional leading '-' followed by digits
    if tail.startswith('-'):
        sign = '-'
        tail = tail[1:]
    else:
        sign = ''
    digits = ''.join(ch for ch in tail if ch.isdigit())
    return (sign + digits) if digits != '' else ''


@torch.no_grad()
def evaluate(model: TinyGPT, tokenizer: CharTokenizer, cfg: GenConfig, n_samples: int, device: torch.device) -> dict:
    """Run a synthetic test set and compute per-op and overall accuracy.

    Returns a dict with keys: 'per_op' mapping op->(correct,total), and 'overall'.
    """
    counts = {op: {'correct': 0, 'total': 0} for op in cfg.ops}
    for _ in range(n_samples):
        op = random.choice(cfg.ops)
        a, b = _sample_operands_for_op(op, cfg.max_digits)
        expected = str(_ground_truth(a, b, op))
        prompt = f"{a}{op}{b}="
        out = generate(model, tokenizer, prompt, max_new_tokens=24, device=device)
        pred = _extract_result_from_text(out)
        counts[op]['total'] += 1
        if pred == expected:
            counts[op]['correct'] += 1
    total_correct = sum(v['correct'] for v in counts.values())
    total = sum(v['total'] for v in counts.values())
    return {
        'per_op': counts,
        'overall': {'correct': total_correct, 'total': total},
    }


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Tiny arithmetic LLM (char-level GPT) in a single file")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_train = sub.add_parser('train', help='Train the tiny arithmetic LLM on synthetic data')
    p_train.add_argument('--steps', type=int, default=5000)
    p_train.add_argument('--batch-size', type=int, default=256)
    p_train.add_argument('--max-digits', type=int, default=3)
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--n-embd', type=int, default=128)
    p_train.add_argument('--n-layer', type=int, default=2)
    p_train.add_argument('--n-head', type=int, default=4)
    p_train.add_argument('--dropout', type=float, default=0.1)
    p_train.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'mps', 'auto'])
    p_train.add_argument('--ckpt', type=str, default='math_llm_model.pt')
    p_train.add_argument('--log-every', type=int, default=100)
    p_train.add_argument('--eval-samples', type=int, default=10000, help='Number of synthetic test samples to evaluate at end of training (0 to skip)')

    p_demo = sub.add_parser('demo', help='Run generation on a prompt like "12+3=" (loads checkpoint if provided)')
    p_demo.add_argument('--prompt', type=str, required=True, help='Prompt such as "12+3=" or "144/12="')
    p_demo.add_argument('--ckpt', type=str, default='math_llm_model.pt')
    p_demo.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'mps', 'auto'])
    p_demo.add_argument('--max-new-tokens', type=int, default=24)

    args = parser.parse_args()

    if args.cmd == 'train':
        train(
            steps=args.steps,
            batch_size=args.batch_size,
            max_digits=args.max_digits,
            lr=args.lr,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            dropout=args.dropout,
            device=args.device,
            ckpt_path=args.ckpt,
            log_every=args.log_every,
            eval_samples=args.eval_samples,
        )
    elif args.cmd == 'demo':
        DEFAULT_CKPT = 'math_llm_model.pt'
        ckpt_used = None
        try:
            model, tokenizer, dev = load_model(args.ckpt, device=args.device)
            ckpt_used = args.ckpt
        except FileNotFoundError:
            # If a custom checkpoint was requested but missing, try falling back to default if it exists.
            if args.ckpt != DEFAULT_CKPT and os.path.exists(DEFAULT_CKPT):
                print(f"Checkpoint '{args.ckpt}' not found. Falling back to default '{DEFAULT_CKPT}'.")
                try:
                    model, tokenizer, dev = load_model(DEFAULT_CKPT, device=args.device)
                    ckpt_used = DEFAULT_CKPT
                except FileNotFoundError:
                    # Shouldn't happen because we checked exists, but guard anyway.
                    print(f"Default checkpoint '{DEFAULT_CKPT}' not found despite existence check. Using randomly initialized model (results will be poor).")
                    dev = pick_device(args.device)
                    tokenizer = CharTokenizer()
                    model = TinyGPT(tokenizer.vocab_size).to(dev)
                    ckpt_used = '(random init)'
            else:
                print(f"Checkpoint '{args.ckpt}' not found. Using randomly initialized model (results will be poor).")
                dev = pick_device(args.device)
                tokenizer = CharTokenizer()
                model = TinyGPT(tokenizer.vocab_size).to(dev)
                ckpt_used = '(random init)'
        print_startup_info(cmd='demo', requested_device=args.device, picked_device=dev, checkpoint=ckpt_used or '(unknown)')
        out = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, device=dev)
        print(out)


if __name__ == '__main__':
    main()
