
"""
006_scratchpad.py — Tiny arithmetic LLM with Scratchpad Training

What it does
- Implements the "scratchpad" / "chain-of-thought" idea.
- The model is no longer trained on "a+b=result\n".
- It is trained on "a+b=[...steps...;F:result]\n".
- This teaches the model the *process* of arithmetic, not just input-output pairs.
- It uses Curriculum Learning: starts with 1-digit problems and gradually
  increases difficulty to 2-digit and then 3-digit+ problems.

Why this works
- Multiplication is an algorithm. A standard transformer cannot easily
  run this serial, multi-step algorithm internally.
- By putting the steps *into the target text*, we change the problem
  from "compute the answer" to "generate the text of the computation".
- Transformers are excellent at the second task.
- This new file is based on 005.py.
"""
from __future__ import annotations

import argparse
import math
import random
import re
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

    Vocabulary:
    - 0-9: digits
    - +-*/=: ops
    - \n: newline
    - []: scratchpad start/end
    - ;: step separator
    - :FAPCWKBMR: scratchpad format markers:
        F = Final answer (e.g., F:132)
        A = Sum of partial products in multiplication (e.g., A:26+130=156)
        P = Partial product in multiplication (e.g., P1:13*2)
        C = Column number in addition/subtraction (e.g., C1:5+7=12)
        W = Write digit to result (e.g., W:2)
        K = Carry value in addition (e.g., K:1)
        B = Borrow operation in subtraction (e.g., B:5->4)
        M = Single-digit multiplication step (e.g., M1:3*2=6)
        R = Result of partial product (e.g., R1:26)
    - <PAD>: padding
    """

    def __init__(self):
        self.pad_token = '<PAD>'
        self.pad_id = 0
        # NEW: Added M and R for detailed multiplication scratchpad
        chars = list("0123456789+-*/= \n[];:FAPCWKBMR>")
        # Reserve 0 for PAD, others start from 1
        self.itos = [self.pad_token] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids if i != self.pad_id)


# -----------------------------
# Synthetic data generation (NEW: Scratchpad)
# -----------------------------

@dataclass
class GenConfig:
    max_digits: int = 3
    ops: str = "+-*/"
    # Optional per-operator sampling probabilities aligned with `ops`.
    op_probs: List[float] | None = None


def _rand_int(max_digits: int) -> int:
    if max_digits == 0: return 0
    lo = 0
    hi = 10 ** max_digits - 1
    return random.randint(lo, hi)


def _make_div_pair(max_digits: int) -> Tuple[int, int]:
    # Ensure integer division with no remainder and non-trivial quotient.
    if max_digits == 0: return 0, 1  # Handle 0-digit case
    hi = 10 ** max_digits - 1
    b = random.randint(1, hi)
    # Choose quotient q to avoid trivial zeros and keep result length challenging.
    lo_q = 0 if max_digits == 1 else 10 ** (max_digits - 1)
    q = random.randint(lo_q, hi)
    a = b * q
    return a, b


def _make_sub_pair(max_digits: int) -> Tuple[int, int]:
    # Allow negative results ~40% of the time for variety
    a = _rand_int(max_digits)
    b = _rand_int(max_digits)
    # 60% chance: ensure non-negative by swapping if needed
    if random.random() > 0.4 and a < b:
        a, b = b, a
    return a, b


# --- NEW: Scratchpad Generator Functions ---

def _make_add_scratchpad(a: int, b: int, max_digits: int) -> str:
    """Generates a scratchpad for right-to-left addition with carry.
    Example: 85+47=[C1:5+7=12;W:2;K:1;C2:8+4+1=13;W:13;F:132]

    Scratchpad format:
    - C{n}: Column n (rightmost is C1)
    - W: Write this digit to result
    - K: Carry this value to next column
    - F: Final answer
    """
    a_s, b_s = str(a), str(b)
    res = a + b
    # Pad to max_digits + 1 (to handle overflow)
    max_len = max(len(a_s), len(b_s)) + 1
    a_s, b_s = a_s.zfill(max_len), b_s.zfill(max_len)

    steps = []
    carry = 0
    result_so_far = ""

    for i in range(max_len - 1, -1, -1):
        d1 = int(a_s[i])
        d2 = int(b_s[i])
        col_num = max_len - i
        s = d1 + d2 + carry

        step_str = f"C{col_num}:{d1}+{d2}"
        if carry > 0:
            step_str += f"+{carry}"
        step_str += f"={s}"
        steps.append(step_str)

        write_val = s % 10
        carry = s // 10

        steps.append(f"W:{write_val}")
        result_so_far = str(write_val) + result_so_far
        if carry > 0:
            steps.append(f"K:{carry}")

    # Clean up leading zero steps: keep only the least-significant columns needed
    final_steps = []
    final_res_str = str(res)
    keep_cols = max(1, len(final_res_str))  # number of columns to keep from the right (C1..Ckeep)
    col_num = 0
    for step in steps:
        if step.startswith("C"):
            col_num = int(re.search(r"C(\d+):", step).group(1))
            if col_num <= keep_cols:
                final_steps.append(step)
        elif step.startswith("W:") or step.startswith("K:"):
            if col_num <= keep_cols:
                final_steps.append(step)

    final_steps.append(f"F:{res}")
    return f"{a}+{b}=[{';'.join(final_steps)}]\n"


def _make_sub_scratchpad(a: int, b: int, max_digits: int) -> str:
    """Generates a scratchpad for right-to-left subtraction with borrow.
    Example: 52-27=[C1:2-7;B:5->4;C1:12-7=5;W:5;C2:4-2=2;W:2;F:25]

    Scratchpad format:
    - C{n}: Column n (rightmost is C1)
    - B: Borrow operation (e.g., B:5->4 means digit 5 becomes 4 after borrowing)
    - W: Write this digit to result
    - F: Final answer
    """
    res = a - b
    # Handle negative results simply
    if res < 0:
        return f"{a}-{b}=[F:{res}]\n"

    a_s, b_s = str(a), str(b)
    max_len = max(len(a_s), len(b_s))
    if max_digits > max_len:
        max_len = max_digits

    a_s, b_s = a_s.zfill(max_len), b_s.zfill(max_len)

    steps = []
    a_list = [int(d) for d in a_s]

    for i in range(max_len - 1, -1, -1):
        d1 = a_list[i]
        d2 = int(b_s[i])
        col_num = max_len - i

        step_str = f"C{col_num}:{d1}-{d2}"

        if d1 < d2:
            steps.append(step_str)
            # Find first non-zero digit to borrow from
            j = i - 1
            while j >= 0 and a_list[j] == 0:
                j -= 1

            if j < 0:  # Should not happen if a > b
                # This indicates a bug in _make_sub_pair or logic
                return f"{a}-{b}=[F:{res}]\n"  # Fallback

            # Borrow from a_list[j]
            steps.append(f"B:{a_list[j]}->{a_list[j] - 1}")
            a_list[j] -= 1
            # Propagate borrow (9s)
            for k in range(j + 1, i):
                # a_list[k] was 0
                steps.append(f"B:0->9")  # Show 0 becomes 9
                a_list[k] = 9

            d1 += 10
            # a_list[i] = d1 # This is implicit, d1 is just used for calc
            steps.append(f"C{col_num}:{d1}-{d2}={d1 - d2}")
            steps.append(f"W:{d1 - d2}")
        else:
            s = d1 - d2
            step_str += f"={s}"
            steps.append(step_str)
            steps.append(f"W:{s}")

    # Clean up leading zero steps: keep only the least-significant columns needed
    final_steps = []
    final_res_str = str(res)
    keep_cols = max(1, len(final_res_str))  # number of columns to keep from the right (C1..Ckeep)
    col_num = 0
    for step in steps:
        if step.startswith("C"):
            col_num = int(re.search(r"C(\d+):", step).group(1))
            if col_num <= keep_cols:
                final_steps.append(step)
        elif step.startswith("W:") or step.startswith("B:"):
            if col_num <= keep_cols:
                final_steps.append(step)

    final_steps.append(f"F:{res}")
    return f"{a}-{b}=[{';'.join(final_steps)}]\n"


def _make_mul_scratchpad(a: int, b: int) -> str:
    """Generates a scratchpad for multiplication via partial products with digit-by-digit breakdown.
    Example: 13*12=[P1:13*2;D1:3*2=6;W:6;D2:1*2=2;W:2;R1:26;P2:13*10;D1:3*1=3;W:3;D2:1*1=1;W:1;R2:130;A:26+130=156;F:156]

    Scratchpad format:
    - P{n}: Start partial product n (e.g., P1:13*2)
    - D{n}: Single digit multiplication step (e.g., D1:3*2=6)
    - W:{d}: Write digit to result
    - K:{c}: Carry value
    - R{n}: Result of partial product (e.g., R1:26)
    - A: Add all partial products together
    - F: Final answer
    """
    res = a * b
    b_s = str(b)
    steps = []
    partials = []

    if a == 0 or b == 0:
        steps.append(f"P1:{a}*{b}=0")
        steps.append(f"F:0")
        return f"{a}*{b}=[{';'.join(steps)}]\n"

    # Process each digit of b from right to left
    for b_idx, b_digit_ch in enumerate(reversed(b_s)):
        b_digit = int(b_digit_ch)
        if b_digit == 0:
            continue  # Skip zero digits

        multiplier = b_digit * (10 ** b_idx)

        # Start this partial product
        steps.append(f"P{b_idx + 1}:{a}*{multiplier}")

        # Multiply a by the single digit b_digit, digit by digit with carries
        a_s = str(a)
        carry = 0
        partial_result = ""

        for a_pos, a_digit_ch in enumerate(reversed(a_s), start=1):
            a_digit = int(a_digit_ch)
            product = a_digit * b_digit + carry
            digit_out = product % 10
            new_carry = product // 10

            # Show the single-digit multiplication step
            base_product = a_digit * b_digit
            if carry > 0:
                steps.append(f"M{a_pos}:{a_digit}*{b_digit}+{carry}={product}")
            else:
                steps.append(f"M{a_pos}:{a_digit}*{b_digit}={base_product}")

            steps.append(f"W:{digit_out}")
            if new_carry > 0 and a_pos < len(a_s):
                steps.append(f"K:{new_carry}")

            partial_result = str(digit_out) + partial_result
            carry = new_carry

        # Write any remaining carry
        if carry > 0:
            steps.append(f"W:{carry}")
            partial_result = str(carry) + partial_result

        # Adjust for place value (multiply by 10^b_idx)
        partial_value = int(partial_result) * (10 ** b_idx)
        steps.append(f"R{b_idx + 1}:{partial_value}")
        partials.append(partial_value)

    # Add all partial products
    if len(partials) == 0:
        partials.append(0)

    if len(partials) == 1:
        steps.append(f"A:{partials[0]}={partials[0]}")
    else:
        sum_str = "+".join(map(str, partials))
        steps.append(f"A:{sum_str}={sum(partials)}")

    steps.append(f"F:{res}")
    return f"{a}*{b}=[{';'.join(steps)}]\n"


def _make_div_scratchpad(a: int, b: int) -> str:
    """
    Long division scratchpad with per-digit steps (integer division).
    Markers used (kept within existing tokenizer letters):
    - C{n}: current chunk context; we record comparator and subtraction
            e.g., C1:1264>=538 and C1:1264-1076=188
    - P{n}: product of the chosen quotient digit and divisor
            e.g., P1:2*538=1076
    - W:{d}: write quotient digit d
    - F:{q}: final quotient

    Note: This representation assumes integer division (no remainder)
    as generated by the sampling code; intermediate remainders may be
    nonzero, but the final remainder is expected to be 0.
    """
    if b == 0:
        return f"{a}/{b}=[F:0]\n"
    dividend_str = str(a)
    divisor = b

    steps: List[str] = []
    q_digits: List[str] = []
    chunk = 0
    started = False
    n = len(dividend_str)
    step_num = 0  # Sequential step counter

    for i, ch in enumerate(dividend_str, start=1):
        # Bring down next digit and form the new chunk
        chunk = chunk * 10 + int(ch)
        # Choose quotient digit for this place
        qd = chunk // divisor if divisor != 0 else 0

        # Skip emitting product/subtraction/write for leading zero quotient digits
        if not started and qd == 0 and i < n:
            # carry chunk forward and continue
            continue

        # From here on, we are emitting quotient digits (including zeros)
        started = True
        step_num += 1  # Increment step counter
        # Comparator context for this step
        steps.append(f"C{step_num}:{chunk}>={divisor}")
        prod = qd * divisor
        rem = chunk - prod
        # Record product and subtraction
        steps.append(f"P{step_num}:{qd}*{divisor}={prod}")
        steps.append(f"C{step_num}:{chunk}-{prod}={rem}")
        # Write the quotient digit
        steps.append(f"W:{qd}")
        q_digits.append(str(qd))
        # Carry remainder to next step
        chunk = rem

    # If we never wrote a digit (e.g., a < b, typically only when a==0 in our data),
    # emit a single zero digit to make the format consistent
    if not q_digits:
        total = int(dividend_str)
        steps.append(f"C1:{total}>={divisor}")
        steps.append(f"P1:0*{divisor}=0")
        steps.append(f"C1:{total}-0={total}")
        steps.append("W:0")
        q_digits.append("0")

    # Final quotient via integer division for robustness
    res = a // b if b != 0 else 0
    return f"{a}/{b}=[{';'.join(steps)};F:{res}]\n"


# --- End new functions ---

def _choose_op(cfg: GenConfig) -> str:
    """Choose an operator, optionally using configured probabilities."""
    ops = list(cfg.ops)
    weights = None
    if cfg.op_probs is not None:
        if len(cfg.op_probs) == len(ops):
            weights = [max(0.0, float(w)) for w in cfg.op_probs]
            if sum(weights) <= 0:
                weights = None
    if weights is None:
        return random.choice(ops)
    return random.choices(ops, weights=weights, k=1)[0]


def generate_scratchpad_sample(cfg: GenConfig) -> str:
    """Generate one synthetic arithmetic sample as text WITH SCRATCHPAD.
    """
    op = _choose_op(cfg)
    if op == '+':
        a, b = _rand_int(cfg.max_digits), _rand_int(cfg.max_digits)
        text = _make_add_scratchpad(a, b, cfg.max_digits)
    elif op == '-':
        a, b = _make_sub_pair(cfg.max_digits)
        text = _make_sub_scratchpad(a, b, cfg.max_digits)
    elif op == '*':
        a, b = _rand_int(cfg.max_digits), _rand_int(cfg.max_digits)
        text = _make_mul_scratchpad(a, b)
    elif op == '/':
        a, b = _make_div_pair(cfg.max_digits)
        text = _make_div_scratchpad(a, b)
    else:
        raise ValueError(f"Unknown operator: {op}")

    return text


def make_batch(tokenizer: CharTokenizer, batch_size: int, cfg: GenConfig, device: torch.device, max_pos: int) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """Create a training batch of token ids (inputs and next-token targets).

    NEW: This function is simplified. It just calls `generate_scratchpad_sample`
    `batch_size` times. The old `generate_augmented_samples` is removed.

    NEW: Takes `max_pos` to warn about truncation.
    """
    samples: List[str] = []
    # Keep adding samples until we have at least batch_size samples
    while len(samples) < batch_size:
        samples.append(generate_scratchpad_sample(cfg))

    encoded = [tokenizer.encode(s) for s in samples]
    max_len_in_batch = max(len(x) for x in encoded)

    # CRITICAL: Check if scratchpad is longer than model's context
    if max_len_in_batch >= max_pos:
        # Truncate and warn
        print(f"Warning: Batch max scratchpad length {max_len_in_batch} >= model max_pos {max_pos}. Truncating.")
        max_len = max_pos - 1  # Leave room for target
        encoded = [x[:max_len] for x in encoded]
    else:
        max_len = max_len_in_batch

    # build input and target by shifting
    pad_id = tokenizer.pad_id
    # We need max_len for the full sequence (x + y)
    # x will be max_len - 1
    x = torch.full((batch_size, max_len - 1), pad_id, dtype=torch.long)
    y = torch.full((batch_size, max_len - 1), pad_id, dtype=torch.long)

    for i, seq in enumerate(encoded):
        if len(seq) < 2: continue  # Skip empty or single-char sequences
        seq_len = min(len(seq), max_len)  # Ensure we don't go over

        # Input: all but last token; Target: all but first token
        inp = seq[:seq_len - 1]
        tgt = seq[1:seq_len]

        x[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)

    return x.to(device), y.to(device)


# -----------------------------
# Model: Tiny GPT-style decoder-only Transformer
# (This section is UNCHANGED from 005.py, but max_pos is now an arg)
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
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float('-inf'))
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
    def __init__(self, vocab_size: int, n_embd: int = 128, n_layer: int = 2, n_head: int = 4, dropout: float = 0.1,
                 max_pos: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_pos, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.max_pos = max_pos  # store for saving/loading

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        if T >= self.max_pos:
            # This should not happen if make_batch truncates, but as a safeguard
            idx = idx[:, :self.max_pos]
            T = idx.size(1)

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        try:
            tok_embs = self.tok_emb(idx)
            pos_embs = self.pos_emb(pos)
        except IndexError as e:
            print(
                f"Error during embedding lookup. T={T}, max_pos={self.max_pos}, idx.shape={idx.shape}, pos.shape={pos.shape}")
            print(f"Max index in idx: {idx.max()}, Vocab size: {self.vocab_size}")
            raise e

        x = tok_embs + pos_embs
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


# -----------------------------
# Training and generation utilities
# (compute_loss is UNCHANGED)
# -----------------------------

def compute_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), ignore_index=pad_id)
    return loss


@torch.no_grad()
def generate(model: TinyGPT, tokenizer: CharTokenizer, prompt: str, max_new_tokens: int = 128,
             device: torch.device | None = None) -> str:
    """Greedy decoding from the model given a string prompt.

    NEW: Increased max_new_tokens default to 128 for scratchpad.
    """
    device = device or next(model.parameters()).device
    model.eval()
    text = prompt
    ids = tokenizer.encode(text)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    max_pos = model.max_pos

    for _ in range(max_new_tokens):
        # Truncate input sequence if it exceeds max_pos
        x_cond = x if x.size(1) <= max_pos else x[:, -max_pos:]

        if x_cond.size(1) == 0: break  # Should not happen

        logits = model(x_cond)
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
# (UNCHANGED from 005.py)
# -----------------------------

def pick_device(device: str) -> torch.device:
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
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
    try:
        return torch.device(device)
    except Exception:
        print(f"Requested device '{device}' is invalid or unavailable. Falling back to CPU.")
        return torch.device('cpu')


# -----------------------------
# Startup info helper
# (UNCHANGED from 005.py)
# -----------------------------

def print_startup_info(cmd: str, requested_device: str, picked_device: torch.device, checkpoint: str | None = None):
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


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _format_num(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def print_model_summary(model: nn.Module):
    name = model.__class__.__name__
    try:
        vocab_size = getattr(model, 'vocab_size', None)
        n_embd = getattr(model.tok_emb, 'embedding_dim', None)
        n_layer = len(getattr(model, 'blocks', []))
        n_head = None
        if n_layer > 0: n_head = model.blocks[0].attn.n_heads
        dropout = model.drop.p
        max_pos = getattr(model, 'max_pos', None)
    except Exception:
        vocab_size = n_embd = n_layer = n_head = dropout = max_pos = None

    total, trainable = _count_parameters(model)
    try:
        first_param = next(model.parameters())
        bytes_per_param = torch.finfo(first_param.dtype).bits // 8
    except Exception:
        bytes_per_param = 4
    approx_mb = (total * bytes_per_param) / (1024 * 1024)

    print("=== Model Summary ===")
    print(f"arch:       {name}")
    if vocab_size is not None: print(f"vocab_size: {vocab_size}")
    if n_embd is not None: print(f"n_embd:     {n_embd}")
    if n_layer is not None: print(f"n_layer:    {n_layer}")
    if n_head is not None: print(f"n_head:     {n_head}")
    if dropout is not None: print(f"dropout:    {dropout}")
    if max_pos is not None: print(f"max_pos:    {max_pos}")
    print(f"params:     {total} ({_format_num(total)}) | trainable: {trainable} ({_format_num(trainable)})")
    print(f"approx size: {approx_mb:.2f} MB (parameters)")
    print("====================")


# -----------------------------
# Orchestration (NEW: Curriculum Training)
# -----------------------------

def train(
        steps: int = 1000,
        batch_size: int = 256,
        max_digits: int = 3,  # This is now the *final* max_digits
        lr: float = 3e-4,
        n_embd: int = 128,
        n_layer: int = 2,
        n_head: int = 4,
        dropout: float = 0.1,
        device: str = 'auto',
        ckpt_path: str = 'math_llm_scratchpad_model.pt',  # New checkpoint path
        log_every: int = 100,
        eval_samples: int = 5000,
        ops: str = "+-*/",
        op_probs: List[float] | None = None,
        max_pos: int = 512,  # NEW: Make max_pos configurable
        use_amp: bool = False,  # NEW: Enable mixed precision training
):
    """
    Train the model using a CURRICULUM.
    We start with 1-digit problems, then 2-digit, then up to max_digits.
    This is essential for the model to learn the algorithmic patterns.
    """
    # random.seed(1337)
    # torch.manual_seed(1337)

    dev = pick_device(device)
    tokenizer = CharTokenizer()

    if os.path.exists(ckpt_path):
        print(f"Loading existing checkpoint from {ckpt_path}")
        try:
            model, tokenizer, dev = load_model(ckpt_path, device=device)
            dev = pick_device(device)  # Respect user's device choice
            model = model.to(dev)
            # Ensure loaded model's max_pos matches arg
            if model.max_pos != max_pos:
                print(
                    f"Warning: Model max_pos ({model.max_pos}) differs from arg ({max_pos}). Using loaded model's value.")
                max_pos = model.max_pos
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Creating new model instead.")
            model = TinyGPT(tokenizer.vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=dropout,
                            max_pos=max_pos).to(dev)
    else:
        print(f"No checkpoint found at {ckpt_path}, creating new model.")
        model = TinyGPT(tokenizer.vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=dropout,
                        max_pos=max_pos).to(dev)

    print_startup_info(cmd='train', requested_device=device, picked_device=dev, checkpoint=ckpt_path)
    print_model_summary(model)

    # NEW: Setup for mixed precision training
    if use_amp:
        # Use torch.amp for PyTorch 2.0+, fallback to torch.cuda.amp for older versions
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            print("Using torch.amp for mixed precision training (bfloat16/float16)")
            scaler = torch.amp.GradScaler(dev.type) if dev.type in ['cuda', 'xpu'] else None
            autocast_context = lambda: torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16 if dev.type == 'cpu' else torch.float16)
        elif hasattr(torch.cuda, 'amp'):
            print("Using torch.cuda.amp for mixed precision training (float16)")
            scaler = torch.cuda.amp.GradScaler() if dev.type == 'cuda' else None
            autocast_context = lambda: torch.cuda.amp.autocast()
        else:
            print("Warning: AMP requested but not available. Falling back to full precision.")
            use_amp = False
            scaler = None
            autocast_context = None
    else:
        scaler = None
        autocast_context = None

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- NEW: Curriculum Definition ---
    # Modified curriculum to prevent catastrophic forgetting:
    # Stage 1: Train on 1-digit problems only (20% of steps)
    # Stage 2: Train on a mix of 1 to max_digits problems (80% of steps)
    # This ensures the model maintains single-digit proficiency throughout training

    if max_digits <= 1:
        # Special case: if max_digits is 1 or less, just train on that
        curriculum_stages = [1 if max_digits >= 1 else 0]
        stage_steps = [steps]
        stage_is_mixed = [False]
    else:
        # Stage 1: 1-digit only for 20% of steps
        stage1_steps = steps // 5
        # Stage 2: mixed digits for remaining 80% of steps
        stage2_steps = steps - stage1_steps

        curriculum_stages = [1, max_digits]  # 1-digit, then mixed up to max_digits
        stage_steps = [stage1_steps, stage2_steps]
        stage_is_mixed = [False, True]  # Second stage samples from full range

    print("=== Curriculum Plan ===")
    current_step = 0
    for i, (stage_max_digits, stage_n_steps, is_mixed) in enumerate(zip(curriculum_stages, stage_steps, stage_is_mixed)):
        end_step = current_step + stage_n_steps
        if is_mixed:
            print(f"  Stage {i + 1}/{len(curriculum_stages)}: mixed (1 to {stage_max_digits} digits) (Steps {current_step + 1} - {end_step})")
        else:
            print(f"  Stage {i + 1}/{len(curriculum_stages)}: {stage_max_digits}-digit only (Steps {current_step + 1} - {end_step})")
        current_step = end_step
    print("=======================")

    t0 = time.time()
    ema_loss = None
    total_step_counter = 0
    _printed_log_legend = False

    for stage_max_digits, stage_n_steps, is_mixed in zip(curriculum_stages, stage_steps, stage_is_mixed):
        if is_mixed:
            print(f"--- Starting Stage: mixed (1 to {stage_max_digits} digits) for {stage_n_steps} steps ---")
        else:
            print(f"--- Starting Stage: {stage_max_digits}-digit only for {stage_n_steps} steps ---")

        for stage_step in range(1, stage_n_steps + 1):
            total_step_counter += 1

            model.train()

            # Sample digit count for this batch
            if is_mixed:
                # Randomly sample max_digits from 1 to stage_max_digits for each batch
                batch_max_digits = random.randint(1, stage_max_digits)
            else:
                batch_max_digits = stage_max_digits

            cfg = GenConfig(max_digits=batch_max_digits, ops=ops, op_probs=op_probs)
            x, y = make_batch(tokenizer, batch_size, cfg, dev, max_pos)

            if x.size(1) == 0:  # Skip empty batches
                print(f"Warning: Skipping empty batch at step {total_step_counter}. Check data generation and max_pos.")
                continue

            # NEW: Mixed precision training
            optim.zero_grad(set_to_none=True)

            if use_amp and autocast_context is not None:
                with autocast_context():
                    logits = model(x)
                    loss = compute_loss(logits, y, tokenizer.pad_id)

                if scaler is not None:
                    # Use gradient scaling for CUDA
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    # CPU or MPS without scaling
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            else:
                # Standard full precision training
                logits = model(x)
                loss = compute_loss(logits, y, tokenizer.pad_id)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            loss_val = loss.item()
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = 0.9 * ema_loss + 0.1 * loss_val

            if total_step_counter % log_every == 0 or total_step_counter == 1 or total_step_counter == steps:
                dt = time.time() - t0
                if dt == 0: dt = 1e-6  # avoid div by zero
                print(
                    f"step {total_step_counter:5d}/{steps} | loss {loss_val:.4f} | ema {ema_loss:.4f} | {(total_step_counter / dt):.2f} it/s")
                # quick qualitative check
                with torch.no_grad():
                    # Generate a prompt from the *current* difficulty
                    test_cfg = GenConfig(max_digits=batch_max_digits, ops=ops, op_probs=op_probs)
                    prompt = generate_sample_prompt(test_cfg)
                    out = generate(model, tokenizer, prompt, max_new_tokens=max_pos, device=dev)
                    # One-time legend to make logs self-explanatory
                    if not _printed_log_legend:
                        print("  --- Log legend ---")
                        print("  [IN]  input prompt given to the model")
                        print("  [OUT] model's generated scratchpad/text in response to [IN]")
                        print("  [GT]  ground-truth scratchpad for the [IN] prompt (what model should produce)")
                        _printed_log_legend = True

                    print(f"  [IN]  {prompt!r}")
                    print(f"  [OUT] {out!r}")
                    # Generate ground truth for the same prompt
                    m = re.match(r"^\s*(\d+)([+\-*/])(\d+)=", prompt)
                    if m:
                        a_i, b_i, op_s = int(m.group(1)), int(m.group(3)), m.group(2)
                        if op_s == '+':
                            gt_text = _make_add_scratchpad(a_i, b_i, batch_max_digits).strip()
                        elif op_s == '-':
                            gt_text = _make_sub_scratchpad(a_i, b_i, batch_max_digits).strip()
                        elif op_s == '*':
                            gt_text = _make_mul_scratchpad(a_i, b_i).strip()
                        elif op_s == '/':
                            gt_text = _make_div_scratchpad(a_i, b_i).strip()
                        else:
                            gt_text = "(unknown operator)"
                        print(f"  [GT]  {gt_text!r}")

    # Save checkpoint
    # Derive hyperparameters from the current model to ensure consistency on reload
    model_n_embd = getattr(model.tok_emb, 'embedding_dim', n_embd)
    model_n_layer = len(getattr(model, 'blocks', []))
    model_n_head = getattr(model.blocks[0].attn, 'n_heads', n_head) if model_n_layer > 0 else n_head
    model_dropout = getattr(model.drop, 'p', dropout)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': getattr(model, 'vocab_size', tokenizer.vocab_size),
        'n_embd': model_n_embd,
        'n_layer': model_n_layer,
        'n_head': model_n_head,
        'dropout': model_dropout,
        'max_pos': getattr(model, 'max_pos', max_pos),
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # End-of-training evaluation on multiple difficulty levels
    if eval_samples and eval_samples > 0:
        print("\n=== Multi-Level Evaluation ===")
        # Evaluate on each curriculum stage (1-digit, 2-digit, 3-digit, etc.)
        for eval_digits in range(1, max_digits + 1):
            print(f"\n--- Evaluating on {eval_digits}-digit problems ({eval_samples} samples) ---")
            eval_cfg = GenConfig(max_digits=eval_digits, ops=ops)
            stats = evaluate(model, tokenizer, eval_cfg, n_samples=eval_samples, device=dev, max_pos=max_pos)
            overall = stats['overall']
            overall_acc = 100.0 * overall['correct'] / max(1, overall['total'])
            print(f"Results for {eval_digits}-digit:")
            for op in ops:
                c = stats['per_op'][op]
                acc = 100.0 * c['correct'] / max(1, c['total'])
                print(f"  {op}: {c['correct']}/{c['total']} = {acc:.2f}%")
            print(f"  Overall: {overall['correct']}/{overall['total']} = {overall_acc:.2f}%")

        # Extrapolation test: Evaluate on 4-digit problems (minimal sample size)
        extrapolation_samples = max(20, eval_samples // 10)  # Use 10% of eval_samples, minimum 20
        print(f"\n--- Extrapolation Test: 4-digit problems ({extrapolation_samples} samples) ---")
        print("(Testing if model can generalize beyond training difficulty)")
        eval_cfg_4d = GenConfig(max_digits=4, ops=ops)
        stats_4d = evaluate(model, tokenizer, eval_cfg_4d, n_samples=extrapolation_samples, device=dev, max_pos=max_pos)
        overall_4d = stats_4d['overall']
        overall_acc_4d = 100.0 * overall_4d['correct'] / max(1, overall_4d['total'])
        print(f"Results for 4-digit:")
        for op in ops:
            c = stats_4d['per_op'][op]
            acc = 100.0 * c['correct'] / max(1, c['total'])
            print(f"  {op}: {c['correct']}/{c['total']} = {acc:.2f}%")
        print(f"  Overall: {overall_4d['correct']}/{overall_4d['total']} = {overall_acc_4d:.2f}%")

        print("\n===============================")


def generate_sample_prompt(cfg: GenConfig) -> str:
    # (This function is unchanged from 005.py, but is still useful)
    op = _choose_op(cfg)
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
    """Load a saved checkpoint and return (model, tokenizer, device).
    NEW: Now handles 'max_pos'
    """
    dev = pick_device(device)
    tokenizer = CharTokenizer()
    ckpt = torch.load(ckpt_path, map_location=dev)

    # Prefer checkpoint's saved vocab_size to ensure shapes match; fall back to tokenizer size.
    vocab_size = int(ckpt.get('vocab_size', getattr(tokenizer, 'vocab_size', 50)))
    if vocab_size != tokenizer.vocab_size:
        raise RuntimeError(f"Checkpoint vocab_size ({vocab_size}) != current tokenizer vocab_size ({tokenizer.vocab_size}). Incompatible tokenizer/vocab; please retrain or use a matching checkpoint.")

    model = TinyGPT(
        vocab_size=vocab_size,
        n_embd=ckpt.get('n_embd', 128),
        n_layer=ckpt.get('n_layer', 2),
        n_head=ckpt.get('n_head', 4),
        dropout=ckpt.get('dropout', 0.1),
        max_pos=ckpt.get('max_pos', 512),  # Load max_pos
    ).to(dev)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, tokenizer, dev


# -----------------------------
# Evaluation utilities (NEW: Parse scratchpad)
# -----------------------------

def _sample_operands_for_op(op: str, max_digits: int) -> Tuple[int, int]:
    # (UNCHANGED from 005.py)
    if max_digits == 0:  # Handle 0-digit case
        if op == '/': return 0, 1
        return 0, 0
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
    # (UNCHANGED from 005.py)
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
    """
    NEW: Extract the model-predicted result from a scratchpad string.
    We look for the pattern "F:..." where F = Final answer marker.
    Example: From "85+47=[C1:5+7=12;W:2;K:1;C2:8+4+1=13;W:13;F:132]" extracts "132"
    """
    # Regex: capture the last occurrence of F:<int> (allow optional leading minus)
    matches = re.findall(r"F:(-?\d+)", text)
    if matches:
        return matches[-1]

    # Fallback: if no "F:", try to find "...=...[...]\n"
    # and extract the ... part. This is for robustness.
    # But the "F:" marker is the primary method.

    # Fallback 2: your old method (for debugging simple lookups like division)
    if '=' not in text:
        return ''
    tail = text.split('=', 1)[1]
    if '[' in tail:
        tail = tail.split('[', 1)[0]
    if '\n' in tail:
        tail = tail.split('\n', 1)[0]

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
def evaluate(model: TinyGPT, tokenizer: CharTokenizer, cfg: GenConfig, n_samples: int, device: torch.device,
             max_pos: int) -> dict:
    """Run a synthetic test set and compute per-op and overall accuracy.
    NEW: Uses the new _extract_result_from_text
    """
    counts = {op: {'correct': 0, 'total': 0} for op in cfg.ops}
    log_interval = max(1, n_samples // 10)

    for i in range(n_samples):
        if i % log_interval == 0 and i > 0:
            print(f"  eval sample {i}/{n_samples}")

        op = random.choice(cfg.ops)
        a, b = _sample_operands_for_op(op, cfg.max_digits)
        expected = str(_ground_truth(a, b, op))
        prompt = f"{a}{op}{b}="
        # Generate the full scratchpad
        out = generate(model, tokenizer, prompt, max_new_tokens=max_pos, device=device)
        # Extract the final answer from the scratchpad
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
    p_train.add_argument('--steps', type=int, default=1000,
                         help="Total number of training steps/iterations (default: 5000)")
    p_train.add_argument('--batch-size', type=int, default=128,
                         help="Batch size for training. Longer sequences may need smaller batches (default: 128)")
    p_train.add_argument('--max-digits', type=int, default=3,
                         help="Final max digits for curriculum training. E.g., 3 means 1-digit, 2-digit, 3-digit stages (default: 3)")
    p_train.add_argument('--lr', type=float, default=3e-4,
                         help="Learning rate for AdamW optimizer (default: 3e-4)")
    p_train.add_argument('--n-embd', type=int, default=128,
                         help="Embedding dimension size for token and position embeddings (default: 128)")
    p_train.add_argument('--n-layer', type=int, default=6,
                         help="Number of transformer layers/blocks (default: 6)")
    p_train.add_argument('--n-head', type=int, default=4,
                         help="Number of attention heads in multi-head attention (default: 4)")
    p_train.add_argument('--dropout', type=float, default=0.1,
                         help="Dropout probability for regularization (default: 0.1)")
    p_train.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'mps', 'auto'],
                         help="Device to train on: cpu, cuda (NVIDIA GPU), mps (Apple Silicon), or auto (default: auto)")
    p_train.add_argument('--ckpt', type=str, default='math_llm_scratchpad_model-007.pt',
                         help="Checkpoint file path for saving/loading model (default: math_llm_scratchpad_model-007.pt)")
    p_train.add_argument('--log-every', type=int, default=100,
                         help="Print training progress every N steps (default: 100)")
    p_train.add_argument('--eval-samples', type=int, default=500,
                         help='Number of synthetic test samples for evaluation; set to 0 to skip evaluation (default: 500)')
    p_train.add_argument('--ops', type=str, default='+-*/',
                         help="String of operators to train on, e.g., '+-*/' for all four operations (default: +-*/)")
    p_train.add_argument('--op-probs', type=str, default=None,
                         help="Comma-separated operator probabilities (e.g., '0.25,0.25,0.25,0.25'). Not recommended; curriculum is better (default: None)")
    p_train.add_argument('--max-pos', type=int, default=512,
                         help="Maximum sequence length for positional embeddings (context window size) (default: 512)")
    p_train.add_argument('--use-amp', action='store_true',
                         help="Enable automatic mixed precision training using FP16/BF16 for faster training (default: False)")

    p_demo = sub.add_parser('demo', help='Run generation on a prompt like "12+3="')
    p_demo.add_argument('--prompt', type=str, required=True,
                        help='Arithmetic prompt to evaluate, e.g., "12+3=" or "144/12="')
    p_demo.add_argument('--ckpt', type=str, default='math_llm_scratchpad_model.pt',
                        help="Path to checkpoint file to load trained model from (default: math_llm_scratchpad_model.pt)")
    p_demo.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'mps', 'auto'],
                        help="Device to run inference on: cpu, cuda (NVIDIA GPU), mps (Apple Silicon), or auto (default: auto)")
    p_demo.add_argument('--max-new-tokens', type=int, default=512,
                        help="Maximum number of tokens to generate in response. Should be >= model's max_pos (default: 512)")

    args = parser.parse_args()

    if args.cmd == 'train':
        probs_list = None
        if args.op_probs is not None:
            try:
                probs_list = [float(x.strip()) for x in args.op_probs.split(',') if x.strip() != '']
            except ValueError:
                print(f"Warning: Could not parse --op-probs='{args.op_probs}'.")
                probs_list = None
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
            ops=args.ops,
            op_probs=probs_list,
            max_pos=args.max_pos,
            use_amp=args.use_amp,
        )
    elif args.cmd == 'demo':
        DEFAULT_CKPT = 'math_llm_scratchpad_model.pt'
        ckpt_to_load = args.ckpt if os.path.exists(args.ckpt) else DEFAULT_CKPT

        max_pos_for_demo = args.max_new_tokens

        if not os.path.exists(ckpt_to_load):
            print(f"Checkpoint '{ckpt_to_load}' not found. Using randomly initialized model (results will be poor).")
            dev = pick_device(args.device)
            tokenizer = CharTokenizer()
            model = TinyGPT(tokenizer.vocab_size, max_pos=max_pos_for_demo).to(dev)
            ckpt_used = '(random init)'
        else:
            try:
                model, tokenizer, dev = load_model(ckpt_to_load, device=args.device)
                ckpt_used = ckpt_to_load
                # Ensure model's max_pos is respected for generation
                max_pos_for_demo = model.max_pos
                if args.max_new_tokens > model.max_pos:
                    print(
                        f"Warning: --max-new-tokens ({args.max_new_tokens}) > model max_pos ({model.max_pos}). Clamping to {model.max_pos}.")
                    args.max_new_tokens = model.max_pos
            except Exception as e:
                print(f"Failed to load checkpoint '{ckpt_to_load}': {e}. Using random model.")
                dev = pick_device(args.device)
                tokenizer = CharTokenizer()
                model = TinyGPT(tokenizer.vocab_size, max_pos=max_pos_for_demo).to(dev)
                ckpt_used = '(random init)'

        print_startup_info(cmd='demo', requested_device=args.device, picked_device=dev, checkpoint=ckpt_used)
        print_model_summary(model)
        out = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, device=dev)
        print("--- Prompt ---")
        print(args.prompt)
        print("--- Model Output ---")
        print(out)


if __name__ == '__main__':
    main()
