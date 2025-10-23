# My Tiny LLM
#
# Learn simple addition and multiplication
# A 15mn training run (which is probably already overkill) on my macbook air M3 give me 95~99% accuracy
# Total parameters count is around 400k. Which is nanoscopic by LLM standards.
# it can probably be squeezed down to fewer parameters. But i'm planning to add complexity later in 002.py
#
# Let's just say it's a good enough baseline.
# Maybe i'll see if i can do something about learning rate scheduling before switching to 002.py
# And add some more documentation as well. Yeah, let's start with that.


# Imports and setup
from __future__ import annotations

import sys
import numpy as np
import math
import random
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

# overly complicated pytorch import routine
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise SystemExit(
        "This notebook requires PyTorch. Please install it first, e.g.\n"
        "pip install torch --extra-index-url https://download.pytorch.org/whl/cpu\n"
        f"Import error was: {e}"
    )

# Print some useless stuff that may or may not be useful for debugging.
print("Torch version:", torch.__version__)
print("Numpy version:", np.__version__)
print("Python version:", sys.version)

# Device selection: prefer CUDA, then Apple MPS (Metal), then CPU
# I'm on a mac so i use MPS and the hyperparameters are tunned for it. Feel free to change it, especially if your on NVidia.
if torch.cuda.is_available():
    Device = torch.device("cuda")
# MPS might not be the fastest device on apple silicon, feel free to experiment with pure cpu
# by commenting this related section below.
# I could make something nicer to pick MPS/CPU but if you're at the level of not being able to do that yourself,
# then you're probably not at the level of being able to do anything useful with LLMs. So... yeah. Comment it, or not.
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    Device = torch.device("mps")
else:
    Device = torch.device("cpu")
print("Using device:", Device)


# Note: no fixed random seed to keep runs stochastic by default.
# If you need reproducibility, you can set a seed manually here or via an env var.
# Example (disabled by default):
# import os
# seed_str = os.environ.get("SEED")
# if seed_str is not None:
#     seed = int(seed_str)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# Vocabulary and tokenization
# Create a set of unique characters from the training data.
class CharVocab:
    def __init__(self, chars: str):
        # ensure deterministic order and uniqueness
        uniq = []
        seen = set()
        for c in chars:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        self.chars = uniq
        # string-to-index mapping for encoding text to token IDs
        self.stoi: Dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        # index-to-string mapping for decoding token IDs back to text
        self.itos: Dict[int, str] = {i: c for i, c in enumerate(self.chars)}

    def encode(self, s: str) -> List[int]:
        """Convert a string to a list of token IDs."""
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to a string."""
        return ''.join(self.itos[i] for i in ids)


# Digits + operators + equals + newline + optional space for padding (also makes text nicer)
VOCAB_CHARS = "0123456789+*=\n "
vocab = CharVocab(VOCAB_CHARS)
EOS_TOKEN = "\n"  # we generate until newline
EOS_ID = vocab.stoi[EOS_TOKEN]  # the token ID for end-of-sequence


# Data generation utilities
def sample_expression(add_max: int = 999, mul_max: int = 20, p_add: float = 0.5) -> str:
    """
    Randomly sample either an addition or multiplication problem and its answer as a single line.

    This function generates mathematical problems for training the model. It randomly selects
    between addition and multiplication operations with specified value ranges.

    Args:
        add_max (int): Maximum value for addition operands (default: 999)
        mul_max (int): Maximum value for multiplication operands (default: 20)
        p_add (float): Probability of generating an addition problem (default: 0.5)

    Returns:
        str: A complete training sequence including the problem and answer with newline,
             e.g. "12+7=19\n" or "3*4=12\n"

    Example:
        >>> sample_expression()
        "42+18=60\n"
        >>> sample_expression(mul_max=5, p_add=0.0)
        "3*4=12\n"
    """
    if random.random() < p_add:
        # generate addition problem
        a = random.randint(0, add_max)
        b = random.randint(0, add_max)
        expr = f"{a}+{b}={a + b}\n"
    else:
        # generate multiplication problem
        a = random.randint(0, mul_max)
        b = random.randint(0, mul_max)
        expr = f"{a}*{b}={a * b}\n"
    return expr


# Obviously, the cool stuff happens here.
class TinyCausalLM(nn.Module):
    """
    A tiny causal language model for learning simple mathematical operations.

    This transformer-based model is designed to learn addition and multiplication
    by predicting the next character in mathematical expressions. It uses a 
    causal (masked) self-attention mechanism to process sequences of mathematical
    expressions.

    The model architecture includes:
    - Token embeddings to map discrete tokens to dense vectors
    - Positional encodings to provide sequence order information  
    - Multiple transformer encoder layers for contextual processing
    - Output projection to vocabulary logits for next-token prediction

    Args:
        vocab_size (int): Size of the vocabulary (number of unique characters)
        d_model (int): Dimension of the token embeddings and transformer layers (default: 128)
        nhead (int): Number of attention heads in transformer layers (default: 4)
        num_layers (int): Number of transformer encoder layers (default: 3)
        dim_ff (int): Dimension of the feedforward network in transformer layers (default: 256)
        max_len (int): Maximum sequence length the model can handle (default: 64)
        dropout (float): Dropout rate for regularization (default: 0.1)

    Example:
        >>> model = TinyCausalLM(vocab_size=15, d_model=128, nhead=4, num_layers=3)
        >>> logits = model(torch.tensor([[1, 2, 3]]))  # (batch_size, sequence_length, vocab_size)
    """

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dim_ff: int = 256, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        # token embedding: map token IDs to d_model-dimensional vectors
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        """**Token Embeddings**: Token embeddings map token IDs to dense vectors in the embedding space.
        This allows the model to process discrete tokens as continuous representations,
        enabling it to capture semantic and syntactic relationships between tokens.
        """
        # positional embedding: add position information to tokens
        self.pos_emb = nn.Embedding(max_len, d_model)
        """**Positional Encodings**: Positional encodings provide additional information about the position of tokens in a sequence.
        Since transformer models are permutation invariant, adding positional encodings helps them understand the order of tokens.
        In this model, we use simple learned positional embeddings that are added to token embeddings.
        """
        # build a single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True,
            activation="gelu"
        )
        # stack multiple transformer layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # output projection: map transformer outputs to vocabulary logits
        self.lm_head = nn.Linear(d_model, vocab_size)
        # Precompute and cache a full causal mask up to max_len; will be moved with .to(Device)
        # I have no idea what this does, Claude AI Suggested it.
        self.register_buffer(
            "causal_mask_full",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) integer token ids
        returns logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")
        # create positional indices [0, 1, 2, ..., T-1] for each batch element
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0).expand(B, T)
        # combine token and positional embeddings
        x = self.tok_emb(idx) + self.pos_emb(pos)
        # slice the precomputed causal mask to current sequence length
        mask = self.causal_mask_full[:T, :T]
        # pass through transformer layers with causal masking
        x = self.transformer(x, mask=mask)
        # project to vocabulary to get next-token logits
        logits = self.lm_head(x)
        return logits


# Reading the loss and perplexity
# - The printed training loss is token-level cross-entropy (negative log-likelihood) averaged over characters.
# - Vocabulary size here is 15 characters (digits 0–9, '+', '*', '=', newline, space).
# - A random next-character guess has expected loss ln(15) ≈ 2.71 (perplexity ≈ 15).
# - Perplexity = exp(loss). Rough intuition for this task:
#   - ~2.7: random; just started.
#   - 1.0 (ppl≈2.7): model is learning patterns but answers may be unreliable.
#   - 0.7–0.4 (ppl≈2.0–1.5): typically usable.
#   - <0.3 (ppl<1.35): often near-perfect on training distribution.
#   - <0.1 (ppl≈1.1): essentially deterministic.
# - Rule of thumb: judge by exact-match accuracy on held-out samples (see eval cell below). If accuracy is high (e.g., >99%), your loss is “good enough,” even if it’s not extremely small.
#
@dataclass
class TrainConfig:
    """Hyperparameters for model architecture and training."""
    steps: int = 10000  # total training steps
    lr: float = 3e-3  # learning rate
    print_every: int = 100  # how often to print training progress
    add_max: int = 999  # maximum value for addition operands
    mul_max: int = 20  # maximum value for multiplication operands
    p_add: float = 0.5  # probability of sampling addition vs multiplication
    d_model: int = 128  # transformer embedding dimension
    nhead: int = 4  # number of attention heads
    num_layers: int = 3  # number of transformer layers
    dim_ff: int = 256  # feedforward network dimension
    max_len: int = 64  # maximum sequence length
    dropout: float = 0.1  # dropout rate
    batch_size: int = 128  # batch size for training


cfg = TrainConfig()
# instantiate the model and move it to the selected device
model = TinyCausalLM(
    vocab_size=len(vocab.chars),
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dim_ff=cfg.dim_ff,
    max_len=cfg.max_len,
    dropout=cfg.dropout,
).to(Device)

# AdamW optimizer with weight decay for better generalization
optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

# Checkpointing: auto-load if exists, and configure a save path
ckpt_path = Path("tiny_lm_addmul.pt")
resume_info = None
if ckpt_path.exists():
    try:
        # load saved model and optimizer state
        state = torch.load(ckpt_path, map_location=Device)
        model.load_state_dict(state.get("model", {}))
        if "optim" in state:
            try:
                optim.load_state_dict(state["optim"])
            except Exception:
                # If optimizer dimensions changed, continue without optimizer state
                pass
        loaded_step = state.get("step", None)
        ema_loss_loaded = state.get("ema_loss", None)
        resume_info = (loaded_step, ema_loss_loaded)
        print(f"Loaded checkpoint from {ckpt_path} (step={loaded_step}, ema_loss={ema_loss_loaded}).")
    except Exception as e:
        print(f"Warning: failed to load checkpoint from {ckpt_path}: {e}\nTraining from scratch.")
else:
    print("No checkpoint found; training from scratch.")


# Keep a simple per-string loss for eval; training uses batched preallocated buffers
def loss_on_string(s: str) -> torch.Tensor:
    """
    Compute cross-entropy loss for a single training string (for evaluation).

    This function calculates the negative log-likelihood loss for a given training sequence,
    which is used during evaluation to assess model performance on individual examples.

    Args:
        s (str): A complete training sequence including problem and answer (e.g., "12+7=19\n")

    Returns:
        torch.Tensor: The computed cross-entropy loss for the sequence

    Note:
        This function is primarily used for evaluation purposes. For training, batched
        computation is used for efficiency.
    """
    ids = vocab.encode(s)
    # input: all tokens except the last
    x = torch.tensor(ids[:-1], dtype=torch.long, device=Device).unsqueeze(0)  # (1, T-1)
    # target: all tokens except the first (shifted by 1)
    y = torch.tensor(ids[1:], dtype=torch.long, device=Device).unsqueeze(0)  # (1, T-1)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return loss


# Preallocate training buffers (B, T) and reuse them every step for efficiency
pad_id = vocab.stoi[' ']  # use space as padding token
B = cfg.batch_size
T = cfg.max_len - 1  # leave room for shifted target
V = len(vocab.chars)
# allocate input and target buffers on device, filled with padding
x_buf = torch.full((B, T), pad_id, dtype=torch.long, device=Device)
y_buf = torch.full((B, T), pad_id, dtype=torch.long, device=Device)


def fill_batch_inplace() -> str:
    """
    Fills x_buf and y_buf with a freshly sampled batch of training data.

    This function populates the preallocated training buffers with randomly sampled
    mathematical problems. It's designed for efficiency by reusing allocated memory
    instead of creating new tensors on each iteration.

    Returns:
        str: One example string from the batch, useful for logging training progress

    Note:
        The function modifies x_buf and y_buf in-place and returns a sample example
        for display purposes. The buffers are filled with padding tokens initially,
        then overwritten with actual training data.
    """
    x_buf.fill_(pad_id)
    y_buf.fill_(pad_id)
    last_s = None
    for i in range(B):
        s = sample_expression(cfg.add_max, cfg.mul_max, cfg.p_add)
        ids = vocab.encode(s)
        # truncate to fit into buffers if needed
        t = min(len(ids) - 1, T)
        if t > 0:
            xi = torch.tensor(ids[:t], dtype=torch.long)
            yi = torch.tensor(ids[1:1 + t], dtype=torch.long)
            x_buf[i, :t] = xi.to(Device)
            y_buf[i, :t] = yi.to(Device)
        last_s = s
    return last_s or ""


# === TRAINING LOOP EXPLANATION ===
# This is the core training process where the model learns to predict math operations.
# The training loop:
# 1. Samples a batch of math problems (addition/multiplication with answers)
# 2. For each problem, the model predicts what comes next character by character
# 3. Compares predictions against actual answers using cross-entropy loss
# 4. Uses backpropagation to adjust model weights to reduce prediction errors
# 5. Repeats this process for many steps (cfg.steps) to improve accuracy

print("Training...")
model.train()
ema_loss = None
ema_alpha = 0.05  # EMA smoothing factor for exponentially weighted average loss
save_every = max(1000, cfg.print_every)  # checkpointing frequency
for step in range(1, cfg.steps + 1):
    # Sample a fresh batch into the preallocated buffers (this is where we get new training data)
    example = fill_batch_inplace()

    # === FORWARD PASS ===
    # The model processes the input tokens through all transformer layers to generate predictions
    # For each position in the sequence, the model outputs logits for every possible token in vocabulary
    logits = model(x_buf)

    # === LOSS COMPUTATION ===
    # Cross-entropy loss measures how "wrong" our predictions are compared to actual answers
    # We reshape logits and targets to flatten them into 2D tensors (batch_size*sequence_length, vocab_size)
    # ignore_index=pad_id ignores padding tokens when computing loss (they don't count toward accuracy)
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        y_buf.reshape(-1),
        ignore_index=pad_id,
    )

    # === BACKWARD PASS AND OPTIMIZATION ===
    # Zero gradients from previous step to prevent accumulation
    optim.zero_grad(set_to_none=True)
    # Compute gradients of loss with respect to model parameters using backpropagation
    loss.backward()
    # Update model weights using AdamW optimizer based on computed gradients
    optim.step()

    # === TRAINING PROGRESS TRACKING ===
    # Keep track of average loss over time using exponential moving average for smoother display
    curr = float(loss.item())
    ema_loss = curr if ema_loss is None else (1 - ema_alpha) * ema_loss + ema_alpha * curr
    ppl = math.exp(curr)
    ema_ppl = math.exp(ema_loss)

    # Print training progress every cfg.print_every steps
    if step % cfg.print_every == 0 or step == 1:
        print(
            f"step {step:4d} | loss {curr:.4f} (ppl {ppl:.2f}) | ema {ema_loss:.4f} (ppl {ema_ppl:.2f}) | ex: {example.strip()}")

    # === CHECKPOINTING ===
    # Periodically save the model state so we can resume training if interrupted
    # This saves model weights, optimizer state, and training progress
    if (step % save_every == 0) or (step == cfg.steps):
        try:
            # save model, optimizer, config, and training state
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "cfg": cfg.__dict__,
                "vocab": VOCAB_CHARS,
                "step": step,
                "ema_loss": float(ema_loss) if ema_loss is not None else None,
            }, ckpt_path)
            if step % save_every == 0 or step == cfg.steps:
                pass  # quiet success; print can add noise
        except Exception as e:
            print(f"Warning: failed to save checkpoint to {ckpt_path}: {e}")


# %%
# Generation (greedy)

def generate(prompt: str, max_new_tokens: int = 8) -> str:
    """
    Generate tokens autoregressively from a prompt using greedy decoding.

    This function takes a mathematical expression prompt and generates the complete
    answer by predicting one token at a time until end-of-sequence token is generated
    or max_new_tokens is reached.

    Args:
        prompt (str): The mathematical expression to complete (e.g., "12+7=")
        max_new_tokens (int): Maximum number of tokens to generate (default: 8)

    Returns:
        str: The complete mathematical expression with generated answer

    Example:
        >>> generate("12+7=")
        "12+7=19\n"
    """
    model.eval()
    with torch.no_grad():
        ids = vocab.encode(prompt)
        for _ in range(max_new_tokens):
            # use only the most recent (max_len-1) tokens as context
            x = torch.tensor(ids[-(cfg.max_len - 1):], dtype=torch.long, device=Device).unsqueeze(0)
            logits = model(x)  # (1, T, V)
            next_logits = logits[0, -1]  # get logits for the next token (V,)
            next_id = int(torch.argmax(next_logits).item())  # pick most likely token
            ids.append(next_id)
            # stop if we generated end-of-sequence token
            if next_id == EOS_ID:
                break
    return vocab.decode(ids)


# %%
# Evaluation function

def eval_examples(n: int = 1000, seed: int | None = None):
    """
    Evaluate the model on randomly sampled math problems.

    This function tests the model's ability to solve mathematical problems by
    generating answers for randomly sampled addition and multiplication problems.
    It computes exact match accuracy and average loss over the evaluation set.

    Args:
        n (int): Number of examples to evaluate (default: 1000)
        seed (int | None): Random seed for reproducible evaluation (default: None)

    Returns:
        None: Prints evaluation results to console

    Example:
        >>> eval_examples(100)
        Eval on 100 samples -> exact-match: 95.0% | loss 0.3421 | ppl 1.40

    The function:
    - Samples n random math problems (addition/multiplication with answers)
    - For each problem, extracts the prompt (everything up to and including '=')
    - Generates complete answers using the generate() function
    - Compares generated answers with ground truth
    - Reports exact match accuracy, average loss, and perplexity
    - Shows up to 5 examples of incorrect predictions for debugging
    """
    if seed is not None:
        # set seed for reproducible evaluation
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    model.eval()
    exact = 0
    total = 0
    avg_loss = 0.0
    wrong = []
    with torch.no_grad():
        for _ in range(n):
            # sample a problem
            s = sample_expression(cfg.add_max, cfg.mul_max, cfg.p_add)
            avg_loss += float(loss_on_string(s).item())
            # extract the prompt (everything up to and including '=')
            eq_idx = s.index("=")
            prompt = s[:eq_idx + 1]
            # generate answer
            out = generate(prompt)
            correct = (out == s)
            exact += int(correct)
            total += 1
            if not correct and len(wrong) < 5:
                # store ground truth and predicted answer tail for a quick glance
                wrong.append((s.strip(), out[len(prompt):].strip()))
    avg_loss /= max(total, 1)
    print(
        f"Eval on {total} samples -> exact-match: {exact / total * 100:.1f}% | loss {avg_loss:.4f} | ppl {math.exp(avg_loss):.2f}")
    if wrong:
        print("Examples of mistakes (truth -> predicted):")
        if wrong:
            # Group similar mistakes together
            grouped_mistakes = {}
            for gt, pred in wrong:
                if pred not in grouped_mistakes:
                    grouped_mistakes[pred] = []
                grouped_mistakes[pred].append(gt)

            # Print grouped mistakes with a limit of 5 examples per group
            for pred, gt_list in grouped_mistakes.items():
                print(f"  {pred!r}:")
                for gt in gt_list[:5]:
                    print(f"    {gt}")
                if len(gt_list) > 5:
                    print("    ... (and more)")


# Run a small eval after training
try:
    eval_examples(1000)
except (ValueError, TypeError) as ve:
    print(f"Evaluation failed with a {type(ve).__name__}:")
    print(str(ve))
except Exception as e:
    print("Eval skipped due to an unexpected error:", str(e))
