from __future__ import annotations

import os
import re
import math
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Tuple



class CharTokenizer:
    """Simple character-level tokenizer

    Vocabulary:
    - 0-9: digits
    - +-*/=: arithmetic operators
    - space and newline: whitespace
    - []: scratchpad start/end brackets
    - ;: step separator in scratchpad
    - :: marker prefix separator
    - >: used in borrow notation (e.g., 5->4)
    - FAPCWKBMR: scratchpad format markers:
        F = Final answer (e.g., F:132)
        A = Sum of partial products in multiplication (e.g., A:26+130=156)
        P = Partial product in multiplication (e.g., P1:13*2)
        C = Column number in addition/subtraction (e.g., C1:5+7=12)
        W = Write digit to result (e.g., W:2)
        K = Carry value in addition (e.g., K:1)
        B = Borrow operation in subtraction (e.g., B:5->4)
        M = Single-digit multiplication step (e.g., M1:3*2=6)
        R = Result of partial product (e.g., R1:26)
    - <PAD>: padding token (ID 0)
    """

    def __init__(self):
        """Initialize tokenizer with character vocabulary."""
        self.pad_token = '<PAD>'
        self.pad_id = 0
        # Build vocabulary: digits, operators, special chars, scratchpad markers
        chars = list("0123456789+-*/= \n[];:FAPCWKBMR>") # TODO: is space really necessary?
        # Reserve 0 for PAD, others start from 1
        self.itos = [self.pad_token] + chars  # index to string mapping
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}  # string to index mapping

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size (number of unique tokens)."""
        return len(self.itos)

    def encode(self, s: str) -> List[int]:
        """Convert string to list of token IDs. Skips unknown characters."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        """Convert list of token IDs back to string. Skips padding tokens."""
        return ''.join(self.itos[i] for i in ids if i != self.pad_id)


@dataclass
class GenConfig:
    """Configuration for synthetic arithmetic data generation."""
    max_digits: int = 3  # Maximum number of digits in operands
    ops: str = "+-*/"  # String of operators to use
    # Optional per-operator sampling probabilities aligned with `ops`. TODO: remove it
    op_probs: List[float] | None = None


def _rand_int(max_digits: int) -> int:
    """Generate random integer with up to max_digits digits.

    Args:
        max_digits: Maximum number of digits (e.g., 2 generates 0-99)

    Returns:
        Random integer in range [0, 10^max_digits - 1]
    """
    #if max_digits == 0: return 0
    low = 0
    high = 10 ** max_digits - 1
    return random.randint(low, high)


def _make_div_pair(max_digits: int) -> Tuple[int, int]:
    """Generate operand pair (a, b) for integer division.

    Generates two random numbers within max_digits constraint.
    Uses integer division (//) in scratchpad generation.

    Args:
        max_digits: Maximum digits for both dividend and divisor

    Returns:
        Tuple (dividend, divisor) where both have ≤ max_digits digits
        and divisor is guaranteed to be non-zero
    """

    dividend = _rand_int(max_digits)  # dividend
    divisor = random.randint(1, 10 ** max_digits - 1)  # divisor (non-zero)

    return dividend, divisor


def _make_sub_pair(max_digits: int) -> Tuple[int, int]:
    """Generate operand pair for subtraction.

    Generates two random integers. Both A-B and B-A orderings are used
    in training to teach commutative understanding.

    Args:
        max_digits: Maximum number of digits in each operand

    Returns:
        Tuple (a, b) of random integers
    """
    a = _rand_int(max_digits)
    b = _rand_int(max_digits)
    return a, b


def _pad_numbers_to_length(a: int, b: int, length: int) -> tuple[str, str]:
    """Pad two numbers with leading zeros to specified length.

    Args:
        a: First number
        b: Second number
        length: Desired string length

    Returns:
        Tuple of (padded_a, padded_b) as strings
    """
    return str(a).zfill(length), str(b).zfill(length)


def _process_multiplication_digit(a_digit: int, b_digit: int, carry: int, pos: int, steps: list[str]) -> tuple[
    int, int]:
    """Process single digit multiplication with carry for one position.

    Updates the steps list in place and returns the digit and new carry.

    Args:
        a_digit: Digit from first operand
        b_digit: Digit from second operand (single digit multiplier)
        carry: Carry from previous position
        pos: Position in result (1-indexed, from right)
        steps: List to append step strings to

    Returns:
        Tuple of (digit_out, new_carry)
    """
    product = a_digit * b_digit + carry
    digit_out = product % 10
    new_carry = product // 10

    # Show the single-digit multiplication step
    if carry > 0:
        steps.append(f"M{pos}:{a_digit}*{b_digit}+{carry}={product}")
    else:
        steps.append(f"M{pos}:{a_digit}*{b_digit}={a_digit * b_digit}")

    steps.append(f"W:{digit_out}")  # write this digit
    return digit_out, new_carry


def _record_column_operation(col_num: int, d1: int, d2: int, op: str, carry: int = 0) -> str:
    """Create a column operation string (e.g., 'C1:5+7=12' or 'C2:8-3=5').

    Args:
        col_num: Column number (1 = rightmost)
        d1: First digit
        d2: Second digit
        op: Operator ('+' or '-')
        carry: Optional carry/borrow value to include

    Returns:
        Formatted column step string
    """
    result = (d1 + carry + d2) if op == '+' else (d1 - d2)
    step = f"C{col_num}:{d1}{op}{d2}"
    if carry > 0:
        step += f"+{carry}"
    step += f"={result}"
    return step


def _format_scratchpad(a: int, b: int, op: str, steps: list[str], result: int) -> str:
    """Format the complete scratchpad string with problem, steps, and result.

    Args:
        a: First operand
        b: Second operand
        op: Operator character
        steps: List of scratchpad step strings
        result: Final answer

    Returns:
        Complete formatted scratchpad string with newline
    """
    final_steps = _cleanup_leading_zero_steps(steps, result)
    final_steps.append(f"F:{result}")
    return f"{a}{op}{b}=[{';'.join(final_steps)}]\n"


def _cleanup_leading_zero_steps(steps: list[str], result: int) -> list[str]:
    """Remove unnecessary leading zero column steps from scratchpad.

    Keeps only the rightmost columns needed to represent the result.
    For example, if result is 132 (3 digits), keep only C1, C2, C3 steps.

    Args:
        steps: List of scratchpad step strings
        result: The final numeric result

    Returns:
        Filtered list of steps without leading zero columns
    """
    final_steps = []
    # Handle negative results, which don't have column steps
    if not any(s.startswith("C") for s in steps):
        return steps

    final_res_str = str(abs(result))
    keep_cols = max(1, len(final_res_str))  # number of columns to keep from the right (C1..Ckeep)
    col_num = 0

    for step in steps:
        if step.startswith("C"):
            # Extract column number from step like "C2:..."
            col_num_match = re.search(r"C(\d+):", step)
            if col_num_match:
                col_num = int(col_num_match.group(1))
            if col_num <= keep_cols:
                final_steps.append(step)
        elif step.startswith("W:") or step.startswith("K:") or step.startswith("B:"):
            # Keep write/carry/borrow steps only for relevant columns
            if col_num <= keep_cols:
                final_steps.append(step)
        else:
            # Keep other steps (like F:, P:, etc.)
            final_steps.append(step)

    return final_steps


def _make_add_scratchpad(a: int, b: int, max_digits: int) -> str:
    """Generates a scratchpad for right-to-left addition with carry.
    Example: 85+47=[C1:5+7=12;W:2;K:1;C2:8+4+1=13;W:13;F:132]

    Scratchpad format:
    - C{n}: Column n (rightmost is C1)
    - W: Write this digit to result
    - K: Carry this value to next column
    - F: Final answer
    """
    # Calculate result
    res = a + b

    # Pad to max_digits + 1 (to handle overflow/carry)
    max_len = max(len(str(a)), len(str(b))) + 1
    a_s, b_s = _pad_numbers_to_length(a, b, max_len)

    steps = []
    carry = 0
    result_so_far = ""

    # Process addition right-to-left (least significant digit first)
    for i in range(max_len - 1, -1, -1):
        d1 = int(a_s[i])  # digit from first number
        d2 = int(b_s[i])  # digit from second number
        col_num = max_len - i  # column number (C1 is rightmost)
        s = d1 + d2 + carry  # sum including carry from previous column

        # Record the addition step for this column
        steps.append(_record_column_operation(col_num, d1, d2, '+', carry))

        # Calculate new digit and carry
        write_val = s % 10  # digit to write (ones place)
        carry = s // 10  # carry to next column (tens place)

        steps.append(f"W:{write_val}")  # write this single digit to result
        result_so_far = str(write_val) + result_so_far
        if carry > 0:
            steps.append(f"K:{carry}")  # record carry for next step

    # Format and return complete scratchpad
    return _format_scratchpad(a, b, '+', steps, res)


def _make_sub_scratchpad(a: int, b: int, max_digits: int) -> str:
    """Generates a scratchpad for right-to-left subtraction with borrow.
    Example: 52-27=[C1:2-7;B:5->4;C1:12-7=5;W:5;C2:4-2=2;W:2;F:25]

    Scratchpad format:
    - C{n}: Column n (rightmost is C1)
    - B: Borrow operation (e.g., B:5->4 means digit 5 becomes 4 after borrowing)
    - W: Write this digit to result
    - F: Final answer
    """
    # Calculate result
    res = a - b

    # Handle negative results simply ( TODO: no, need a scratchpad )
    if res < 0:
        return f"{a}-{b}=[F:{res}]\n"

    # Pad to at least max_digits or the length of the longer number
    max_len = max(len(str(a)), len(str(b)), max_digits)
    a_s, b_s = _pad_numbers_to_length(a, b, max_len)

    steps = []
    a_list = [int(d) for d in a_s]  # convert to mutable list for borrowing

    # Process subtraction right-to-left (least significant digit first)
    for i in range(max_len - 1, -1, -1):
        d1 = a_list[i]  # digit from minuend (top number)
        d2 = int(b_s[i])  # digit from subtrahend (bottom number)
        col_num = max_len - i  # column number (C1 is rightmost)

        # Check if we need to borrow
        if d1 < d2:
            # Record the initial subtraction attempt (shows we need to borrow)
            steps.append(f"C{col_num}:{d1}-{d2}")
            # Find first non-zero digit to the left to borrow from
            j = i - 1
            while j >= 0 and a_list[j] == 0:
                j -= 1

            if j < 0:  # Should not happen if a > b
                # This indicates a bug in _make_sub_pair or logic
                return f"{a}-{b}=[F:{res}]\n"  # Fallback

            # Borrow from a_list[j] (reduce by 1)
            steps.append(f"B:{a_list[j]}->{a_list[j] - 1}")
            a_list[j] -= 1
            # Propagate borrow through intermediate zeros (they become 9)
            for k in range(j + 1, i):
                # a_list[k] was 0, becomes 9 after borrow
                steps.append(f"B:0->9")  # Show 0 becomes 9
                a_list[k] = 9

            d1 += 10  # add 10 to current digit (borrowed 1 from higher place)
            # Record the subtraction with borrowed amount
            steps.append(f"C{col_num}:{d1}-{d2}={d1 - d2}")
            steps.append(f"W:{d1 - d2}")
        else:
            # No borrow needed, simple subtraction
            steps.append(_record_column_operation(col_num, d1, d2, '-'))
            steps.append(f"W:{d1 - d2}")

    # Format and return complete scratchpad
    return _format_scratchpad(a, b, '-', steps, res)


def _make_mul_scratchpad(a: int, b: int) -> str:
    """Generates a scratchpad for multiplication via partial products with digit-by-digit breakdown.
    Example: 13*12=[P1:13*2;M1:3*2=6;W:6;M2:1*2=2;W:2;R1:26;P2:13*10;M1:3*1=3;W:3;M2:1*1=1;W:1;R2:130;A:26+130=156;F:156]

    Scratchpad format:
    - P{n}: Start partial product n (e.g., P1:13*2)
    - M{n}: Single digit multiplication step (e.g., M1:3*2=6)
    - W:{d}: Write digit to result
    - K:{c}: Carry value
    - R{n}: Result of partial product (e.g., R1:26)
    - A: Add all partial products together
    - F: Final answer
    """
    # Calculate result
    res = a * b

    b_s = str(b)
    steps = []
    partials = []

    if a == 0 or b == 0:
        steps.append(f"P1:{a}*0=0")
        steps.append(f"F:0")
        return f"{a}*{b}=[{';'.join(steps)}]\n"

    # Process each digit of b from right to left (ones, tens, hundreds, etc.)
    for b_idx, b_digit_ch in enumerate(reversed(b_s)):
        b_digit = int(b_digit_ch)  # current digit of second operand
        if b_digit == 0:
            continue  # Skip zero digits (no contribution to product)

        multiplier = b_digit * (10 ** b_idx)  # place value (e.g., 2*1, 1*10)

        # Start this partial product (e.g., "P1:13*2")
        steps.append(f"P{b_idx + 1}:{a}*{multiplier}")

        # Multiply a by the single digit b_digit, processing digit by digit with carries
        a_s = str(a)
        carry = 0
        partial_result = ""

        # Process each digit of a from right to left
        for a_pos, a_digit_ch in enumerate(reversed(a_s), start=1):
            a_digit = int(a_digit_ch)  # current digit of first operand

            # Process this digit multiplication
            digit_out, new_carry = _process_multiplication_digit(a_digit, b_digit, carry, a_pos, steps)

            # Record carry for next digit if needed
            if new_carry > 0 and a_pos < len(a_s):
                steps.append(f"K:{new_carry}")

            partial_result = str(digit_out) + partial_result
            carry = new_carry

        # Write any remaining carry (most significant digit)
        if carry > 0:
            steps.append(f"W:{carry}")
            partial_result = str(carry) + partial_result

        # Adjust for place value (multiply by 10^b_idx to shift left)
        partial_value = int(partial_result) * (10 ** b_idx)
        steps.append(f"R{b_idx + 1}:{partial_value}")  # record partial product result
        partials.append(partial_value)

    # Add all partial products to get final result
    if len(partials) == 0:
        partials.append(0)  # edge case: no partials generated (e.g. b was 0)

    if len(partials) == 1:
        # Only one partial product (e.g., single-digit multiplier)
        steps.append(f"A:{partials[0]}={partials[0]}")
    else:
        # Multiple partial products to add (e.g., "A:26+130=156")
        sum_str = "+".join(map(str, partials))
        steps.append(f"A:{sum_str}={sum(partials)}")

    # Format and return complete scratchpad (no leading zero cleanup for multiplication)
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
    - F:{q}: final quotient (integer division, remainder discarded)

    Note: Uses integer division (//). Any remainder after processing all digits
    is discarded. The scratchpad shows the step-by-step process but only reports
    the final quotient (remainder is not included in the output).
    """
    if b == 0:
        # raise an exception if b is zero, this should not happen
        raise ValueError("Division by zero is undefined")
        #return f"{a}/{b}=[F:0]\n"
    dividend_str = str(a)
    divisor = b

    steps: List[str] = []
    q_digits: List[str] = []
    chunk = 0
    started = False
    n = len(dividend_str)
    step_num = 0  # Sequential step counter

    # Process each digit of dividend from left to right (long division)
    for i, ch in enumerate(dividend_str, start=1):
        # Bring down next digit and form the new chunk (working value)
        chunk = chunk * 10 + int(ch)
        # Choose quotient digit for this place (how many times divisor fits in chunk)
        qd = chunk // divisor if divisor != 0 else 0

        # Skip emitting steps for leading zero quotient digits (haven't started yet)
        if not started and qd == 0 and i < n:
            # carry chunk forward and continue to next digit
            continue

        # From here on, we are emitting quotient digits (including zeros)
        started = True
        step_num += 1  # Increment step counter for scratchpad markers
        # Comparator context for this step (is chunk >= divisor?)
        steps.append(f"C{step_num}:{chunk}>={divisor}")
        prod = qd * divisor  # product of quotient digit and divisor
        rem = chunk - prod  # remainder after subtracting product
        # Record product and subtraction steps
        steps.append(f"P{step_num}:{qd}*{divisor}={prod}")
        steps.append(f"C{step_num}:{chunk}-{prod}={rem}")
        # Write the quotient digit
        steps.append(f"W:{qd}")
        q_digits.append(str(qd))
        # Carry remainder to next step (bring down next digit in next iteration)
        chunk = rem

    # If we never wrote a digit (e.g., a < b),
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


def _choose_op(cfg: GenConfig) -> str:
    """Choose an operator, optionally using configured probabilities.

    Args:
        cfg: Configuration with operators and optional probabilities

    Returns:
        Single character operator ('+', '-', '*', or '/')
    """
    ops = list(cfg.ops)
    weights = None
    # TODO: remove all this probability stuff
    if cfg.op_probs is not None:
        # Use custom operator probabilities if provided and valid
        if len(cfg.op_probs) == len(ops):
            weights = [max(0.0, float(w)) for w in cfg.op_probs]
            if sum(weights) <= 0:
                weights = None  # invalid weights, fall back to uniform
    if weights is None:
        return random.choice(ops)  # uniform random selection
    return random.choices(ops, weights=weights, k=1)[0]  # weighted random selection


def generate_scratchpad_sample(cfg: GenConfig) -> str:
    """Generate one synthetic arithmetic sample as text WITH SCRATCHPAD.

    Creates a training sample showing the step-by-step working (scratchpad)
    for an arithmetic operation.

    Args:
        cfg: Generation configuration (max_digits, operators, etc.)

    Returns:
        String like "85+47=[C1:5+7=12;W:2;K:1;C2:8+4+1=13;W:13;F:132]\n"
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


def generate_scratchpad_pair(cfg: GenConfig) -> Tuple[str, str | None]:
    """Generate a pair of scratchpad samples with operands in both orders.

    Creates two training samples with swapped operands to teach commutativity
    and improve generalization. Division (by zero) is a special case

    Args:
        cfg: Generation configuration (max_digits, operators, etc.)

    Returns:
        Tuple (sample1, sample2) where:
        - For +, -, *: (A op B, B op A)
        - For /: (A / B, None)
    """
    op = _choose_op(cfg)
    if op == '+':
        a, b = _rand_int(cfg.max_digits), _rand_int(cfg.max_digits)
        t1 = _make_add_scratchpad(a, b, cfg.max_digits)
        t2 = _make_add_scratchpad(b, a, cfg.max_digits)
        return t1, t2
    elif op == '-':
        a, b = _make_sub_pair(cfg.max_digits)
        t1 = _make_sub_scratchpad(a, b, cfg.max_digits)
        t2 = _make_sub_scratchpad(b, a, cfg.max_digits)
        return t1, t2
    elif op == '*':
        a, b = _rand_int(cfg.max_digits), _rand_int(cfg.max_digits)
        t1 = _make_mul_scratchpad(a, b)
        t2 = _make_mul_scratchpad(b, a)
        return t1, t2
    elif op == '/':
        a, b = _make_div_pair(cfg.max_digits)
        t1 = _make_div_scratchpad(a, b)
        t2 = _make_div_scratchpad(b, a) if a != 0 else None # if a is not zero, generate a swapped pair as well
        return t1, t2
    else:
        raise ValueError(f"Unknown operator: {op}")


def make_batch(tokenizer: CharTokenizer, batch_size: int, cfg: GenConfig, device: torch.device, max_pos: int) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """Create a training batch of token ids (inputs and next-token targets).

    Generates scratchpad samples: for each problem, creates both A op B and B op A
    (except division, which may return one or two to avoid division by zero).
    Pads sequences to uniform length and creates shifted input/target pairs for language modeling.

    Args:
        tokenizer: Character tokenizer for encoding
        batch_size: Number of samples in batch
        cfg: Generation configuration (max_digits, operators)
        device: Device to place tensors on
        max_pos: Maximum sequence length - sequences longer than this are truncated

    Returns:
        Tuple of (inputs, targets) where:
        - inputs: (batch_size, seq_len-1) tensor of token IDs
        - targets: (batch_size, seq_len-1) tensor of next-token IDs
    """
    samples: List[str] = []
    # For each generated (A, B), also include (B, A) to teach operand order.
    while len(samples) < batch_size:
        t1, t2 = generate_scratchpad_pair(cfg)
        samples.append(t1)
        if t2 is not None and len(samples) < batch_size:
            samples.append(t2)

    encoded = [tokenizer.encode(s) for s in samples]
    max_len_in_batch = max(len(x) for x in encoded)

    # CRITICAL: Check if scratchpad is longer than model's context
    # TODO: Meh ...
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


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention mechanism.

    Implements masked self-attention where each position can only attend to
    previous positions (causal/autoregressive). Used in decoder-only transformers.
    """

    def __init__(self, n_embd: int, n_heads: int, dropout: float):
        """Initialize attention module.

        Args:
            n_embd: Embedding dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        # Linear projections for key, query, value
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)  # output projection
        self.attn_drop = nn.Dropout(dropout)  # dropout on attention weights
        self.resid_drop = nn.Dropout(dropout)  # dropout on output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of shape (batch, seq_len, n_embd)
        """
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        H = self.n_heads
        head_dim = C // H  # dimension per attention head

        # Compute key, query, value and reshape for multi-head attention
        k = self.key(x).view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, d)
        q = self.query(x).view(B, T, H, head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, H, head_dim).transpose(1, 2)

        # Compute attention scores (scaled dot-product)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, H, T, T)
        # Apply causal mask (lower triangular) to prevent attending to future positions
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float('-inf'))  # mask future positions
        att = F.softmax(att, dim=-1)  # normalize attention weights
        att = torch.nan_to_num(att, nan=0.0)  # replace NaN with 0 (e.g., when all -inf)
        att = self.attn_drop(att)
        # Apply attention weights to values
        y = att @ v  # (B, H, T, d)
        # Concatenate heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """Transformer block with attention and feedforward layers.

    Implements the standard transformer block: LayerNorm -> Attention -> Add,
    followed by LayerNorm -> MLP -> Add (residual connections).
    """

    def __init__(self, n_embd: int, n_heads: int, dropout: float, mlp_mult: int = 4):
        """Initialize transformer block.

        Args:
            n_embd: Embedding dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            mlp_mult: MLP hidden dimension multiplier (default 4x embedding dim)
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)  # layer norm before attention
        self.attn = CausalSelfAttention(n_embd, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embd)  # layer norm before MLP
        # MLP: 2-layer feedforward network with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_mult * n_embd),  # expand
            nn.GELU(),  # non-linearity
            nn.Linear(mlp_mult * n_embd, n_embd),  # project back
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, n_embd)

        Returns:
            Output tensor of same shape
        """
        x = x + self.attn(self.ln1(x))  # attention with residual
        x = x + self.mlp(self.ln2(x))  # MLP with residual
        return x


class TinyGPT(nn.Module):
    """Tiny GPT-style decoder-only transformer for arithmetic.

    A minimal implementation of a transformer language model similar to GPT.
    Uses token + position embeddings, stacked transformer blocks, and a
    language modeling head.
    """

    def __init__(self, vocab_size: int, n_embd: int = 128, n_layer: int = 2, n_head: int = 4, dropout: float = 0.1,
                 max_pos: int = 512):
        """Initialize the model.

        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            n_embd: Embedding dimension
            n_layer: Number of transformer blocks
            n_head: Number of attention heads per block
            dropout: Dropout probability
            max_pos: Maximum sequence length (context window size)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # token embeddings
        self.pos_emb = nn.Embedding(max_pos, n_embd)  # positional embeddings
        self.drop = nn.Dropout(dropout)  # input dropout
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # language modeling head
        self.max_pos = max_pos  # store for saving/loading

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass: convert token indices to logits.

        Args:
            idx: Token indices of shape (batch, seq_len)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        B, T = idx.size()
        if T >= self.max_pos:
            # This should not happen if make_batch truncates, but as a safeguard
            idx = idx[:, :self.max_pos]
            T = idx.size(1)

        # Create position indices (0, 1, 2, ..., T-1)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        try:
            tok_embs = self.tok_emb(idx)  # (B, T, n_embd)
            pos_embs = self.pos_emb(pos)  # (B, T, n_embd)
        except IndexError as e:
            print(
                f"Error during embedding lookup. T={T}, max_pos={self.max_pos}, idx.shape={idx.shape}, pos.shape={pos.shape}")
            print(f"Max index in idx: {idx.max()}, Vocab size: {self.vocab_size}")
            raise e

        # Combine token and position embeddings
        x = tok_embs + pos_embs
        x = self.drop(x)
        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)  # final layer norm
        logits = self.head(x)  # project to vocabulary
        return logits


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Compute cross-entropy loss for language modeling.

    Args:
        logits: Model predictions of shape (batch, seq_len, vocab_size)
        targets: Target token IDs of shape (batch, seq_len)
        pad_id: Padding token ID to ignore in loss computation

    Returns:
        Scalar loss value
    """
    B, T, V = logits.shape
    # Flatten to (batch*seq_len, vocab_size) and (batch*seq_len,)
    loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T), ignore_index=pad_id)
    return loss


@torch.no_grad()
def generate(model: TinyGPT, tokenizer: CharTokenizer, prompt: str, max_new_tokens: int = 128,
             device: torch.device | None = None) -> str:
    """Greedy decoding from the model given a string prompt.

    Generates tokens one at a time using greedy decoding (argmax) until
    newline is encountered or max_new_tokens is reached.

    Args:
        model: The trained TinyGPT model
        tokenizer: Character tokenizer for encoding/decoding
        prompt: Input string to continue from (e.g., "12+3=")
        max_new_tokens: Maximum number of tokens to generate (default 128)
        device: Device to run on (inferred from model if None)

    Returns:
        Complete generated string including prompt
    """
    device = device or next(model.parameters()).device
    model.eval()
    text = prompt
    ids = tokenizer.encode(text)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    max_pos = model.max_pos

    for _ in range(max_new_tokens):
        # Truncate input sequence if it exceeds max_pos (sliding window)
        x_cond = x if x.size(1) <= max_pos else x[:, -max_pos:]

        if x_cond.size(1) == 0: break  # Should not happen

        logits = model(x_cond)  # get predictions
        next_logits = logits[:, -1, :]  # take last position
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # greedy selection
        x = torch.cat([x, next_id], dim=1)  # append to sequence
        ch = tokenizer.itos[next_id.item()]  # decode token
        if ch == '\n':  # stop at newline
            break
    out = tokenizer.decode(x[0].tolist())
    return out


def pick_device(device: str) -> torch.device:
    """Select compute device (CPU, CUDA, or MPS).

    Args:
        device: Device string ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        torch.device object for the selected/available device
    """
    # TODO: add the special case of TPU
    if device == 'auto':
        # Auto-detect best available device
        if torch.cuda.is_available():
            return torch.device('cuda')  # NVIDIA GPU
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')  # Apple Silicon GPU
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


def _count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _format_num(n: int) -> str:
    """Format large numbers with K/M/B suffixes.

    Args:
        n: Number to format

    Returns:
        Formatted string (e.g., "1.23M", "456K", "7.89B")
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def print_model_summary(model: nn.Module):
    """Print a formatted summary of model architecture and parameter count.

    Displays model name, hyperparameters (vocab_size, n_embd, n_layer, etc.),
    total and trainable parameters, and approximate memory size.

    Args:
        model: PyTorch model to summarize (expected to be TinyGPT or similar)
    """
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
        ckpt_path: str = 'math_llm_scratchpad_model.pt',
        log_every: int = 100,
        eval_samples: int = 5000,
        ops: str = "+-*/",
        op_probs: List[float] | None = None, # TODO: remove op prob stuff
        max_pos: int = 512,
        use_amp: bool = False,
):
    """Train the model using curriculum learning with scratchpad supervision.

    Implements a two-stage curriculum to prevent catastrophic forgetting:
    - Stage 1 (20% of steps): Train on 1-digit problems only
    - Stage 2 (80% of steps): Train on mixed 1 to max_digits problems

    This approach teaches algorithmic patterns progressively and maintains
    performance on simpler problems while learning harder ones.

    Args:
        steps: Total number of training iterations
        batch_size: Number of samples per training batch
        max_digits: Maximum number of digits in final curriculum stage
        lr: Learning rate for AdamW optimizer
        n_embd: Embedding dimension size
        n_layer: Number of transformer blocks
        n_head: Number of attention heads per block
        dropout: Dropout probability for regularization
        device: Device to train on ('auto', 'cpu', 'cuda', 'mps')
        ckpt_path: Path to save/load model checkpoint
        log_every: Print training progress every N steps
        eval_samples: Number of test samples for final evaluation (0 to skip)
        ops: String of operators to train on (e.g., '+-*/')
        op_probs: Optional list of probabilities for each operator (must match ops length)
        max_pos: Maximum sequence length (context window size)
        use_amp: Enable automatic mixed precision training (FP16) for faster training
    """
    # TODO: remove it ?
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
    scaler = None
    autocast_context = None
    if use_amp and dev.type in ['cuda', 'mps']:
        if dev.type == 'cuda':
            print("Using torch.cuda.amp for mixed precision training (float16)")
            scaler = torch.cuda.amp.GradScaler()
            autocast_context = torch.cuda.amp.autocast
        elif dev.type == 'mps':
            print("Using torch.amp for mixed precision training (float16 on MPS)")
            # MPS doesn't use a scaler, but autocast is available
            scaler = None
            autocast_context = lambda: torch.amp.autocast(device_type=dev.type, dtype=torch.float16)
    elif use_amp:
        print(f"Warning: AMP requested but not supported on device '{dev.type}'. Falling back to full precision.")
        use_amp = False

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

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
    for i, (stage_max_digits, stage_n_steps, is_mixed) in enumerate(
            zip(curriculum_stages, stage_steps, stage_is_mixed)):
        end_step = current_step + stage_n_steps
        if is_mixed:
            print(
                f"  Stage {i + 1}/{len(curriculum_stages)}: mixed (1 to {stage_max_digits} digits) (Steps {current_step + 1} - {end_step})")
        else:
            print(
                f"  Stage {i + 1}/{len(curriculum_stages)}: {stage_max_digits}-digit only (Steps {current_step + 1} - {end_step})")
        current_step = end_step
    print("=======================")

    t0 = time.time()
    ema_loss = None
    total_step_counter = 0
    _printed_log_legend = False

    # Track evaluation results across stages to detect catastrophic forgetting
    stage_eval_results = []

    for stage_idx, (stage_max_digits, stage_n_steps, is_mixed) in enumerate(
            zip(curriculum_stages, stage_steps, stage_is_mixed)):
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

            cfg = GenConfig(max_digits=batch_max_digits, ops=ops, op_probs=op_probs) # TODO: remove op prob stuff
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
                    test_cfg = GenConfig(max_digits=batch_max_digits, ops=ops, op_probs=op_probs) # TODO: remove op prob stuff
                    prompt = generate_sample_prompt(test_cfg)
                    out = generate(model, tokenizer, prompt, max_new_tokens=max_pos, device=dev)
                    # One-time legend to make logs self-explanatory
                    if not _printed_log_legend:
                        print("  --- Log legend ---")
                        #print("  [IN]  input prompt given to the model")
                        print("  [OUT] model's generated scratchpad/text")
                        print("  [GT]  ground-truth scratchpad (what model should produce)")
                        _printed_log_legend = True

                    #print(f"  [IN]  {prompt!r}")
                    print(f"  [OUT] {out!r}")

                    # Generate ground truth for comparison
                    m = re.match(r"^\s*(\d+)([+\-*/])(\d+)=", prompt)
                    if m:
                        a_str, op_s, b_str = m.group(1), m.group(2), m.group(3)
                        a_val = int(a_str)
                        b_val = int(b_str)

                        # Generate GT using the operands from the prompt
                        gt_text = ""
                        if op_s == '+':
                            gt_text = _make_add_scratchpad(a_val, b_val, batch_max_digits).strip()
                        elif op_s == '-':
                            gt_text = _make_sub_scratchpad(a_val, b_val, batch_max_digits).strip()
                        elif op_s == '*':
                            gt_text = _make_mul_scratchpad(a_val, b_val).strip()
                        elif op_s == '/':
                            gt_text = _make_div_scratchpad(a_val, b_val).strip()
                        else:
                            gt_text = "(unknown operator)"

                        actual_result = _ground_truth(a_val, b_val, op_s)
                        print(f"  [GT]  {gt_text!r}")
                        #print(f"  [NOTE] {a_val}{op_s}{b_val}={actual_result}")

        # Run mini evaluation after each stage (20 steps) to check for catastrophic forgetting
        print(f"\n=== Stage {stage_idx + 1} Evaluation (20 samples per difficulty) ===")
        stage_results = {}
        # Evaluate on all difficulty levels seen so far (1-digit up to current stage)
        for eval_digits in range(1, min(stage_max_digits, max_digits) + 1):
            eval_cfg = GenConfig(max_digits=eval_digits, ops=ops)
            stats = evaluate(model, tokenizer, eval_cfg, n_samples=20, device=dev, max_pos=max_pos, verbose=False)
            overall = stats['overall']
            overall_acc = 100.0 * overall['correct'] / max(1, overall['total'])
            stage_results[eval_digits] = overall_acc
            print(f"  {eval_digits}-digit: {overall['correct']}/{overall['total']} = {overall_acc:.2f}%")

        stage_eval_results.append({
            'stage': stage_idx + 1,
            'stage_max_digits': stage_max_digits,
            'is_mixed': is_mixed,
            'results': stage_results
        })
        print()

    # Display catastrophic forgetting analysis
    print("\n=== Catastrophic Forgetting Analysis ===")
    print("Accuracy across stages for each difficulty level:")
    print()
    # Header
    header = "Difficulty |"
    for stage_info in stage_eval_results:
        stage_desc = f"Stage {stage_info['stage']}"
        header += f" {stage_desc:^12} |"
    print(header)
    print("-" * len(header))

    # Track all difficulty levels evaluated
    all_difficulties = set()
    for stage_info in stage_eval_results:
        all_difficulties.update(stage_info['results'].keys())

    # Print results for each difficulty level
    for difficulty in sorted(all_difficulties):
        row = f"{difficulty}-digit    |"
        for stage_info in stage_eval_results:
            if difficulty in stage_info['results']:
                acc = stage_info['results'][difficulty]
                row += f" {acc:>10.2f}% |"
            else:
                row += "      -      |"
        print(row)
    print()

    # Detect catastrophic forgetting
    # TODO: too verbose, clean this stuff
    if len(stage_eval_results) > 1:
        print("Catastrophic forgetting warnings:")
        found_forgetting = False
        for difficulty in sorted(all_difficulties):
            max_acc = -1
            max_stage = -1
            for stage_info in stage_eval_results:
                if difficulty in stage_info['results']:
                    acc = stage_info['results'][difficulty]
                    if acc > max_acc:
                        max_acc = acc
                        max_stage = stage_info['stage']

            # Check if final stage has significantly lower accuracy than max
            final_stage = stage_eval_results[-1]
            if difficulty in final_stage['results']:
                final_acc = final_stage['results'][difficulty]
                drop = max_acc - final_acc
                if drop > 10.0:  # More than 10% drop
                    print(
                        f"  ⚠️  {difficulty}-digit: Dropped {drop:.1f}% from stage {max_stage} ({max_acc:.1f}%) to final ({final_acc:.1f}%)")
                    found_forgetting = True

        if not found_forgetting:
            print("  ✓ No significant catastrophic forgetting detected!")
    print("=====================================\n")

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

        # Clean up old evaluation files before starting
        import glob
        old_eval_files = glob.glob(ckpt_path.replace('.pt', '_eval_*.txt'))
        for old_file in old_eval_files:
            try:
                os.remove(old_file)
            except Exception:
                pass  # Ignore errors if file doesn't exist or can't be deleted

        # Evaluate on each curriculum stage (1-digit, 2-digit, 3-digit, etc.)
        final_eval_results = {}
        for eval_digits in range(1, max_digits + 1):
            print(f"Evaluating {eval_digits}-digit problems ({eval_samples} samples)...", end=" ", flush=True)
            eval_cfg = GenConfig(max_digits=eval_digits, ops=ops)
            # Save outputs to file
            save_path = ckpt_path.replace('.pt', f'_eval_{eval_digits}digit.txt')
            stats = evaluate(model, tokenizer, eval_cfg, n_samples=eval_samples, device=dev, max_pos=max_pos,
                             verbose=False, save_outputs=save_path)
            overall = stats['overall']
            overall_acc = 100.0 * overall['correct'] / max(1, overall['total'])
            final_eval_results[eval_digits] = overall_acc
            print(f"{overall['correct']}/{overall['total']} = {overall_acc:.2f}%")
            for op in ops:
                c = stats['per_op'][op]
                acc = 100.0 * c['correct'] / max(1, c['total'])
                print(f"  {op}: {c['correct']}/{c['total']} = {acc:.2f}%")
            print(f"  Saved outputs to: {save_path}")

        # Extrapolation test: Evaluate on 4-digit problems (minimal sample size)
        extrapolation_samples = max(20, eval_samples // 10)  # Use 10% of eval_samples, minimum 20
        print(f"\nExtrapolation: 4-digit problems ({extrapolation_samples} samples)...", end=" ", flush=True)
        eval_cfg_4d = GenConfig(max_digits=4, ops=ops)
        save_path_4d = ckpt_path.replace('.pt', '_eval_4digit_extrapolation.txt')
        stats_4d = evaluate(model, tokenizer, eval_cfg_4d, n_samples=extrapolation_samples, device=dev, max_pos=max_pos,
                            verbose=False, save_outputs=save_path_4d)
        overall_4d = stats_4d['overall']
        overall_acc_4d = 100.0 * overall_4d['correct'] / max(1, overall_4d['total'])
        print(f"{overall_4d['correct']}/{overall_4d['total']} = {overall_acc_4d:.2f}%")
        for op in ops:
            c = stats_4d['per_op'][op]
            acc = 100.0 * c['correct'] / max(1, c['total'])
            print(f"  {op}: {c['correct']}/{c['total']} = {acc:.2f}%")
        print(f"  Saved outputs to: {save_path_4d}")

        print("\n===============================")

        # Final catastrophic forgetting analysis comparing stage evals to final eval
        if len(stage_eval_results) > 0:
            print("\n=== Final Catastrophic Forgetting Analysis ===")
            print("Comparing peak performance during training vs. final evaluation:")
            print()
            print(f"{'Difficulty':<12} | {'Peak (Training)':<20} | {'Final Eval':<15} | {'Change':<10}")
            print("-" * 70)

            found_forgetting = False
            for difficulty in sorted(final_eval_results.keys()):
                # Find peak accuracy during training stages for this difficulty
                peak_acc = -1
                peak_stage = -1
                for stage_info in stage_eval_results:
                    if difficulty in stage_info['results']:
                        acc = stage_info['results'][difficulty]
                        if acc > peak_acc:
                            peak_acc = acc
                            peak_stage = stage_info['stage']

                final_acc = final_eval_results[difficulty]

                if peak_acc >= 0:
                    change = final_acc - peak_acc
                    change_str = f"{change:+.1f}%"
                    if change < -10.0:
                        change_str += " ⚠️"
                        found_forgetting = True
                    elif change > 5.0:
                        change_str += " ✓"

                    print(
                        f"{difficulty}-digit{'':<6} | Stage {peak_stage}: {peak_acc:>5.1f}%{'':<8} | {final_acc:>6.1f}%{'':<7} | {change_str}")
                else:
                    print(f"{difficulty}-digit{'':<6} | {'N/A':<20} | {final_acc:>6.1f}%{'':<7} | {'N/A'}")

            print()
            if found_forgetting:
                print("⚠️  Significant performance drops detected (>10%).")
            else:
                print("✓ Model maintained or improved performance on all difficulty levels!")
            print("==============================================\n")


def generate_sample_prompt(cfg: GenConfig) -> str:
    """Generate a random arithmetic prompt for testing/demonstration.

    Creates a prompt in the format "a op b=" matching the training data format.

    Args:
        cfg: Generation configuration (max_digits, operators)

    Returns:
        String prompt like "12+34=" or "144/12="
    """
    # Generate prompt matching training data format
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

    Args:
        ckpt_path: Path to checkpoint file (.pt)
        device: Device string ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Tuple of (model, tokenizer, device)

    Raises:
        RuntimeError: If checkpoint vocab_size doesn't match tokenizer
    """
    dev = pick_device(device)
    tokenizer = CharTokenizer()
    ckpt = torch.load(ckpt_path, map_location=dev)

    # Prefer checkpoint's saved vocab_size to ensure shapes match; fall back to tokenizer size.
    vocab_size = int(ckpt.get('vocab_size', getattr(tokenizer, 'vocab_size', 50)))
    if vocab_size != tokenizer.vocab_size:
        raise RuntimeError(
            f"Checkpoint vocab_size ({vocab_size}) != current tokenizer vocab_size ({tokenizer.vocab_size}). Incompatible tokenizer/vocab; please retrain or use a matching checkpoint.")

    # Reconstruct model with saved hyperparameters
    model = TinyGPT(
        vocab_size=vocab_size,
        n_embd=ckpt.get('n_embd', 128),
        n_layer=ckpt.get('n_layer', 2),
        n_head=ckpt.get('n_head', 4),
        dropout=ckpt.get('dropout', 0.1),
        max_pos=ckpt.get('max_pos', 512),  # load max_pos from checkpoint
    ).to(dev)
    model.load_state_dict(ckpt['model_state_dict'])  # load trained weights
    model.eval()  # set to evaluation mode
    return model, tokenizer, dev


def _sample_operands_for_op(op: str, max_digits: int) -> Tuple[int, int]:
    """Generate operands for a given operator.

    Args:
        op: Operator character ('+', '-', '*', '/')
        max_digits: Maximum digits for operands

    Returns:
        Tuple of (operand1, operand2)
    """
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
    """Compute ground truth result for arithmetic operation.

    Args:
        a: First operand
        b: Second operand
        op: Operator character ('+', '-', '*', '/')

    Returns:
        Integer result of the operation
    """
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        return a // b if b != 0 else 0  # integer division
    raise ValueError(f"Unknown operator: {op}")


def _extract_result_from_text(text: str) -> str:
    """Extract the model-predicted result from a scratchpad string.

    Looks for the "F:..." pattern where F = Final answer marker.

    Args:
        text: Generated text containing scratchpad
            Example: "85+47=[C1:5+7=12;W:2;K:1;C2:8+4+1=13;W:3;F:132]\n"

    Returns:
        Extracted answer as string (e.g., "132"), or empty string if not found
    """
    # Primary method: regex to capture the last occurrence of F:<int> (allow optional leading minus)
    matches = re.findall(r"F:(-?\d+)", text)
    if matches:
        return matches[-1]  # take last match

    # Fallback: if no "F:", try to find "...=...[...]\n"
    # and extract the ... part. This is for robustness with malformed outputs.
    # But the "F:" marker is the primary method.

    # Fallback 2: old extraction method (for debugging simple lookups like division)
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
             max_pos: int, verbose: bool = True, save_outputs: str | None = None) -> dict:
    """Run a synthetic test set and compute per-op and overall accuracy.

    Generates random arithmetic problems, evaluates model predictions,
    and computes accuracy statistics per operator and overall.

    Args:
        model: Trained TinyGPT model
        tokenizer: Character tokenizer
        cfg: Generation config (operators, max_digits)
        n_samples: Number of test samples to evaluate
        device: Device to run on
        max_pos: Maximum position for generation
        verbose: If True, print progress during evaluation
        save_outputs: If provided, save outputs to this file path

    Returns:
        Dictionary with 'per_op' (per-operator stats) and 'overall' stats
    """
    counts = {op: {'correct': 0, 'total': 0} for op in cfg.ops}
    log_interval = max(1, n_samples // 10)  # log progress every 10%

    # Collect outputs if saving
    outputs_to_save = [] if save_outputs else None

    for i in range(n_samples):
        if verbose and i % log_interval == 0 and i > 0:
            print(f"  eval sample {i}/{n_samples}")

        op = random.choice(cfg.ops)  # pick random operator
        a, b = _sample_operands_for_op(op, cfg.max_digits)  # generate operands

        # Create the prompt matching training data
        prompt = f"{a}{op}{b}="
        expected = str(_ground_truth(a, b, op))

        # Generate the full scratchpad from model
        out = generate(model, tokenizer, prompt, max_new_tokens=max_pos, device=device)
        # Extract the final answer from the scratchpad
        pred = _extract_result_from_text(out)

        counts[op]['total'] += 1
        is_correct = pred == expected
        if is_correct:
            counts[op]['correct'] += 1

        # Save output if requested
        if outputs_to_save is not None:
            # Generate ground truth using the operands
            if op == '+':
                gt_text = _make_add_scratchpad(a, b, cfg.max_digits).strip()
            elif op == '-':
                gt_text = _make_sub_scratchpad(a, b, cfg.max_digits).strip()
            elif op == '*':
                gt_text = _make_mul_scratchpad(a, b).strip()
            elif op == '/':
                gt_text = _make_div_scratchpad(a, b).strip()
            else:
                gt_text = f"{a}{op}{b}=?"

            outputs_to_save.append({
                'prompt': prompt,
                'output': out,
                'ground_truth': gt_text,
                'predicted': pred,
                'expected': expected,
                'correct': is_correct,
                'calculation': f"{a}{op}{b}={expected}"
            })

    # Aggregate overall statistics
    total_correct = sum(v['correct'] for v in counts.values())
    total = sum(v['total'] for v in counts.values())

    # Save outputs to file if requested
    if save_outputs and outputs_to_save:
        with open(save_outputs, 'w', encoding='utf-8') as f:
            for entry in outputs_to_save:
                f.write(f"[PROMPT] {entry['prompt']!r}\n")
                f.write(f"[OUT]    {entry['output']!r}\n")
                f.write(f"[GT]     {entry['ground_truth']!r}\n")
                f.write(f"[CALC]   {entry['calculation']}\n")
                f.write(
                    f"[PRED]   {entry['predicted']!r} (Expected: {entry['expected']!r}) - {'CORRECT' if entry['correct'] else 'WRONG'}\n")
                f.write(f"\n" + "-" * 80 + "\n")

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
    # TODO: add TPU
    p_train.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'mps', 'auto'],
                         help="Device to train on: cpu, cuda (NVIDIA GPU), mps (Apple Silicon), or auto (default: auto)")
    p_train.add_argument('--ckpt', type=str, default='math_llm_scratchpad_model-010.pt',
                         help="Checkpoint file path for saving/loading model (default: math_llm_scratchpad_model-010.pt)")
    p_train.add_argument('--log-every', type=int, default=100,
                         help="Print training progress every N steps (default: 500)")
    p_train.add_argument('--eval-samples', type=int, default=500,
                         help='Number of synthetic test samples for evaluation; set to 0 to skip evaluation (default: 500)')
    p_train.add_argument('--ops', type=str, default='+-*/',
                         help="String of operators to train on, e.g., '+-*/' for all four operations (default: +-*/)")
    # TODO: remove op prob stuff
    p_train.add_argument('--op-probs', type=str, default=None,
                         help="Comma-separated operator probabilities (e.g., '0.25,0.25,0.25,0.25'). Not recommended; curriculum is better (default: None)")
    p_train.add_argument('--max-pos', type=int, default=512,
                         help="Maximum sequence length for positional embeddings (context window size) (default: 512)")
    p_train.add_argument('--use-amp', action='store_true',
                         help="Enable automatic mixed precision training using FP16/BF16 for faster training (default: False)")

    p_demo = sub.add_parser('demo', help='Run generation on a prompt like "12+3="')
    p_demo.add_argument('--prompt', type=str, required=True,
                        help='Arithmetic prompt to evaluate, e.g., "12+3=" or "144/12="')
    p_demo.add_argument('--ckpt', type=str, default='math_llm_scratchpad_model-010.pt',
                        help="Path to checkpoint file to load trained model from (default: math_llm_scratchpad_model-010.pt)")
    # TODO: add TPU
    p_demo.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'mps', 'auto'],
                        help="Device to run inference on: cpu, cuda (NVIDIA GPU), mps (Apple Silicon), or auto (default: auto)")
    p_demo.add_argument('--max-new-tokens', type=int, default=512,
                        help="Maximum number of tokens to generate in response. Should be >= model's max_pos (default: 512)")

    args = parser.parse_args()

    if args.cmd == 'train':
        probs_list = None
        # TODO: remove op prob stuff
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
