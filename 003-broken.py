import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import deque
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# Define the mathematical operations
OPERATIONS = ['+', '-', '*', '/']
MAX_NUMBER = 100
MIN_NUMBER = 1

class MathLLM(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=64, num_heads=4, num_layers=2, ff_dim=128, max_seq_len=32):
        super(MathLLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer to predict next token
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # Embed tokens
        embedded = self.token_embedding(x) * math.sqrt(self.embed_dim)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        embedded += self.pos_encoding[positions].expand_as(embedded)
        
        # Apply dropout
        embedded = self.dropout(embedded)
        
        # Create mask for transformer (avoid attending to padding tokens)
        src_key_padding_mask = (x == 0)  # Assuming 0 is padding token
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)
        
        # Project to vocabulary space
        output = self.output_projection(transformer_out)
        
        return output

def generate_math_expression(max_depth=3):
    """Generate a random mathematical expression with integers"""
    if max_depth <= 0:
        return str(random.randint(MIN_NUMBER, MAX_NUMBER))
    
    # Randomly decide whether to create a simple number or an operation
    if random.random() < 0.3 and max_depth > 1:
        # Create a more complex expression
        op = random.choice(OPERATIONS)
        left = generate_math_expression(max_depth - 1)
        right = generate_math_expression(max_depth - 1)
        
        # Ensure division doesn't result in zero or negative numbers
        if op == '/' and right != '0':
            return f"({left} {op} {right})"
        elif op == '/':
            # If we have a zero denominator, change to another operation
            op = random.choice(['+', '-', '*'])
            return f"({left} {op} {right})"
        else:
            return f"({left} {op} {right})"
    else:
        # Generate a simple number
        return str(random.randint(MIN_NUMBER, MAX_NUMBER))

def tokenize_expression(expr):
    """Convert expression to token indices"""
    # Simple tokenizer that splits on spaces and operators
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
        elif expr[i] in '()':
            tokens.append(expr[i])
            i += 1
        elif expr[i] in '+-*/':
            tokens.append(expr[i])
            i += 1
        elif expr[i].isdigit():
            num = ''
            while i < len(expr) and expr[i].isdigit():
                num += expr[i]
                i += 1
            tokens.append(num)
        else:
            i += 1
    
    # Convert tokens to indices (simple mapping for demonstration)
    token_to_idx = {}
    idx_to_token = {}
    
    # Add special tokens
    special_tokens = ['<PAD>', '<START>', '<END>']
    for i, token in enumerate(special_tokens):
        token_to_idx[token] = i
        idx_to_token[i] = token
    
    # Add numbers and operators
    vocab_size = 1000  # We'll use a fixed vocabulary size
    for i, token in enumerate(tokens):
        if token not in token_to_idx:
            # Use a simple mapping - this is a simplified approach
            if token.isdigit():
                token_to_idx[token] = int(token) + 100  # Reserve first 100 for special tokens
            elif token in OPERATIONS:
                token_to_idx[token] = ord(token) + 200  # Use ASCII values for operators
            else:
                token_to_idx[token] = len(token_to_idx) + 300
    
    # Create reverse mapping
    for token, idx in token_to_idx.items():
        if idx not in idx_to_token:
            idx_to_token[idx] = token
    
    return [token_to_idx.get(token, 0) for token in tokens], token_to_idx, idx_to_token

def generate_training_data(num_samples=100000):
    """Generate synthetic training data"""
    expressions = []
    results = []
    
    for _ in range(num_samples):
        expr = generate_math_expression(max_depth=3)
        try:
            # Evaluate the expression
            result = eval(expr)
            if isinstance(result, (int, float)) and not math.isnan(result) and not math.isinf(result):
                expressions.append(expr)
                results.append(str(result))
        except:
            # Skip invalid expressions
            continue
    
    return expressions, results

def prepare_data(expressions, results, max_seq_len=32):
    """Prepare data for training"""
    # Simple tokenization - in a real implementation we'd use a better tokenizer
    X = []
    y = []
    
    for expr, result in zip(expressions, results):
        # Create input sequence: "expression = "
        input_seq = f"{expr} = "
        
        # Tokenize the input
        tokens = []
        i = 0
        while i < len(input_seq):
            if input_seq[i].isspace():
                i += 1
            elif input_seq[i] in '()':
                tokens.append(input_seq[i])
                i += 1
            elif input_seq[i] in '+-*/=':
                tokens.append(input_seq[i])
                i += 1
            elif input_seq[i].isdigit():
                num = ''
                while i < len(input_seq) and input_seq[i].isdigit():
                    num += input_seq[i]
                    i += 1
                tokens.append(num)
            else:
                i += 1
        
        # Convert to indices (simplified approach)
        token_to_idx = {}
        idx_to_token = {}
        
        # Add special tokens
        special_tokens = ['<PAD>', '<START>', '<END>']
        for i, token in enumerate(special_tokens):
            token_to_idx[token] = i
            idx_to_token[i] = token
        
        # Add numbers and operators
        vocab_size = 1000
        for i, token in enumerate(tokens):
            if token not in token_to_idx:
                if token.isdigit():
                    token_to_idx[token] = int(token) + 100
                elif token in OPERATIONS or token == '=':
                    token_to_idx[token] = ord(token) + 200
                else:
                    token_to_idx[token] = len(token_to_idx) + 300
        
        # Convert tokens to indices
        seq_indices = [token_to_idx.get(token, 0) for token in tokens]
        
        # Pad or truncate sequence
        if len(seq_indices) > max_seq_len - 1:
            seq_indices = seq_indices[:max_seq_len - 1]
        else:
            seq_indices.extend([0] * (max_seq_len - len(seq_indices)))
        
        X.append(seq_indices)
        
        # Create target sequence: result
        result_tokens = []
        for char in result:
            if char.isdigit():
                result_tokens.append(char)
            elif char == '.':
                result_tokens.append(char)
        
        # Convert result to indices
        result_indices = [token_to_idx.get(token, 0) for token in result_tokens]
        
        # Pad or truncate result sequence
        if len(result_indices) > max_seq_len - 1:
            result_indices = result_indices[:max_seq_len - 1]
        else:
            result_indices.extend([0] * (max_seq_len - len(result_indices)))
            
        y.append(result_indices)
    
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train_model():
    """Train the mathematical LLM"""
    print("Generating training data...")
    expressions, results = generate_training_data(num_samples=100000)
    
    print(f"Generated {len(expressions)} training examples")
    
    # Prepare data
    print("Preparing data for training...")
    X, y = prepare_data(expressions, results, max_seq_len=32)
    
    # Create model
    print("Creating model...")
    model = MathLLM(vocab_size=1000, embed_dim=64, num_heads=4, num_layers=2, ff_dim=128, max_seq_len=32)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(5):  # 5 epochs should be sufficient for demonstration
        total_loss = 0
        num_batches = 0
        
        # Simple batching - in practice, you'd want more sophisticated batching
        batch_size = 32
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Reshape for loss calculation
            loss = criterion(outputs.reshape(-1, model.vocab_size), batch_y.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/5, Average Loss: {avg_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'math_llm_model.pt')
    print("Model saved as 'math_llm_model.pt'")
    
    return model

def evaluate_model(model, expression):
    """Evaluate the model with a given expression"""
    # This is a simplified evaluation function
    print(f"Expression: {expression}")
    try:
        result = eval(expression)
        print(f"Expected result: {result}")
    except:
        print("Could not evaluate expected result")
    
    # In a real implementation, we would use the model to predict the result
    # For now, just return that we've processed it
    return "Model processed expression"

if __name__ == "__main__":
    print("Training mathematical LLM...")
    model = train_model()
    
    # Test with some examples
    test_expressions = [
        "5 + 3",
        "10 - 4",
        "6 * 7",
        "20 / 4"
    ]
    
    print("\nTesting the model:")
    for expr in test_expressions:
        evaluate_model(model, expr)
