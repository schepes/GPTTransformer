import torch
import torch.nn as nn
from torch.nn import functional as F

# configuration
batch_sz = 64
context_length = 256
max_iterations = 5000
eval_frequency = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluation_iters = 200
embedding_dim = 384
num_heads = 6
num_layers = 6
dropout_rate = 0.2
# ------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    content = f.read()

unique_chars = sorted(list(set(content)))
vocab_size = len(unique_chars)
char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}
encode_str = lambda s: [char_to_idx[c] for c in s]
decode_str = lambda l: ''.join([idx_to_char[i] for i in l])

data = torch.tensor(encode_str(content), dtype=torch.long)
train_data_len = int(0.9 * len(data))
train_data = data[:train_data_len]
validation_data = data[train_data_len:]

def get_data_batch(split):
    dataset = train_data if split == 'train' else validation_data
    indices = torch.randint(len(dataset) - context_length, (batch_sz,))
    input_data = torch.stack([dataset[i:i + context_length] for i in indices])
    target_data = torch.stack([dataset[i + 1:i + context_length + 1] for i in indices])
    input_data, target_data = input_data.to(device), target_data.to(device)
    return input_data, target_data

@torch.no_grad()
def compute_loss():
    results = {}
    model.eval()
    for split_type in ['train', 'val']:
        loss_values = torch.zeros(evaluation_iters)
        for k in range(evaluation_iters):
            X, Y = get_data_batch(split_type)
            logit_vals, loss_val = model(X, Y)
            loss_values[k] = loss_val.item()
        results[split_type] = loss_values.mean()
    model.train()
    return results

class AttentionHead(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.key_linear = nn.Linear(embedding_dim, head_dim, bias=False)
        self.query_linear = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value_linear = nn.Linear(embedding_dim, head_dim, bias=False)
        self.register_buffer('tril_matrix', torch.tril(torch.ones(context_length, context_length)))

        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key_linear(x)
        q = self.query_linear(x)
        attn_weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        attn_weights = attn_weights.masked_fill(self.tril_matrix[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        v = self.value_linear(x)
        output = attn_weights @ v
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.attention_heads = nn.ModuleList([AttentionHead(head_dim) for _ in range(num_heads)])
        self.projection = nn.Linear(head_dim * num_heads, embedding_dim)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x):
        concat_heads = torch.cat([h(x) for h in self.attention_heads], dim=-1)
        output = self.dropout_layer(self.projection(concat_heads))
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        head_dim = embedding_dim // num_heads
        self.self_attention = MultiHeadSelfAttention(num_heads, head_dim)
        self.feed_forward = PositionwiseFeedForward(embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(context_length, embedding_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.language_model_head = nn.Linear(embedding_dim, vocab_size)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embeddings(idx)
        pos_emb = self.position_embeddings(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.language_model_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = GPT()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iteration in range(max_iterations):
    if iteration % eval_frequency == 0 or iteration == max_iterations - 1:
        losses = compute_loss()
        print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    x_batch, y_batch = get_data_batch('train')

    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode_str(model.generate(context, max_new_tokens=500)[0].tolist()))


