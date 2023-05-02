import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    torch.manual_seed(1337)

    config = {
        'batch_num': 32, 'sequence_len': 8, 'max_steps': 3000, 'eval_step': 300,
        'learning_rate': 1e-2, 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'eval_steps': 200
    }

    text_data = read_file('input.txt')
    unique_chars, char_to_idx, idx_to_char = prepare_vocab(text_data)
    train_tensor, val_tensor = prepare_tensors(text_data, char_to_idx, 0.9)

    model = SimpleBigramModel(len(unique_chars)).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    for step in range(config['max_steps']):
        if step % config['eval_step'] == 0:
            loss_values = calculate_loss(model, train_tensor, val_tensor, config)
            print(f"step {step}: train loss {loss_values['train']:.4f}, val loss {loss_values['val']:.4f}")

        x_batch, y_batch = generate_batch(train_tensor, config['sequence_len'], config['batch_num'], config['device'])
        train_step(model, optimizer, x_batch, y_batch)

    init_context = torch.zeros((1, 1), dtype=torch.long, device=config['device'])
    generated_seq = model.generate_seq(init_context, max_new_tokens=500)
    print(decode_str(generated_seq[0].tolist(), idx_to_char))


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def prepare_vocab(text_data):
    unique_chars = sorted(list(set(text_data)))
    char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
    idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}
    return unique_chars, char_to_idx, idx_to_char


def prepare_tensors(text_data, char_to_idx, train_split_ratio):
    encode_str = lambda s: [char_to_idx[c] for c in s]
    data_tensor = torch.tensor(encode_str(text_data), dtype=torch.long)
    n = int(train_split_ratio * len(data_tensor))
    return data_tensor[:n], data_tensor[n:]


def generate_batch(data_tensor, sequence_len, batch_num, device):
    index = torch.randint(len(data_tensor) - sequence_len, (batch_num,))
    x = torch.stack([data_tensor[i:i + sequence_len] for i in index])
    y = torch.stack([data_tensor[i + 1:i + sequence_len + 1] for i in index])
    return x.to(device), y.to(device)


@torch.no_grad()
def calculate_loss(model, train_tensor, val_tensor, config):
    model.eval()
    loss_output = {}
    for split in ['train', 'val']:
        data_split = train_tensor if split == 'train' else val_tensor
        losses = torch.zeros(config['eval_steps'])
        for k in range(config['eval_steps']):
            x, y = generate_batch(data_split, config['sequence_len'], config['batch_num'], config['device'])
            logits, loss = model(x, y)
            losses[k] = loss.item()
        loss_output[split] = losses.mean()
    model.train()
    return loss_output


def train_step(model, optimizer, x_batch, y_batch):
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

def decode_str(encoded_str, idx_to_char):
    return ''.join([idx_to_char[i] for i in encoded_str])

class SimpleBigramModel(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_len, vocab_len)

    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate_seq(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


if __name__ == '__main__':
    main()
