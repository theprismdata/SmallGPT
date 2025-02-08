import torch
import torch.nn as nn
import torch.nn.functional as F

with open("naver_news_extracted_text.txt", "r", encoding="utf-8") as f:
    data = f.readlines()
ko_text = "".join(data)
ko_chars = sorted(list(set((ko_text))))
ko_vocab_size = len(ko_chars)

character_to_ids = {char:i for i, char in enumerate(ko_chars)}
ids_to_character = {i:char for i, char in enumerate(ko_chars)}
token_encode = lambda s:[character_to_ids[c] for c in s]
token_decode = lambda l: "".join([ids_to_character[i] for i in l])
tokenized_data = torch.tensor(token_encode(ko_text), dtype=torch.long)

n = int(0.9 * len(tokenized_data))
train_dataset = tokenized_data[:n]
test_dataset = tokenized_data[n:]

batch_size = 32
block_size = 8
max_iteration = 50000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iteration = 200
n_embed = 32
n_head = 4
n_layer = 4
dropout = 0.1

def batch_function(mode):
    dataset = train_dataset if mode == "train" else test_dataset
    idx = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[index:index+block_size] for index in idx])
    y = torch.stack([dataset[index+1:index+block_size+1] for index in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def compute_loss_metrics():
    out = {}
    model.eval()
    for mode in ["train", "eval"]:
        losses = torch.zeros(eval_iteration)
        for k in range(eval_iteration):
            inputs, targets = batch_function(mode)
            logits, loss = model(inputs, targets)
            losses[k] = loss.item()
        out[mode] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, inputs):
        batch_size, sequence_length, embedding_dim = inputs.shape
        keys = self.key(inputs)
        queries = self.query(inputs)
        weights = queries @ keys.transpose(-2, -1) * (embedding_dim ** -0.5)
        weights = weights.masked_fill(self.tril[:sequence_length, :sequence_length] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        values = self.value(inputs)
        output = weights @ values
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self,inputs):
        return torch.cat([head(inputs) for head in self.heads], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, input_tensor):
        return self.layer(input_tensor)


class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.attention = MultiHeadAttention(n_heads, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, input_tensor):
        input_tensor = input_tensor + self.attention(self.layer_norm1(input_tensor))
        input_tensor = input_tensor + self.feed_forward(self.layer_norm2(input_tensor))
        return input_tensor


class Small_GPT(nn.Module):
    def __init__(self, vocab_length):
        super().__init__()
        self.embedding_token_table = nn.Embedding(vocab_length, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, 4) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_length)

    def forward(self, inputs, targets=None):
        batch, sequence = inputs.shape

        token_embed = self.embedding_token_table(inputs) # (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(sequence, device=device)) # (T, C)
        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, sequence, embed_size = logits.shape
            logits = logits.view(batch * sequence, embed_size)
            targets = targets.view(batch * sequence)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            inputs_cond = inputs[:, -block_size:]

            logits, loss = self(inputs_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_inputs = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, next_inputs), dim=1)
        return inputs

model = Small_GPT(ko_vocab_size).to(device)
def main(mode='train', model_path='transformer_model.pth'):
    if mode == 'train':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        for step in range(max_iteration):
            if step % eval_interval == 0:
                losses = compute_loss_metrics()
                print(f'step : {step}, train loss : {losses["train"]:.4f}, val loss : {losses["eval"]:.4f}')

            example_x, example_y = batch_function("train")
            logits, loss = model(example_x, example_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        # 모델 저장
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    elif mode == 'generate':
        device = torch.device('cpu')

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        input_word = "비상계엄의 위헌성을 가릴 핵심 쟁점이 뭐야?"
        input_ids = [character_to_ids[char] for char in input_word if char in character_to_ids]

        # 입력 텐서 생성
        inputs = torch.tensor([input_ids], dtype=torch.long).to(device)

        # 모델을 사용하여 텍스트 생성
        outputs = model.generate(inputs, 100)

        # 생성된 결과 디코딩
        generated_text = "".join([ids_to_character.get(i, '') for i in outputs[0].tolist()])

        print("-----------------------------------------------")
        print("Generated Text: ", generated_text)

if __name__ == "__main__":
    # 학습 모드로 실행
    # main(mode='train')

    # 텍스트 생성 모드로 실행
    main(mode='generate')



