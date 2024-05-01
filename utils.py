import torch
import torch.nn as nn
import loralib as lora
import time
from torch.nn import functional as F

batch_size = 4
vocab_size=96
n_embed=384
n_embd=384
block_size=256
n_layer=6
n_head=6
dropout=0.0
low_rank=True
# chars = sorted(list(set(text)))
chars=['\n', ' ', '!', '"', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\xad', '´', 'é', 'ñ', '\u200a', '\u200b', '–', '—', '‘', '’', '“', '”', '…', '🎓']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
stoi = { ch:i for i, ch in enumerate(chars) }
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[x] for x in l])

class MoeLayer(nn.Module):
    def __init__(self, experts, gate, k=1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.k = k

    def forward(self, inputs: torch.Tensor):
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.gate(inputs_squashed)
        weights, selected_experts = torch.topk(
            gate_logits, self.k
        )
        weights = nn.functional.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(inputs)
        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs_squashed[batch_idx]
            )
        return results.view_as(inputs)
import loralib as lora

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = lora.Linear(n_embed, head_size, r=16)
        # nn.Linear(n_embed, head_size, bias = False)
        self.query = lora.Linear(n_embed, head_size, r=16)
        # nn.Linear(n_embed, head_size, bias = False)
        self.value = lora.Linear(n_embed, head_size, r=16)
        # nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MulitHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = lora.Linear(n_embed, n_embed, r=16)
        #  nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x =  torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(x))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            lora.Linear(n_embed, 4* n_embed, r=16),
            nn.ReLU(),
            lora.Linear(4 * n_embed, n_embed, r=16),
         nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head, num_experts=4):
        super().__init__()
        self.sa_head= MulitHeadAttention(n_head, n_embed//n_head)
        self.ffw = MoeLayer(
            experts=[FeedForward(n_embed) for _ in range(num_experts)],
            gate=lora.Linear(n_embed, num_experts, r=16),
        )

#       self.ffw=  FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x+self.ffw(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed, device=device)
        self.position_embedding_table = nn.Embedding(block_size, n_embed, device=device)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.lm_head = lora.Linear(n_embed, vocab_size, r=16)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T).to(device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokes):
        for _ in range(max_new_tokes):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx
def get_model_output(prompt):
    loaded_model = Transformer()
    loaded_model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')))
    start_time = time.time()
    d = prompt
    x = torch.tensor(encode(d), dtype = torch.long,device=device).unsqueeze(0)
    gen_out = decode(loaded_model.generate(x, max_new_tokes=500)[0].tolist())
    end_time = time.time()
    elapsed_time = end_time - start_time
    return gen_out, elapsed_time