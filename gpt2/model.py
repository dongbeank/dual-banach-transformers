"""
GPT-2 (nanoGPT) with the dual Banach regularizer L_DB.

Model definition after nanoGPT (https://github.com/karpathy/nanoGPT, MIT license), which follows
1) the official GPT-2 TensorFlow implementation released by OpenAI:
   https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
   https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Our additions (``reg=True``): per block, the attention-output norm (``CausalSelfAttention.attention_norm``),
1-step power-iteration spectral norms of the two FFN weights (``Block.spectral_norm``), the per-token
amplification factor kappa and the block term  mean_t[(||attn||^2 + ||x||^2/H) * kappa_t^2]
(``Block.forward``); ``GPT.forward`` averages over blocks and returns it as the third output.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304   # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True         # True: bias in Linears and LayerNorms, like GPT-2
    word_dropout: float = 0.0 # probability of replacing an input token by <|endoftext|> during training


class WordDropout(nn.Module):
    """Replaces each input token by the <|endoftext|> token with probability ``p`` (training only)."""

    def __init__(self, p=0.0, unk_token_id=50256):
        super().__init__()
        self.p = p
        self.unk_token_id = unk_token_id

    def forward(self, input_ids):
        if not self.training or self.p == 0.0:
            return input_ids
        mask = torch.rand_like(input_ids.float()) < self.p
        output_ids = input_ids.clone()
        output_ids[mask] = self.unk_token_id
        return output_ids


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config, reg=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.reg = reg
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def attention_norm(self, y):
        """Per-token squared norm of the attention output after the output projection, averaged over heads
        (y: (B, H, T, d) head outputs before ``c_proj``). Computed in fp32."""
        y = y.float()
        W_split = self.c_proj.weight.float().view(self.n_embd, self.n_head, -1)          # D h d
        W_square = torch.einsum('dhs,dhn->hsn', W_split, W_split)                        # h d d
        term1 = (torch.matmul(y, W_square.unsqueeze(0)) * y).sum(-1).sum(1) / (self.n_embd * self.n_head)
        bias_split = self.c_proj.bias.float() / self.n_head
        bias_W = torch.einsum('D,Dhd->hd', bias_split, W_split) / self.n_head
        term2 = 2 * torch.einsum('hd,bhtd->bt', bias_W, y) / self.n_embd
        term3 = bias_split.square().mean().view(1, 1)
        return term1 + term2 + term3                                                      # (B, T)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        attn_norm = self.attention_norm(y) if self.reg else None
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, attn_norm


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):

    def __init__(self, config, reg=False, kappa_mode='full'):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, reg=reg)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.n_head = config.n_head
        self.reg = reg
        self.kappa_mode = kappa_mode  # 'full' | 'one' (kappa = 1) | 'detach' (stop-gradient through kappa)
        self.reg_loss = None

    def init_power_iteration(self):
        """Power-iteration vectors for the spectral norms of the two FFN weights (buffers, saved in the state dict)."""
        for linear in (self.mlp.c_fc, self.mlp.c_proj):
            W = linear.weight
            linear.register_buffer('_u', F.normalize(torch.randn(W.size(0), device=W.device), dim=0))
            linear.register_buffer('_v', F.normalize(torch.randn(W.size(1), device=W.device), dim=0))

    def spectral_norm(self, linear, n_iters=1):
        """1-step power iteration (updated in training mode); the gradient flows through W only."""
        W, u, v = linear.weight, linear._u, linear._v
        if self.training:
            with torch.no_grad():
                for _ in range(n_iters):
                    v = F.normalize(torch.mv(W.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(W, v), dim=0, out=u)
        return torch.dot(u, torch.mv(W, v))

    def kappa_sq(self, x):
        """Squared amplification factor kappa_t^2 = (1 + max|gamma| * ||W1|| * ||W2|| / std(x_t))^2, in fp32."""
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x = x.float()
            gamma_max = torch.max(self.ln_2.weight.detach())
            w1norm = self.spectral_norm(self.mlp.c_fc)
            w2norm = self.spectral_norm(self.mlp.c_proj)
            return (gamma_max * w1norm * w2norm / x.std(dim=-1) + 1).pow(2)

    def forward(self, x):
        if not self.reg:
            x = x + self.attn(self.ln_1(x))[0]
            x = x + self.mlp(self.ln_2(x))
            return x
        x_norm = x.pow(2).mean(dim=-1) / self.n_head
        a, attn_norm = self.attn(self.ln_1(x))
        x = x + a
        fdual_norm = x_norm + attn_norm
        kappa_sq = self.kappa_sq(x)
        if self.kappa_mode == 'one':
            kappa_sq = torch.ones_like(kappa_sq)
        elif self.kappa_mode == 'detach':
            kappa_sq = kappa_sq.detach()
        x = x + self.mlp(self.ln_2(x))
        self.reg_loss = (fdual_norm * kappa_sq).mean()
        return x


class GPT(nn.Module):
    """nanoGPT. ``forward`` returns ``(logits, loss, reg_loss)``; ``reg_loss`` is L_DB when ``reg=True`` and 0.0 otherwise."""

    def __init__(self, config, reg=False, kappa_mode='full'):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.reg = reg
        self.kappa_mode = kappa_mode

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, reg=reg, kappa_mode=kappa_mode) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.word_dropout = WordDropout(config.word_dropout)

        self.apply(self._init_weights)
        # special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        if reg:
            for block in self.transformer.h:
                block.init_power_iteration()
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        idx = self.word_dropout(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        reg_loss = 0.0
        for block in self.transformer.h:
            x = block(x)
            if self.reg:
                reg_loss = reg_loss + block.reg_loss
        if self.reg:
            reg_loss = reg_loss / self.config.n_layer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        else:
            logits = logits[:, [-1], :]
        return logits, loss, reg_loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
