"""Dual Banach regularizer L_DB for timm Vision Transformers.

Implementation notes:
  * the 1-step power iteration for the FFN spectral norms ||W1^l||, ||W2^l|| is batched over all L blocks (fp32);
  * the per-block regularizer terms (attention-output norm, per-token kappa_l) are pure functions, so that torch.compile can fuse
    them (compile_mode='reg') or the whole forward pass can be compiled (compile_mode='model' / 'model-cg');
  * the L per-block losses are averaged with one stack+mean.
kappa_l of forward k uses the spectral-norm estimates produced at forward k-1 (one-forward lag), the gradient flows through u, v and
W of that estimate, and the power iteration runs only in training mode.  The simple regularization controls (ctrl_mode 'specnorm' /
'blocknorm', added to the loss by the training engine as ctrl_lambda * model.ctrl_loss) are supported.
"""
from typing import Optional, Type
import torch, torch.nn as nn, torch.nn.functional as F
from timm.models import create_model
from timm.layers import use_fused_attn


class LinearLayerWithL2norm(nn.Module):
    """Linear layer + power-iteration vectors. The iteration itself is done batched over all blocks in VisionTransformer_RegOpt."""
    def __init__(self, linear_layer):
        super().__init__()
        self.linear = linear_layer
        W = self.linear.weight
        self._u = nn.Parameter(F.normalize(torch.randn(W.size(0)), dim=0), requires_grad=False)
        self._v = nn.Parameter(F.normalize(torch.randn(W.size(1)), dim=0), requires_grad=False)
        self.sigma = 1.0   # last estimate (detached), for inspection only

    def forward(self, x):
        return self.linear(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        self.fc1 = LinearLayerWithL2norm(nn.Linear(in_features, hidden_features, bias=bias))
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.Identity()
        self.fc2 = LinearLayerWithL2norm(nn.Linear(hidden_features, in_features, bias=bias))
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


# ---- regularizer cores (pure functions; compiled when compile_mode == 'reg') -------------------------------------------------
def attn_norm_core(x, weight, bias, dim, heads):
    """||proj(x_t)||^2-type attention-output norm per token: x [B,H,T,d] (pre-projection attention output) -> [B,T], fp32.
"""
    x = x.float(); W_split = weight.float().view(dim, heads, -1)                      # [D,H,d]
    W_square = torch.einsum('dhs,dhn->hsn', W_split, W_split)                        # [H,d,d]
    term = (torch.matmul(x, W_square.unsqueeze(0)) * x).sum(-1).sum(1) / (dim * heads)
    if bias is None:
        return term
    bias_split = bias.float() / heads; bias_W = torch.einsum('D,Dhd->hd', bias_split, W_split) / heads
    return term + 2 * torch.einsum('hd,bhtd->bt', bias_W, x) / dim + bias_split.square().mean().view(1, 1)


def block_reg_core(x_in, attn_norm, x_mid, gamma_max, s1, s2, lsigma, heads, kappa_mode):
    """Per-block L_DB term: mean_t [ (attn_norm + ||x_in||^2/H) * kappa_t^2 ],  kappa_t = 1 + lsigma*max|gamma|*||W2||*||W1|| / std(x_mid,t)."""
    x_norm = x_in.float().square().mean(-1) / heads
    fdual = attn_norm + x_norm
    kappa = (lsigma * gamma_max * s1 * s2 / x_mid.float().std(dim=-1) + 1).square()
    if kappa_mode == 'one':
        kappa = torch.ones_like(kappa)
    elif kappa_mode == 'detach':
        kappa = kappa.detach()
    return (fdual * kappa).mean()


_CORES = {'attn': attn_norm_core, 'block': block_reg_core}


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, proj_bias=True, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads; self.dim = dim; self.head_dim = dim // num_heads; self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1); attn = self.attn_drop(attn.softmax(dim=-1)); x = attn @ v
        with torch.autocast('cuda', enabled=False):
            attn_norm = _CORES['attn'](x, self.proj.weight, self.proj.bias, self.dim, self.num_heads)
        x = self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))
        return x, attn_norm


class DropPath(nn.Module):
    """Stochastic depth that drops the attention-output norm of a sample together with its branch output."""
    def __init__(self, drop_prob=0., scale_by_keep=True):
        super().__init__(); self.drop_prob = drop_prob; self.scale_by_keep = scale_by_keep

    def forward(self, x, attn_norm=None):
        if self.drop_prob == 0. or not self.training:
            return x, attn_norm
        keep_prob = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep: mask.div_(keep_prob)
        return x * mask, (None if attn_norm is None else attn_norm * mask.view(-1, 1))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, proj_bias=True, proj_drop=0., attn_drop=0.,
                 init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        assert init_values is None, 'LayerScale not used in the paper models'
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        self.ls1 = nn.Identity(); self.drop_path1 = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, bias=proj_bias, drop=proj_drop)
        self.ls2 = nn.Identity(); self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.num_heads = num_heads; self.kappa_mode = 'full'; self.lsigma = 1.0
        self.ctrl_mode = 'none'   # 'blocknorm': also return mean_t ||x_t^out||^2 (block-output-norm control)

    def forward(self, x, s1, s2):
        x_in = x
        y, attn_norm = self.attn(self.norm1(x))
        y, attn_norm = self.drop_path1(y, attn_norm)
        x = x + y
        x_mid = x
        y = self.mlp(self.norm2(x))
        x = x + (self.drop_path2(y)[0] if isinstance(self.drop_path2, DropPath) else y)
        reg = _CORES['block'](x_in, attn_norm, x_mid, self.norm2.weight.detach().max(), s1, s2, self.lsigma, self.num_heads, self.kappa_mode)
        out_sq = x.float().square().sum(-1).mean() if self.ctrl_mode == 'blocknorm' else None
        return x, reg, out_sq


class VisionTransformer_RegOpt(nn.Module):
    """timm ViT + L_DB.  forward(x) -> (logits, L_DB).
    compile_mode: 'none' (eager) | 'reg' (torch.compile the two regularizer cores)
                  | 'model' (torch.compile the whole forward) | 'model-cg' (same with CUDA graphs, mode='reduce-overhead')."""
    def __init__(self, model_name, num_classes, kappa_mode='full', lsigma=1.0, compile_mode='none', ctrl_mode='none'):
        super().__init__()
        assert ctrl_mode in ('none', 'specnorm', 'blocknorm'), ctrl_mode
        self.ctrl_mode = ctrl_mode; self.ctrl_loss = 0.0   # specnorm: sum_l (sigma1_l^2 + sigma2_l^2); blocknorm: (1/L) sum_l mean_t ||b_t^l||^2
        self.model = create_model(model_name, num_classes=num_classes, pretrained=False)
        for i, block in enumerate(self.model.blocks):
            dim = block.attn.qkv.in_features
            drop_path = 0.0 if isinstance(block.drop_path1, nn.Identity) else block.drop_path1.drop_prob
            self.model.blocks[i] = Block(dim=dim, num_heads=block.attn.num_heads, mlp_ratio=block.mlp.fc1.out_features / dim,
                                         qkv_bias=block.attn.qkv.bias is not None, qk_norm=not isinstance(block.attn.q_norm, nn.Identity),
                                         proj_bias=block.attn.proj.bias is not None, proj_drop=block.attn.proj_drop.p,
                                         attn_drop=block.attn.attn_drop.p, init_values=None, drop_path=drop_path)
            self.model.blocks[i].kappa_mode = kappa_mode; self.model.blocks[i].lsigma = lsigma; self.model.blocks[i].ctrl_mode = ctrl_mode
        self.L = len(self.model.blocks)
        self._sig = None            # (sigma1 [L], sigma2 [L]) from the last training forward, graph attached
        self.compile_mode = compile_mode
        if compile_mode == 'reg':
            _CORES['attn'] = torch.compile(attn_norm_core, dynamic=False); _CORES['block'] = torch.compile(block_reg_core, dynamic=False)
            self._core = self._forward_core
        elif compile_mode in ('model', 'model-cg'):
            self._core = torch.compile(self._forward_core, mode='reduce-overhead' if compile_mode == 'model-cg' else 'default', dynamic=False)
        else:
            assert compile_mode == 'none', compile_mode
            self._core = self._forward_core

    # -- pure part (compiled in 'model' modes) --
    def _forward_core(self, x, s1, s2):
        vt = self.model
        x = vt.patch_embed(x); x = vt._pos_embed(x); x = vt.patch_drop(x); x = vt.norm_pre(x)
        regs, outs = [], []
        for l, blk in enumerate(vt.blocks):
            x, r, o = blk(x, s1[l], s2[l]); regs.append(r); outs.append(o)
        x = vt.norm(x); x = vt.forward_head(x)
        blocknorm = torch.stack(outs).mean() if self.ctrl_mode == 'blocknorm' else None
        return x, torch.stack(regs).mean(), blocknorm

    # -- batched 1-step power iteration for all blocks (eager; ~25 kernels instead of ~250) --
    def _power_iteration(self):
        blocks = self.model.blocks; out = []
        with torch.autocast('cuda', enabled=False):
            for fc in ('fc1', 'fc2'):
                layers = [getattr(b.mlp, fc) for b in blocks]
                W = torch.stack([l.linear.weight for l in layers]).float()             # [L, out, in]  (copy: keeps the graph valid across the optimizer step)
                u0 = torch.stack([l._u for l in layers])                                 # [L, out]
                v = F.normalize(torch.bmm(W.transpose(1, 2), u0.unsqueeze(-1)).squeeze(-1), dim=-1)   # [L, in]
                Wv = torch.bmm(W, v.unsqueeze(-1)).squeeze(-1)                           # [L, out]
                u = F.normalize(Wv, dim=-1)
                sigma = (u * Wv).sum(-1)                                                  # = dot(u, W v) per layer, grad through u, v, W
                torch._foreach_copy_([l._u for l in layers], list(u.detach().unbind(0)))
                torch._foreach_copy_([l._v for l in layers], list(v.detach().unbind(0)))
                for l, s in zip(layers, sigma.detach().unbind(0)): l.sigma = s
                out.append(sigma)
        return out[0], out[1]

    def forward(self, x):
        if self._sig is None:
            ones = torch.ones(self.L, device=x.device); self._sig = (ones, ones)
        s1, s2 = self._sig
        if self.ctrl_mode != 'none':   # control runs do not train with L_DB: keep kappa's use of sigma out of the graph (the compiled backward
            s1, s2 = s1.detach(), s2.detach()   # would otherwise traverse the previous sigma graph a second time, after ctrl_loss consumed it)
        logits, reg, blocknorm = self._core(x, s1, s2)
        if self.training:
            self._sig = self._power_iteration()
        if self.ctrl_mode == 'specnorm':      # current estimates, graph attached
            self.ctrl_loss = (self._sig[0].square() + self._sig[1].square()).sum()
        elif self.ctrl_mode == 'blocknorm':
            self.ctrl_loss = blocknorm
        return logits, reg

    def load_state_dict(self, state_dict, strict=True):
        self._sig = None
        return super().load_state_dict(state_dict, strict=strict)
