import math
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from lit_gpt.config import Config
from einops import rearrange
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat, pack, unpack
from lit_gpt.rmsnorm import FusedRMSNorm
from .s6 import S6
from flash_attn import flash_attn_func

KVCache = Tuple[torch.Tensor, torch.Tensor]

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)

# @torch.compile
# def relu2(sim):
#     return torch.square(F.relu(sim))

@torch.compile
def gelu(sim):
    return F.silu(sim)
    #return torch.sigmoid(1.702*sim) * sim

@torch.compile
def relu2(sim):
    #return torch.softmax(sim, dim=-1) * sim
    return torch.relu(sim) * sim


@torch.compile
def geglu(sim):
    g, x=torch.chunk(sim, 2, dim=-1)
    return gelu(g) * x

@torch.compile
def geglu_op(g, x):
    return gelu(g) * x


class Relu2Attention(nn.Module):
    def __init__(self, config: Config, layer_idx: int , n_embd: int, ) -> None:
        super().__init__()

        
        self.softmax_attn = True
        self.gated = False
        self.flash_attn = False or not self.gated
        if self.gated:
        
            self.qk_h = 128
            self.v_h = 256
            n_inner = n_embd * 2
            if not self.flash_attn:
                q_dim = self.qk_h
            else:
                q_dim = n_inner // 2
            v_dim  = n_inner
            self.q_dim = q_dim
            self.k_dim = self.qk_h #mqa
            self.v_dim = v_dim
            o_dim = v_dim
            shape = v_dim * 2 + q_dim + self.k_dim 
            dt_rank="auto"
        else: 
            self.qk_h = 128
            self.v_h = 128 
            self.q_dim = n_embd
            n_head = self.q_dim // self.qk_h
            kv_head = n_head //4
            self.k_dim = kv_head * self.qk_h #mqa
            self.v_dim = kv_head * self.v_h    
            shape = self.q_dim + self.k_dim + self.v_dim
            o_dim = self.q_dim
            dt_rank = "auto" #64


        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        # self.gamma = nn.Parameter(torch.Tensor(2, n_embd))
        # self.beta = nn.Parameter(torch.Tensor(2, n_embd))
        # nn.init.normal_(self.gamma, mean=0.0, std=1/math.sqrt(q_dim))
        # nn.init.constant_(self.beta, 0.0)
        #self.norm = FusedRMSNorm(n_embd, eps=config.norm_eps)
        # output projection
        factory_kwargs = {"device": "cuda", "dtype": torch.float32}
        self.s6 = S6( self.k_dim + self.v_dim, dt_rank= dt_rank, **factory_kwargs)        
        self.o_proj = nn.Linear( o_dim, n_embd, bias=config.bias)
        self.config = config
        self.window_size = config.local_window
        self.scale = 1.0 / math.sqrt(self.qk_h)
        
    def forward(
        self,
        x: torch.Tensor,
        rope = None,
        max_seq_length=4096,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkvg = self.attn(x)
        # split batched computation into three
        if self.gated:
            q, kv, g = qkvg.split((self.q_dim, self.k_dim + self.v_dim, self.v_dim), dim=-1)  
        else: 
            q, kv = qkvg.split((self.q_dim, self.k_dim + self.v_dim), dim=-1)  
        kv = self.s6(kv)
        k, v = kv.split((self.k_dim, self.v_dim), dim=-1)
        # z = z.unsqueeze(-2) * self.gamma + self.beta
        # # B x L x 2 x S -> B x L x S
        # q, k = torch.unbind(z, dim=-2)

        if self.flash_attn:
            q = q.reshape(B,  T, -1, self.qk_h )  # (B, T, nh_q, hs)
            k = k.reshape(B,  T, -1, self.qk_h )  
            v = v.reshape(B,  T, -1, self.v_h )  
            if self.config.local_window > -1:
                win_tuple = (self.config.local_window-1, 0)
            else:
                win_tuple = (-1,-1)
            y = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=self.scale, causal=True, window_size=win_tuple)
        else:
            q = q.reshape(B,  T, -1 )  # (B, T, nh_q, hs)
            k = k.reshape(B,  T, -1 )  
            v = v.reshape(B,  T, -1 )  
            y = self.local_attention(q, k, v)
        
        y = y.reshape(B, T, -1)
        # output projection
        #y = self.norm(y)
        if self.gated:
            y = self.o_proj(geglu_op(g,y))
        else:
            y = self.o_proj(y)
        return y, kv_cache
    
    def local_attention(self, q, k, v):
        
        # attn_bias: b,k,c,c
        autopad, pad_value, window_size = True, -1, self.config.local_window
        exact_windowsize = True

        look_backward = 1
        look_forward = 0
        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))


        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype

        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size


        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)

        # bucketing
        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )

        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)


        # calculate positions for masking
        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')

        pad_mask = bq_k == pad_value

        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)
        
        if self.softmax_attn:
            mask_value = -torch.finfo(sim.dtype).max
        else:
            mask_value = 0

        causal_mask = bq_t < bq_k

        if exact_windowsize:
            max_causal_window_size = (self.window_size * look_backward)
            causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

        sim = sim.masked_fill(causal_mask, mask_value)
        #lengths = ( ~(causal_mask + pad_mask)).sum(-1, keepdim = True)
        del causal_mask


        sim = sim.masked_fill(pad_mask, mask_value)
        
        if self.softmax_attn:
            attn = F.softmax(sim, dim = -1, dtype = torch.float32)
        else:
            attn = gelu(sim)

        # aggregation
        out = einsum('b h i j, b h j e -> b h i e', attn, bv) #/ lengths #* 2 ** 0.5 / self.config.local_window
        out = rearrange(out, 'b w n d -> b (w n) d')

        out, *_ = unpack(out, packed_shape, '* n d')
        
        return out
    
class Relu2MLP(nn.Module):
    def __init__(self, config: Config,) -> None:
        super().__init__()
        self.in_proj = nn.Linear(config.n_embd, 2 * config.intermediate_size, bias=config.bias)
        self.o_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        x = self.o_proj(geglu(self.in_proj(x)))
        return x