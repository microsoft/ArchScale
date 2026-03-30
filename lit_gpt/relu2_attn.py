import math
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from lit_gpt.config import Config
from einops import rearrange
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from torch import einsum
from einops import rearrange, repeat, pack, unpack

KVCache = Tuple[torch.Tensor, torch.Tensor]

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim = dim)

@torch.compile
def relu2(sim):
    return torch.square(F.relu(sim))

class Relu2Attention(nn.Module):
    def __init__(self, config: Config, layer_idx: int , n_embd: int, ) -> None:
        super().__init__()
        shape = n_embd *3
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(n_embd, shape, bias=config.bias)
        # output projection
        self.o_proj = nn.Linear(n_embd, n_embd, bias=config.bias)
        self.config = config
        self.sc = config.sc_attn
        self.window_size = config.local_window
        assert config.local_window >0, "local window should be greater than 0"
        if self.sc:
            self.q_dim = n_embd
            self.kv_dim = n_embd
            d_conv = 4
            self.q_conv1d = nn.Conv1d(
                in_channels=self.q_dim,
                out_channels=self.q_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.q_dim,
                padding=d_conv - 1,
            )
            self.k_conv1d = nn.Conv1d(
                in_channels=self.kv_dim,
                out_channels=self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups=self.kv_dim,
                padding=d_conv - 1,
            )
            self.v_conv1d = nn.Conv1d(
                in_channels= self.kv_dim,
                out_channels= self.kv_dim,
                bias=False,
                kernel_size=d_conv,
                groups= self.kv_dim,
                padding=d_conv - 1,
            ) 

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

        qkv = self.attn(x)
        # split batched computation into three
        q, k, v = qkv.split((C, C, C), dim=-1)
        q = q.reshape(B,  T, -1 )  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1 )  
        v = v.reshape(B,  T, -1 )  
        if self.sc:
            q = causal_conv1d_fn(
                        x = q.transpose(-1,-2),
                        weight=rearrange(self.q_conv1d.weight, "d 1 w -> d w"),
                        bias=self.q_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            k = causal_conv1d_fn(
                        x = k.transpose(-1,-2),
                        weight=rearrange(self.k_conv1d.weight, "d 1 w -> d w"),
                        bias=self.k_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2)
            v = causal_conv1d_fn(
                        x = v.transpose(-1,-2),
                        weight=rearrange(self.v_conv1d.weight, "d 1 w -> d w"),
                        bias=self.v_conv1d.bias,
                        activation="silu",
                    ).transpose(-1,-2) 

        q = q / self.config.local_window
        
        y = self.local_attention(q, k, v)
        # output projection
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
        

        mask_value = 0

        causal_mask = bq_t < bq_k

        if exact_windowsize:
            max_causal_window_size = (self.window_size * look_backward)
            causal_mask = causal_mask | (bq_t > (bq_k + max_causal_window_size))

        sim = sim.masked_fill(causal_mask, mask_value)
        del causal_mask


        sim = sim.masked_fill(pad_mask, mask_value)
        

        attn = relu2(sim)

        # aggregation
        out = einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        out, *_ = unpack(out, packed_shape, '* n d')
        
        return out
    
class Relu2MLP(nn.Module):
    def __init__(self, config: Config,) -> None:
        super().__init__()
        self.in_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.o_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.o_proj(relu2(self.in_proj(x)))
        return x