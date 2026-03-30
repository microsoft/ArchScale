import torch
def sort_tokens(n_routed_experts, x, topk_ids, topk_weights):
    #credit: https://github.com/pytorch/torchtitan/blob/main/experiments/deepseek_v3/model.py

    # This part sorts the token indices so that tokens routed to the same expert reside consecutively.
    # An implication is that tokens to the same "expert group" (i.e., device) are also consecutive.
    # Since this is an "aritificial" index creation (final outcome being
    # `idxs`), we don't need gradients here.

    with torch.no_grad():
        # [seq_len, n_routed_experts]
        expert_counts = topk_ids.new_zeros(
            (topk_ids.shape[0], n_routed_experts)
        )
        # Fill 1 to the selected experts
        expert_counts.scatter_(1, topk_ids, 1)
        tokens_per_expert = expert_counts.sum(dim=0)
        # Token indices for each expert
        token_indices = topk_ids.view(-1).argsort(stable=True)

    sorted_tokens = x[token_indices // topk_ids.shape[1]]
    # assert sorted_tokens.shape == sorted_tokens_shape

    return (sorted_tokens, token_indices, tokens_per_expert)

def ref_gmm(sorted_tokens,w_gate,tokens_per_expert):
    n_exp = tokens_per_expert.shape[0]
    y_ref = torch.empty(sorted_tokens.shape[0], d_out, dtype=torch.bfloat16, device="cuda")
    offset = 0
    for i in range(n_exp):
        y_ref[offset:offset+tokens_per_expert[i],:] = (sorted_tokens[offset:offset+tokens_per_expert[i],:] @ w_gate[i,:])   # (K) @ (K,N) -> (N)
        offset =offset+tokens_per_expert[i]
    return y_ref

d_m=8
slen=16
expt=8
d_out=16
scores = torch.randn(slen,expt)
print(f"{scores=}")
topk_weight, topk_idx = torch.topk(scores, k=2,dim=-1, sorted=False)
print(f"{topk_weight=},{topk_idx=}")
x = torch.randn(slen,d_m)
print(f"{x=}")
#print(sort_tokens(expt,x,topk_idx,topk_weight))

sorted_tokens, token_indices, tokens_per_expert = sort_tokens(expt,x,topk_idx,topk_weight)
print(f"{token_indices=}")
m_offsets = torch.cumsum(tokens_per_expert, 0) #- tokens_per_expert
m_offsets = m_offsets.to(dtype=torch.int32, device="cuda")
#m_offsets = torch.cat([m_offsets,torch.tensor([slen],dtype=torch.int32,device="cuda")])
print(f"{m_offsets=}")
print(f"{tokens_per_expert=}")
#print(f"{sorted_tokens=}")
print(sorted_tokens.shape, )
w_gate = torch.randn(expt,d_m,d_out).bfloat16().cuda()
sorted_tokens = sorted_tokens.bfloat16().cuda().contiguous()
#gout = torch._grouped_mm(sorted_tokens, w_gate, m_offsets, out_dtype=torch.bfloat16,)
import transformer_engine as te
L=te.pytorch.GroupedLinear(expt,d_m,d_out, bias= False)
for i in range(expt):
    getattr(L, f"weight{i}").data = w_gate[i,:,:].transpose(0,1).contiguous()
gout = L(sorted_tokens, tokens_per_expert.tolist())
print(f"{gout=}")
gout_ref = ref_gmm(sorted_tokens,w_gate,tokens_per_expert)
print(f"{gout_ref=}")

