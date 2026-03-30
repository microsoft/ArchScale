"""Count active and total parameters for transformer_gqa4_h2_moe_s8 at different depths."""

import sys, types

def find_multiple(n, k):
    if n % k == 0:
        return n
    return n + k - (n % k)

# Stub out heavy dependencies so we can import config.py standalone
fake_utils = types.ModuleType('lit_gpt.utils')
fake_utils.find_multiple = find_multiple
sys.modules['lit_gpt.utils'] = fake_utils
sys.modules['lit_gpt'] = types.ModuleType('lit_gpt')
sys.modules['lit_gpt.model'] = types.ModuleType('lit_gpt.model')

import torch  # noqa: needed by Config dataclass
exec(compile(open('lit_gpt/config.py').read(), 'lit_gpt/config.py', 'exec'))


def total_moe_params(model_config):
    """Compute total parameter count including all MoE experts."""
    c = model_config
    n_total = 0

    # Embedding + LM head
    n_total += c.padded_vocab_size * c.n_embd  # wte
    n_total += c.n_embd * c.padded_vocab_size  # lm_head

    per_layer = 0

    # Layer norms (norm_1, norm_2)
    per_layer += 2 * c.n_embd

    # Attention (GQA): Q, K, V, O projections
    per_layer += c.n_embd * (c.n_head * c.head_size)          # Q
    per_layer += c.n_embd * (c.n_query_groups * c.head_size)   # K
    per_layer += c.n_embd * (c.n_query_groups * c.head_size)   # V
    per_layer += (c.n_head * c.head_size) * c.n_embd           # O

    # MoE MLP: n_mods = sparsity * top_k experts
    n_mods = c.sparsity * c.top_k
    mod_in_size = int(c.intermediate_size * 2) // c.top_k   # SwiGLU gate+up
    mod_out_size = c.intermediate_size // c.top_k            # down proj

    per_layer += n_mods * c.n_embd * mod_in_size    # mods_in
    per_layer += n_mods * mod_out_size * c.n_embd   # mods_out
    per_layer += c.n_embd * n_mods                  # router

    n_total += c.n_layer * per_layer

    # Final layer norm
    n_total += c.n_embd

    return n_total


depths = [8, 12, 16, 20]
sparsity = 8
top_k = 8
train_config = 'v2scale'

print(f"{'Model':<45} {'Depth':>5} {'Active (B)':>12} {'Total (B)':>12} {'Ratio':>7}")
print("-" * 85)

for d in depths:
    model_name = f"transformer_gqa4_h2_moe_s{sparsity}_k{top_k}_d{d}"
    cfg = Config(**name_to_config[model_name])  # noqa: Config, name_to_config from exec

    active = get_parameters_count(model_name, depth=d, model_config=cfg, train_config=train_config)
    total = total_moe_params(cfg)

    print(f"{model_name:<45} {d:>5} {active/1e9:>12.3f} {total/1e9:>12.3f} {total/active:>7.1f}x")
