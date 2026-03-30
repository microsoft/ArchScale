import math
for depth in [8,12,16,20,24]:
    for model_name in ["samba","sambay","transformer","sambayoco"]:
        print(f"\nDepth {depth} {model_name}:")
        nodes = 8
        micro_batch_size = 2
        devices = 8
        seq_len = 4096
        train_config = "scaling"
        model_name = model_name+"_d"+str(depth)
        if "samba" in model_name:
            ar = 122
            mult = 15 * (ar ** 2) + 160 * ar
        if "sambay" in model_name:
            ar = 124
            mult = 14.5 * (ar ** 2) + 144 * ar
        if "transformer" in model_name:
            ar = 128
            mult = 14.5 * (ar ** 2) # 237568
        if "sambayoco" in model_name:
            ar = 126
            mult = 13.5 * (ar ** 2) + 208 * ar
        
        # Base parameters (d0 = 16)
        d0 = 16
        eta0 = 4e-4  # Base learning rate
        b0 = 2**21   # Base batch size (2M tokens)
        t0 = int(1e11)  # Base tokens (100B)
        
        # Calculate base parameters with correct multiplier
        n0 = 237568 * (d0**3)  # Base parameters
        
        # Calculate target parameters
        n_target = mult * (depth**3)
        # Scale tokens based on parameter count (Chinchilla scaling)
        train_tokens = int(t0 * n_target / n0)
        # Scale learning rate and batch size
        raw_b = int(b0)
        multiple = nodes * devices * micro_batch_size * seq_len
        b = (raw_b // multiple) * multiple
        
        learning_rate = eta0 * math.sqrt(b*d0/depth/b0)


        global_batch_size =  b // (seq_len * nodes)

        print(f"Parameters: {n_target:,}")
        print(f"Raw batch size: {raw_b}")
        print(f"Batch size: {b}")
        print(f"Width: {ar*depth}")
        print(f"Learning rate: {learning_rate:.2e}")
        print(f"Global batch size: {global_batch_size}")
        print(f"Max tokens: {train_tokens:,}")

def legacy_update_global(depth=None):
    global model_name, train_config, name, out_dir, devices, learning_rate, nodes, train_tokens, \
        global_batch_size, micro_batch_size, total_evals, warmup_tokens, log_step_interval, \
        eval_iters, min_lr, batch_size, gradient_accumulation_steps, log_iter_interval, hparams
    if "20B" in name:
        train_tokens = int(1e11) // 5 # 20 billion
    elif "100B" in name:
        train_tokens = int(1e11) # 100 billion
    elif "2.5B" in name:
        train_tokens = int(2.5e9) # 2.5B billion
        learning_rate = 6e-4
    elif "7B" in name:
        train_tokens = int(7e9) # 7 billion
        learning_rate = 3e-4
    elif "15B" in name:
        train_tokens = int(15e9) # 15 billion
        learning_rate = 2.5e-4
    elif "26B" in name:
        train_tokens = int(26e9) # 26 billion
        learning_rate = 2e-4

    if "512x4k" in name:
        #4k
        global_batch_size = 512 // nodes
        micro_batch_size = 4 # 8
    elif "128x4k" in name:
        #4k
        global_batch_size = 128 // nodes
        micro_batch_size = 1 # 8
    elif "256x8k" in name:
        #8k
        global_batch_size = 256 // nodes
        micro_batch_size = 2 
    elif "128x16k" in name:
        #16k
        global_batch_size = 128 // nodes
        micro_batch_size = 1 # 2 
        # learning_rate = 6e-4
    elif "32x16k" in name:
        #16k
        global_batch_size = 32 // nodes
        micro_batch_size = 1 # 2 
        # learning_rate = 6e-4
    elif "64x8k" in name:
        #16k
        global_batch_size = 64 // nodes
        micro_batch_size = 1 # 2 
        # learning_rate = 6e-4
    elif "64x32k" in name:
        #32k
        global_batch_size = 64 // nodes
        micro_batch_size = 1 
    elif "1024x2k" in name:
        #2k
        global_batch_size = 1024 // nodes
        micro_batch_size = 16
# learning rate decay scheduler (linear warmup and decay)
# learning rate scheduler with warmup, stable period, and decay
def get_lr(it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    
    if "wsd" in train_config:
        # 3) stable period for 5/7 of training after warmup (deepseekv3 schedule)
        stable_iters = int(5/7 * (max_iters - warmup_iters))
        if it < warmup_iters + stable_iters:
            return learning_rate
            
        # 4) decay period for remaining iterations
        decay_iters = max_iters - warmup_iters - stable_iters
        decay_ratio = (it - warmup_iters - stable_iters) / decay_iters
        assert 0 <= decay_ratio <= 1
    else:
        # 3) in between, use linear or cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    if "scaling" in train_config:
        # Linear decay
        return learning_rate + decay_ratio * (min_lr - learning_rate)
    else:
        # Cosine decay
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

def get_param_groups(model):
    """Group parameters by their types to apply different learning rate multipliers"""
        
    no_decay = vector_names
    for n, p in model.named_parameters():
        print(n, p.shape)
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "weight" in n.lower() and not any(nd in n.lower() for nd in no_decay) 
            ],
            "weight_decay": weight_decay,
            "lr_mult": gd0 / gd,  # Base multiplier for weights
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not "weight" in n.lower() or any(nd in n.lower() for nd in no_decay))
                 and not any(x in n.lower() for x in ["wte"])
            ],
            "weight_decay": 0.0,
            "lr_mult": 1.0,  # Base multiplier for no-decay parameters (vectors)
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(x in n.lower() for x in ["wte"])
            ],
            "weight_decay": 0.0,
            "lr_mult": 20.0,  # Higher multiplier for embeddings and lm_head
        }
    ]

    return param_groups

def module_filter_fn(mod, fqn): 
    return ( 
        mod.in_features >= size_limit 
        and mod.out_features >= size_limit 
        and mod.in_features % 16 == 0 
        and mod.out_features % 16 == 0 
    ) 

def update_global(depth=None):
    weight_decay = 0.1
    multiple = nodes * devices * micro_batch_size * seq_len
    beta1 = 0.9
    beta2 = 0.95
    eps = 1e-8
    warmup_tokens = int(1e9)
    if super_mup: 
        raw_b =  b0 #int(b0 * math.sqrt(train_tokens/t0))
        b = (raw_b // multiple) * multiple
        learning_rate = eta0 * math.sqrt(d0/depth) * math.sqrt(b/b0)
        eps = eps / math.sqrt(b/b0)
        beta1 = 1 - (1-beta1) * b/b0
        beta2 = 1 - (1-beta2) * b/b0
        warmup_tokens = int(warmup_tokens *10 * (b / b0 )) # iso-steps
        weight_decay = weight_decay / 10 * eta0 / learning_rate
    else:
        raw_b = b0
        b = (raw_b // multiple) * multiple
        learning_rate = eta0 * math.sqrt(b/b0)
        if mup:
            learning_rate = learning_rate * math.sqrt(d0/depth)
    global_batch_size =  b // (seq_len * nodes)
    
def plot_lr_schedule():
    import matplotlib.pyplot as plt
    import numpy as np
    global learning_rate
    # Test parameters
    test_warmup = 1000
    test_max_iter = 100000
    iterations = np.arange(0, test_max_iter + 500)
    
    # Get learning rates for both scaling and non-scaling
    global train_config
    global min_lr 
    min_lr = 0
    # Test scaling setting
    train_config = "scaling"
    learning_rate = 4e-4
    lr_scaling = [get_lr(it, test_warmup, test_max_iter) for it in iterations]
    
    train_config = "wsd_scaling"
    learning_rate = 4e-3
    lr_scaling_wsd_super= [get_lr(it, test_warmup*10, test_max_iter) for it in iterations]
    
    train_config = "scaling"
    learning_rate = 4e-3
    lr_scaling_super= [get_lr(it, test_warmup*10, test_max_iter) for it in iterations]
       
    # # Test cosine setting
    # train_config = "cosine" 
    # lr_cosine = [get_lr(it, test_warmup, test_max_iter) for it in iterations]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, lr_scaling, label='linear decay')
    plt.plot(iterations, lr_scaling_wsd_super, label='WSD Super')
    plt.plot(iterations, lr_scaling_super, label='super')
    #plt.plot(iterations, lr_cosine, label='Cosine decay')
    plt.axvline(x=test_warmup, color='gray', linestyle='--', label='End of warmup')
    plt.axvline(x=test_max_iter, color='gray', linestyle='--', label='Max iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_schedule.png')
    plt.close()

plot_lr_schedule()
    