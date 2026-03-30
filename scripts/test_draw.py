import numpy as np
import matplotlib.pyplot as plt

# Define learning rate schedule function
def linear_warmup_linear_decay_lr(step, total_steps, warmup_fraction, peak_lr=4e-4, final_lr=4e-5):
    """
    Linear warmup followed by linear decay learning rate schedule.
    
    Args:
        step: Current step number
        total_steps: Total number of steps in training
        warmup_fraction: Fraction of steps for warmup
        peak_lr: Peak learning rate (after warmup)
        final_lr: Final learning rate after decay
    """
    warmup_steps = int(total_steps * warmup_fraction)
    
    if step < warmup_steps:
        # Linear warmup phase
        return peak_lr * (step / warmup_steps)
    else:
        # Linear decay phase
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        return peak_lr + decay_ratio * (final_lr - peak_lr)

# Setup
total_tokens = 100e9  # 100B tokens
tokens_per_step = 1e6  # 1M tokens per step (arbitrary but reasonable)
total_steps = int(total_tokens / tokens_per_step)

peak_lr = 4e-4
final_lr = 0  # 10% of peak_lr

# Create step values from 0 to total_steps
steps = np.linspace(0, total_steps, 1000)
tokens = steps * tokens_per_step / 1e9  # Convert to billions of tokens

# Calculate learning rates for different warmup fractions
lr_10pct_warmup = [linear_warmup_linear_decay_lr(s, total_steps, 0.1, peak_lr, final_lr) for s in steps]
lr_50pct_warmup = [linear_warmup_linear_decay_lr(s, total_steps, 0.5, peak_lr/2, final_lr) for s in steps]
lr_combined = [s+k for s,k in zip(lr_10pct_warmup, lr_50pct_warmup)]
# Plotting
plt.figure(figsize=(10, 6))

plt.plot(tokens, lr_10pct_warmup, 'b-', linewidth=2, label='10% Warmup')
plt.plot(tokens, lr_50pct_warmup, 'r-', linewidth=2, label='50% Warmup')    
plt.plot(tokens, lr_combined, 'g-', linewidth=2, label='Combined Schedule')
# Highlight warmup regions
plt.axvline(x=10, color='b', linestyle='--', alpha=0.5, label='10% Warmup End')
plt.axvline(x=50, color='r', linestyle='--', alpha=0.5, label='50% Warmup End')

# Add labels and title
plt.xlabel('Training Tokens (Billions)')
plt.ylabel('Learning Rate')
plt.title('Linear Decay LR Schedule with Different Warmup Periods (100B Tokens)')
plt.grid(True, alpha=0.3)
plt.legend()

# Format y-axis as scientific notation
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Add annotations showing the warmup regions
plt.fill_between(tokens, 0, peak_lr, where=(tokens <= 10), color='blue', alpha=0.1)
plt.fill_between(tokens, 0, peak_lr, where=(tokens <= 50), color='red', alpha=0.1)

plt.tight_layout()
plt.savefig('lr_schedule_comparison.png')
plt.show()