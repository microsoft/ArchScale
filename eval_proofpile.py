# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional, Union
import gc
from pretrain import use_flce_loss
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import numpy as np
import torch.nn.functional as F
import argparse
import os
import json
from pathlib import Path
from eval import load_model
try:
    from lit_gpt.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
except:
    FusedLinearCrossEntropyLoss = None

class ProofPile:
    """Proof-Pile perplexity and accuracy evaluation. Following LongLoRA: 128 samples; sliding window style inference.

    Reference:
        - ProofPile: https://huggingface.co/datasets/hoskinson-center/proof-pile
        - LongLoRA: https://github.com/dvlab-research/LongLoRA/blob/main/eval.py
        - Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation: https://arxiv.org/abs/2108.12409

    """
    @staticmethod
    def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
        all_ix = list(range(0, len(data) - seq_length, sliding_window))
        all_ix.pop()

        for idx in range(0, len(all_ix), batch_size):
            ix = all_ix[idx:idx+batch_size]
            assert all([idx + seq_length + 1 <= len(data) for idx in ix])
            x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
            if device != 'cpu':
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            yield x, y

    @staticmethod
    def iceildiv(x, y):
        return (x + y - 1) // y

    @staticmethod
    def run(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        batch_size: Optional[int] = 48,
        seq_length: Optional[int] = 128000,
        sliding_window: Optional[int] = 256,
        data_path: str = "test_sampled_data.bin",
        **kwargs,
    ) -> Dict[str, Any]:

        print(f"Running ProofPile evaluation with batch_size={batch_size}, seq_length={seq_length}, sliding_window={sliding_window}, data_path={data_path}")

        data = {'val': np.memmap(data_path, dtype=np.uint16, mode='r')}
        print(f"Num original validation tokens: {len(data['val'])}")

        #ori_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        #print(data['val'])
        #inputs = ori_tokenizer.decode(data['val'])
        #print(inputs)
        #tokenized_example =  np.array(tokenizer(inputs, truncation=False, return_attention_mask=False, add_special_tokens=False)["input_ids"])
        #print(tokenized_example)
        #data = {'val': tokenized_example}
        print(f"Num validation tokens: {len(data['val'])}")
        # if tokenizer.pad_token_id is None:
        #     tokenizer.pad_token_id = tokenizer.eos_token_id
        loss_list_val, acc_list = [], []
        loss_step_list_val = []
        eval_step = 0
        loss_func = FusedLinearCrossEntropyLoss()
        for idx, (x, y) in tqdm(
            enumerate(
                ProofPile.get_as_batch(
                    data['val'],
                    seq_length,
                    batch_size,
                    device=device,
                    sliding_window=sliding_window
                )
            ),
            total=ProofPile.iceildiv(
                ProofPile.iceildiv(len(data['val']), sliding_window),
                batch_size
            )
        ):
            eval_step += 1
            # if eval_step > 1000:
            #     break
            val_loss = 0.
            acc = 0.
            cnt = 0
            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                part_len = x[:, i:i + seq_length].shape[1]
                input_ids = x[:, i:i + seq_length].cuda()
                
                seq_len = part_len
                #chunk_len = 1024 #65536
                seq_cut = seq_len #65536
                if seq_len> seq_cut:
                    chunk_len = seq_cut #80GB
                else:
                    chunk_len = seq_len
                with torch.no_grad():
                    assert seq_len % chunk_len == 0
                    cache = None 
                    part_loss = 0
                    plen = 0
                    for i in range(seq_len//chunk_len):
                        start = i * chunk_len
                        end  = start + chunk_len
                        n_input = input_ids[:,start:end]
                        # print(n_input.shape)
                        m_output = model(input_ids=n_input, use_flce_loss=True) #past_key_values = cache, use_cache = True)
                        logits = m_output.logits

                        #cache = m_output.past_key_values

                        labels = n_input.clone()
                        logits = logits[..., :-1, :].contiguous()
                        labels = labels[..., 1:].contiguous()
                        
                        loss = loss_func(logits, m_output.weight, labels).cpu().item()  
                        # loss = F.cross_entropy(
                        #         logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=32000 #tokenizer.pad_token_id
                        #     ).cpu().item()   
                        part_loss += loss * labels.shape[-1]
                        plen += labels.shape[-1]
                    loss = part_loss / plen
      
                print(loss)
                val_loss = loss * part_len + val_loss
                #acc = ((outputs["logits"].argmax(-1).to(y.device) == y[:, i:i+seq_length]).float().sum()) + acc
                #outputs = None
                # gc.collect()
                # torch.cuda.empty_cache()
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(loss)
            val_loss /= cnt
            acc /= cnt

            loss_list_val.append(val_loss)
            acc_list.append(acc)

        result = {
            "loss": torch.as_tensor(loss_list_val).mean().item(),
            "ppl": 2.71828 ** torch.as_tensor(loss_list_val).mean().item(),
            "ppl_per_chunk": torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1)).item(),
            "acc": torch.as_tensor(acc_list).mean().item()
        }
        return result

def main():
    parser = argparse.ArgumentParser(description='Run ProofPile evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Model configuration name')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--seq_length', type=int, default=65536, help='Sequence length for evaluation')
    parser.add_argument('--sliding_window', type=int, default=4096, help='Sliding window size')
    parser.add_argument('--data_path', type=str, default='test_sampled_data.bin', help='Path to the data file')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='Model dtype (bfloat16, float16, or float32)')
    parser.add_argument('--tokenizer_name', type=str, default='Orkhan/llama-2-7b-absa', help='Tokenizer name')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    # Load model and tokenizer
    print(f"Loading model from {args.checkpoint_path}...")
    model = load_model(args.checkpoint_path, args.config, device, dtype)
    tokenizer = None #AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Run evaluation
    print("Starting ProofPile evaluation...")
    result = ProofPile.run(
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        sliding_window=args.sliding_window,
        data_path=args.data_path
    )
    
    # Save results
    output_file = os.path.join(args.output_dir, f"proofpile_results.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {output_file}")
    print(f"Loss: {result['loss']:.4f}")
    print(f"Perplexity: {result['ppl']:.4f}")
    print(f"Perplexity per chunk: {result['ppl_per_chunk']:.4f}")
    print(f"Accuracy: {result['acc']:.4f}")

if __name__ == "__main__":
    main()