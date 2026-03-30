#!/bin/bash

# Usage:  ./eval.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B aime24 data/evals

echo "START TIME: $(date)"

SEED=$1

MODEL=$2
# supported models:
# #deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

TASK=$3
# supported tasks:
# aime24
# math_500
# gpqa_diamond

OUTPUT_ROOT=$4

#export CUDA_VISIBLE_DEVICES=1,2,3,4
get_gpu_count() {
    # Check if CUDA_VISIBLE_DEVICES is set
    if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
        # Check if CUDA is disabled
        if [[ "$CUDA_VISIBLE_DEVICES" == "-1" ]]; then
            echo 0
            return
        fi

        # Count GPUs from CUDA_VISIBLE_DEVICES
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
        return
    fi

    # If CUDA_VISIBLE_DEVICES not set, count all GPUs using nvidia-smi
    nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l
}

NUM_GPUS=$(get_gpu_count)
# temp 0.6
MODEL_ARGS="seed=$SEED,model_name=$MODEL,max_num_batched_tokens=32768,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,trust_remote_code=True,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# temp 0
#MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.95,trust_remote_code=True,temperature=0"
OUTPUT_DIR=$OUTPUT_ROOT/$MODEL

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# sudo for shadow 
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks eval_reason.py \
    --use-chat-template \
    --save-details \
    --output-dir $OUTPUT_DIR


echo "END TIME: $(date)"
