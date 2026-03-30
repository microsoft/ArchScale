#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
uv venv --clear
source .venv/bin/activate
uv pip install --upgrade pip

# torch 2.8.0+cu129
# gb200:
# Detect GPU type and install corresponding torch wheel
set -e

# Default to "unknown"
GPU_TYPE="unknown"
DEVICE=""
if command -v nvidia-smi &> /dev/null; then
    DEV_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    # Case-insensitive match for B100/GB200/H100
    if [[ "${DEV_NAME,,}" == *"gb200"* || "${DEV_NAME,,}" == *"b100"* ]]; then
        GPU_TYPE="gb200"
    elif [[ "${DEV_NAME,,}" == *"h100"* ]]; then
        GPU_TYPE="hopper"
    elif [[ "${DEV_NAME,,}" == *"h200"* ]]; then
        GPU_TYPE="hopper"
    fi
elif [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    # fallback to env variable
    GPU_TYPE="unknown"
fi

echo "Detected GPU_TYPE: ${GPU_TYPE}"

# prepare torch wheel first
if [[ "$GPU_TYPE" == "gb200" ]]; then
    echo "Installing for GB200/B100..."
    uv pip install torch-2.9.1+cu129-cp312-cp312-manylinux_2_28_aarch64.whl
    uv pip install torchvision==0.24.1
elif [[ "$GPU_TYPE" == "hopper" ]]; then
    echo "Installing for H100..."
    uv pip install torch-2.8.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl
    uv pip install -U torchvision==0.23.0
else
    echo "Unknown GPU type. Defaulting to H100 wheel for safety (x86_64)."
    uv pip install torch-2.8.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl
    uv pip install -U torchvision==0.23.0
fi


uv pip install packaging lightning jsonargparse[signatures] tokenizers sentencepiece wandb \
    torchmetrics tensorboard zstandard pandas pyarrow huggingface_hub transformers numpy \
    torchao==0.13.0 einops opt_einsum

uv pip install git+https://github.com/state-spaces/mamba@v2.2.6.post1 --no-build-isolation
# only for varlen training. comment the following for regular training or eval.
#git clone https://github.com/zigzagcai/varlen_mamba.git --branch feat/add-cu_seqlens
#cd ../varlen_mamba
#uv pip install --no-build-isolation -e .

uv pip install git+https://github.com/Dao-AILab/causal-conv1d@v1.5.3 --no-build-isolation


uv pip install git+https://github.com/renll/flash-linear-attention.git
uv pip install azureml-core lm-eval["ruler"] psutil


git clone https://github.com/Dao-AILab/flash-attention.git ../flash-attention
cd ../flash-attention && git checkout 59635947a020c3a99ce4bd360d4e221b8f3af572
if [[ "$GPU_TYPE" == "gb200" ]]; then
    # flash-attention 4
    uv pip install flash_attn/cute
    cd ../ArchScale

elif [[ "$GPU_TYPE" == "hopper" ]]; then
    uv pip install flash-attn==2.8.1 --no-build-isolation #v2.8.1
    # flash-attention 3.0.0
    cd hopper
    FLASH_ATTENTION_FORCE_BUILD="TRUE" FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" uv pip install . --no-build-isolation
    cd ../../ArchScale
else
    uv pip install flash-attn==2.8.1 --no-build-isolation #v2.8.1
    cd ../ArchScale
fi
# cd flash-attention/hopper 
# python setup.py install
# export PYTHONPATH=$PWD
# pytest -q -s test_flash_attn.py
# cd ../..
uv pip install nvidia-mathdx pybind11
# pip install --no-build-isolation transformer_engine[pytorch]
NVTE_FRAMEWORK=pytorch uv pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@v2.8
# uv pip install --no-build-isolation -e .

uv pip install azure-identity

pip install blobfile==3.1.0

# sonic-moe 
# commit 444fe874883efa9775f4636a6b4eb434c83837b1
uv pip install sonic-moe==0.1.0

uv pip install --no-build-isolation git+https://github.com/renll/quack.git
uv pip install nvidia-cutlass-dsl==4.4.1