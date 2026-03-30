export WANDB_PROJECT="slm-code"
export NCCL_DEBUG=WARN
export NCCL_MNNVL_ENABLE=2
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=2
export NCCL_IB_MERGE_NICS=1
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
export WANDB_ENTITY="<WANDB_ENTITY>"

source .venv/bin/activate

wandb login

echo "-------------Start Training--------------"
MASTER_ADDR="${MASTER_ADDR:-localhost:12345}"
nnodes="${NNODES:-1}"
logdir="${LOGDIR:-.}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
echo "MASTER_ADDR: $MASTER_ADDR with nnodes: $nnodes"

data_storage=<STORAGE_PATH>
model_storage=<STORAGE_PATH>
micro_bsz=16

## PRETRAIN ###
wandb_exp_run_name="<WANDB_EXP_RUN_NAME>"
remote_output_dir=$model_storage/checkpoints/$wandb_exp_run_name
local_data_dir=$data_storage/slim_for_tiny
input_data=$local_data_dir
val_data=$local_data_dir
train_model="transformer_gqa4"  #"samba" "swayoco"
save_step=4000
tokens=null
ctx_len=4096
wd=0.1
resume="auto"
bsz=2097152
bramp_ratio=1
save_mem=false
data_mixture=null
use_flce=true
norm_class=null
fp8=true
fp4=false
post_norm=null
num_extrp=4
min_lr_mult=0.0
wandb_project="discrete-llm"
ortho_init=false
diag_init=false

for depth in 8 12 16 20 24; do
  for wp in 0; do
    for tokens in null; do
      for aux_gamma in 1e-1; do
        for wd in 0; do
          for lr in 1e-2; do # coarse-grained optimal
            for sparsity in 8; do
              for topk in 8; do
                share_expert=true
                global_aux=true
                sqrt_gate=true
                post_norm=null
                wandb_project="muonh-moe-flops-scaling"
                fp8=false
                fsdp2=true
                aux_style="switch"
                min_lr_mult=0.1
                if [[ ${depth} -ge 24 ]]; then
                  micro_bsz=8
                fi                
                train_model="transformer_gqa4_h2_moe"
                act_ckpt=false
                if [[ ${depth} -ge 20 ]]; then
                  act_ckpt=true
                fi
                offloading=false
                base_tokens=10.4e9 # 50 TPP for chinchilla considering embedding parameters
                train_name="v2scale_mup_muonh_lr${lr}x${min_lr_mult}_wd${wd}_tok${tokens}_gblaux${aux_gamma}_ga_qknorm_sgate_shexp"
                USE_QUACK_GEMM=1 PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' LIGHTNING_ARTIFACTS_DIR=${remote_output_dir} torchrun --nnodes=${nnodes} \
                    --nproc_per_node=${GPUS_PER_NODE} --node-rank=$NODE_RANK --rdzv-endpoint=${MASTER_ADDR} pretrain.py \
                      --train_data_dir ${input_data} --val_data_dir ${val_data} --wandb_project ${wandb_project} \
                      --base_hps.eta0=${lr} --base_hps.warmup_tokens=${wp} --base_hps.weight_decay=${wd} --total_bsz=${bsz} \
                      --base_hps.t0=${base_tokens} --base_hps.min_lr_mult=${min_lr_mult} --train_model ${train_model} --depth ${depth} --max_tokens ${tokens} \
                      --train_name ${train_name} --data_mixture ${data_mixture} --resume=${resume} --save_step=${save_step} \
                      --fsdp_save_mem=${save_mem} --micro_bsz=${micro_bsz} --fp8=${fp8} --fp4=${fp4} --use_flce=${use_flce} --norm_class ${norm_class} \
                      --bramp_ratio=${bramp_ratio} --post_norm=${post_norm} \
                      --share_expert=${share_expert} --global_aux=${global_aux} --sqrt_gate=${sqrt_gate} \
                      --ctx_len=${ctx_len} --aux_style=${aux_style} --aux_gamma=${aux_gamma} \
                      --sparsity=${sparsity} --top_k=${topk} --num_extrp=${num_extrp} --fsdp2=${fsdp2} --act_ckpt=${act_ckpt} --offloading=${offloading} \
                      --ortho_init=${ortho_init} --diag_init=${diag_init} \
                      2>&1 | tee -a "$logdir/train_${wandb_exp_run_name}_${HOSTNAME}.log"
              done
            done
          done
        done
      done
    done
  done
done
