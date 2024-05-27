#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config

MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=<Specify path>
CHECKPOINT_PATH_SAVE=<Specify path>
LOG_PATH=<Specify path>
TOKENIZERPATH=<Specify path>
TENSORBOARD_PATH=<Specify path>


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --micro-batch-size 1 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 8 \
    --num-layers 24 \
    --hidden-size 2048 \
    --use-lf-gate \
    --lf-conv2d-group 1 \
    --lf-conv2d-num-pad 0 \
    --swiglu \
    --use-flash-attn \
    --position-embedding-type rope \
    --no-embedding-dropout \
    --flash-attn-drop 0.0 \
    --attention-dropout 0 \
    --fim-rate 0.0 \
    --hidden-dropout 0 \
    --norm-dtype RMSNorm \
    --disable-bias-linear \
    --reset-position-ids \
    --num-attention-heads 16 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
    --attention-projection-size 4096 \
    --num-attention-router-heads 4096 \
    --use-att-gating-method \
    --no-masked-softmax-fusion \
    --use-fp32-router \
    --num-experts 32 \
    --moe-router-load-balancing-type none \
    --moe-router-topk 2 \
    --moe-grouped-gemm \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 10,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000000 \
    --eval-iters 10
"
LOG_ARGS="
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-log-interval 1 \
    --tensorboard-queue-size 1000 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-world-size-to-tensorboard
"


torchrun $DISTRIBUTED_ARGS tools/convert_hf_moe.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $LOG_ARGS \
    --tokenizer-type "YuanTokenizer" \
    --tokenizer-model-path $TOKENIZERPATH \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH_SAVE \
    --load $CHECKPOINT_PATH

