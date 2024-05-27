#!/bin/bash

# Runs the "Yuan-moe" parameter model inference

GPUS_PER_NODE=8
MAX_LENGTH=1024
MASTER_PORT=6000
MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

if [ "$TEMP" == "" ]; then
    TEMP=0
fi
if [ "$TOP_P" == "" ]; then
    TOP_P=0.0
fi
if [ "$TOP_K" == "" ]; then
    TOP_K=1
fi

CHECKPOINT_PATH=<Specify path>
TOKENIZER_MODEL_PATH=<Specify path>
MATH_DATA=<Specify path>
OUTPUT_PATH=<Specify path>

GPT_ARGS="
    --micro-batch-size 1 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 8 \
    --num-layers 24 \
    --hidden-size 2048 \
    --use-lf-gate \
    --rotary-base 40890 \
    --max-tokens-to-oom 16384 \
    --lf-conv2d-group 1 \
    --lf-conv2d-num-pad 0 \
    --position-embedding-type rope \
    --no-embedding-dropout \
    --use-flash-attn \
    --flash-attn-drop 0.0 \
    --attention-dropout 0 \
    --fim-rate 0.0 \
    --hidden-dropout 0 \
    --norm-dtype RMSNorm \
    --disable-bias-linear \
    --reset-position-ids \
    --swiglu \
    --num-attention-heads 16 \
    --seq-length 16384 \
    --max-position-embeddings 16384 \
    --no-async-tensor-model-parallel-allreduce \
    --bf16 \
    --kv-channels 256 \
    --num-attention-router-heads 16384 \
    --rotary-percent 0.5 \
    --use-attention-router \
    --no-masked-softmax-fusion \
    --use-fp32-router \
    --num-experts 32 \
    --moe-router-load-balancing-type none \
    --moe-router-topk 2 \
    --moe-grouped-gemm \
    --repetition-penalty 1.0 \
    --temp $TEMP \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --seed $RANDOM
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS tasks/MMLU/eval_for_mmlu.py \
       $GPT_ARGS \
       --tokenizer-type "YuanTokenizer" \
       --tokenizer-model-path $TOKENIZER_MODEL_PATH \
       --math_datapath ${MATH_DATA} \
       --distributed-backend nccl \
       --num_samples_per_task 1 \
       --max_len $MAX_LENGTH \
       --output_path $OUTPUT_PATH \
       --load $CHECKPOINT_PATH
