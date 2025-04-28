#!/bin/bash
nproc=1

arch="hybrid_gated_delta_net"
tokenization="6mer_1stride"
pretrainTask="NTP"
model_prefix="${arch}_${tokenization}_${pretrainTask}"

data_path="final_pretrain.fa"
total_step=200000
eval_start=100000
eval_step=5000
max_lr=5e-4
min_lr=0
warmup=10000
seqLenWarmup=40000
batchSize=128
accumStep=1
maxToken=98304
maxSeqLen=2048
drop=0.1
randomLenRate=0.1
numWorker=4

dmodel=512
nhead=8
nlayer=6

torchrun --nproc_per_node=${nproc} train.py \
    --modelPrefix ${model_prefix} \
    --mode train \
    --trainMode pretrain \
    --tokenization ${tokenization} \
    --dataPath ${data_path} \
    --batchSize ${batchSize} \
    --accumStep ${accumStep} \
    --numWorker ${numWorker} \
    --maxToken ${maxToken} \
    --maxSeqLen ${maxSeqLen} \
    --randomLenRate ${randomLenRate} \
    --pretrainTask ${pretrainTask} \
    --augment \
    --speciesClassification \
    --dmodel ${dmodel} \
    --nHead ${nhead} \
    --nLayer ${nlayer} \
    --drop ${drop} \
    --totalStep ${total_step} \
    --evalStart ${eval_start} \
    --evalStep ${eval_step} \
    --maxLR ${max_lr} \
    --minLR ${min_lr} \
    --warmup ${warmup} \
    --minLRStep ${total_step} \
    --seqLenWarmup ${seqLenWarmup} \
    --arch ${arch}