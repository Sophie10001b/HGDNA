#!/bin/bash

data_path="/root/autodl-tmp"
lr=5e-5
batch_size=32
accum_step=2

echo "The provided data_path is $data_path"

model_list=(
    "./HGDNA"
    "LongSafari/hyenadna-medium-160k-seqlen-hf"
    "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    "zhihan1996/DNABERT-2-117M"
)

for seed in 17
do
    for model in "${model_list[@]}"
    do
        run_name="${model#*/}"
        run_name="${run_name}"

        echo "Now running ${run_name}"

        for data in gene_classification
        do
            python train.py \
                --dataPath "${data_path}/gener" \
                --dataName "${data}" \
                --modelPath ${model} \
                --trainBatchSize 32 \
                --evalBatchSize 16 \
                --accumStep 4 \
                --numWorker 4 \
                --totalStep 6000 \
                --evalStart 200 \
                --evalStep 200 \
                --precision "bf16-mixed" \
                --maxLR ${lr} \
                --minLR ${lr} \
                --warmup 100 \
                --coreMetric "MCC" \
                --finetuneTask "classification" \
                --drop 0.1 \
                --nPrompt 4 \
                --dPrompt 64
        done

        for data in taxomoic_classification
        do
            python train.py \
                --dataPath "${data_path}/gener" \
                --dataName "${data}" \
                --modelPath ${model} \
                --trainBatchSize 16 \
                --evalBatchSize 2 \
                --accumStep 16 \
                --numWorker 4 \
                --totalStep 3000 \
                --evalStart 200 \
                --evalStep 200 \
                --precision "bf16-mixed" \
                --maxLR ${lr} \
                --minLR ${lr} \
                --warmup 100 \
                --coreMetric "MCC" \
                --finetuneTask "classification" \
                --drop 0.1 \
                --nPrompt 4 \
                --dPrompt 64
        done
    done
done