#!/bin/bash

data_path="/root/autodl-tmp/finetune"
lr=5e-5
batch_size=32
accum_step=1

echo "The provided data_path is $data_path"

# model_list=(
#     "./HGDNA"
#     "LongSafari/hyenadna-medium-160k-seqlen-hf"
#     "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
#     "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
#     "zhihan1996/DNABERT-2-117M"
# )

model_list=(
    "./HGDNA"
    "./hyenadna-medium-160k-seqlen-hf"
    # "LongSafari/hyenadna-medium-160k-seqlen-hf"
    # "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
)

# -m debugpy --listen localhost:5678 --wait-for-client

for seed in 17
do
    for model in "${model_list[@]}"
    do
        run_name="${model#*/}"
        run_name="${run_name}"

        echo "Now running ${run_name}"

        for data in Developmental HouseKeeping
        do
            python train.py \
                --dataPath "${data_path}/enhancer_activity_regression" \
                --dataName "${data}" \
                --modelPath ${model} \
                --trainBatchSize ${batch_size} \
                --evalBatchSize ${batch_size} \
                --accumStep ${accum_step} \
                --numWorker 4 \
                --totalStep 40000 \
                --evalStart 200 \
                --evalStep 200 \
                --precision "bf16-mixed" \
                --maxLR ${lr} \
                --minLR ${lr} \
                --warmup 100 \
                --coreMetric "Pearson" \
                --finetuneTask "regression" \
                --drop 0.1 \
                --nPrompt 4 \
                --dPrompt 64 \
                --finalSave \
                --limitValBatch 0.2
        done

        for data in Developmental HouseKeeping
        do
            python train.py \
                --dataPath "${data_path}/enhancer_activity_generation" \
                --dataName "${data}" \
                --modelPath ${model} \
                --trainBatchSize ${batch_size} \
                --evalBatchSize ${batch_size} \
                --accumStep ${accum_step} \
                --numWorker 4 \
                --totalStep 40000 \
                --evalStart 200 \
                --evalStep 500 \
                --precision "bf16-mixed" \
                --maxLR ${lr} \
                --minLR ${lr} \
                --warmup 100 \
                --coreMetric "ppl" \
                --finetuneTask "generation" \
                --drop 0.1 \
                --nPrompt 64 \
                --dPrompt 64 \
                --finalSave \
                --limitValBatch 0.2
        done
    done
done