#!/bin/bash

data_path="/home/sophie/model/HGDNA_hf"
batch_size=8

echo "The provided data_path is $data_path"

model_list=(
    "./HGDNA"
    "LongSafari/hyenadna-medium-160k-seqlen-hf"
    "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
    "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    "zhihan1996/DNABERT-2-117M"
)

model_list=(
    "./HGDNA"
)

for seed in 17
do
    for model in "${model_list[@]}"
    do
        run_name="${model#*/}"
        run_name="${run_name}"

        echo "Now running ${run_name}"

        for data in species_1024 species_16384 species_32768
        do
            python train.py \
                --dataPath "${data_path}/species_classification" \
                --dataName "${data}_10000_1000_1000/train.parquet" \
                --modelPath ${model} \
                --mode "predict" \
                --trainBatchSize ${batch_size} \
                --evalBatchSize ${batch_size} \
                --numWorker 4 \
                --precision "bf16-mixed" \
                --coreMetric "MCC" \
                --finetuneTask "embedding" \
                --drop 0.0 \
                --nPrompt -1 \
                --dPrompt 64

            python train.py \
                --dataPath "${data_path}/species_classification" \
                --dataName "${data}_10000_1000_1000/test.parquet" \
                --modelPath ${model} \
                --mode "predict" \
                --trainBatchSize ${batch_size} \
                --evalBatchSize ${batch_size} \
                --numWorker 4 \
                --precision "bf16-mixed" \
                --coreMetric "MCC" \
                --finetuneTask "embedding" \
                --drop 0.0 \
                --nPrompt -1 \
                --dPrompt 64
        done
    done
done