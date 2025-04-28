#!/bin/bash

data_path="/root/autodl-tmp/finetune"
lr=5e-5
batch_size=32
accum_step=1

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

        for data in H2AFZ H3K27ac H3K27me3 H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K9ac H3K9me3 H4K20me1
        do
            python train.py \
                --dataPath "${data_path}/NT_revised" \
                --dataName "EMP/${data}" \
                --modelPath ${model} \
                --trainBatchSize ${batch_size} \
                --evalBatchSize ${batch_size} \
                --accumStep ${accum_step} \
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

        for data in enhancers enhancers_types
        do
            python train.py \
                --dataPath "${data_path}/NT_revised" \
                --dataName "Enhancer/${data}" \
                --modelPath ${model} \
                --trainBatchSize ${batch_size} \
                --evalBatchSize ${batch_size} \
                --accumStep ${accum_step} \
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

        for data in promoter_all promoter_tata promoter_no_tata
        do
            python train.py \
                --dataPath "${data_path}/NT_revised" \
                --dataName "Promoter/${data}" \
                --modelPath ${model} \
                --trainBatchSize ${batch_size} \
                --evalBatchSize ${batch_size} \
                --accumStep ${accum_step} \
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

        for data in splice_sites_all splice_sites_acceptors splice_sites_donors
        do
            python train.py \
                --dataPath "${data_path}/NT_revised" \
                --dataName "Splice/${data}" \
                --modelPath ${model} \
                --trainBatchSize ${batch_size} \
                --evalBatchSize ${batch_size} \
                --accumStep ${accum_step} \
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
    done
done