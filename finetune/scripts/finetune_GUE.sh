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

        for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
        do
            python train.py \
                --dataPath "${data_path}/GUE_plus" \
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

        for data in 0 1 2 3 4
        do
            python train.py \
                --dataPath "${data_path}/GUE_plus" \
                --dataName "mouse/${data}" \
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

        for data in covid
        do
            python train.py \
                --dataPath "${data_path}/GUE_plus" \
                --dataName "virus/${data}" \
                --modelPath ${model} \
                --trainBatchSize ${batch_size} \
                --evalBatchSize ${batch_size} \
                --accumStep ${accum_step} \
                --numWorker 4 \
                --totalStep 10000 \
                --evalStart 200 \
                --evalStep 200 \
                --precision "bf16-mixed" \
                --maxLR ${lr} \
                --minLR ${lr} \
                --warmup 100 \
                --coreMetric "F1" \
                --finetuneTask "classification" \
                --drop 0.1 \
                --nPrompt 4 \
                --dPrompt 64
        done

        for data in 0 1 2 3 4
        do
            python train.py \
                --dataPath "${data_path}/GUE_plus" \
                --dataName "tf/${data}" \
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

        for data in prom_300_all prom_300_tata prom_300_notata prom_core_all prom_core_tata prom_core_notata
        do
            python train.py \
                --dataPath "${data_path}/GUE_plus" \
                --dataName "prom/${data}" \
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

        for data in splice
        do
            python train.py \
                --dataPath "${data_path}/GUE_plus" \
                --dataName "${data}" \
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