## Here is the code for HGDNA
Beyond the Bases: Unleashing Overlapping DNA Tokenization via Unified Linear-Time Autoregressive

### Introduction
HGDNA is a 30M long-context friendly DNA language model which combines standard sliding window attention and Gated DeltaNet (linear attention), pretrained on 13B multi-species gene sequences. It combines the overlapping-6mer tokenization with autoregressive training, providing capabilities for both common short-range & long-range sequence classification, regression, zero-shot embedding, and soft-prompt-based generation.

### Huggingface Available
HGDNA is naturally built with Huggingface Transformers, i.e., wrapped in `PretrainedModelForCausalLM` and `GenerationMixin`. You can easily to access it from Huggingface Hub (url).

### Dependencies
HGDNA is built on pytorch, the basic dependencies are as follows:
```
python = 3.12.2
pytorch = 2.5.1
transformers = 4.49.0
pytorch-lightning = 2.5.0.post0
```

To run HGDNA successfully, the following kernel libraries are required:

- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- [Causal Conv1D](https://github.com/Dao-AILab/causal-conv1d)

### Pretraining Corpus & Pretraining
We collect the raw Refseq dataset from NCBI Refseq, you can modify and execute the `ftp_download.py` and `generate_pretrain.py` in `./pretrain/collect_corpus` to collect the target data's url and process with multi-threads, which generates the `final_pretrain.fa` as pretraining corpus.

After obtaining the corpus, you can paste the `final_pretrain.fa` into `./pretrain/data/pretrain` and then run
```
bash ./pretrain/scripts/train_mtsp.sh
```
to start pretraining. For the basic version of HGDNA, it requires 1 A100-40G to pretrain for 14 hours with 200k steps.

### Fine-tuning
We provide the corresponding fine-tuning scripts and checkpoints schedule callback in `./finetune`, you need to download the raw dataset from different sources and pre-process them to obtain the train/val/test splits. Here are the download links:

- [GUE](https://github.com/MAGICS-LAB/DNABERT_2)
- [NT (revised)](https://hf-mirror.com/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised)
- [gener tasks](https://huggingface.co/datasets/GenerTeam/gener-tasks)
- [vertebrate species classification (same as HyenaDNA)](./finetune/data/species_classification/download.sh)
- [Enhancer Regression & Generation](https://huggingface.co/datasets/GenerTeam/DeepSTARR-enhancer-activity)

The corresponded pre-process scripts are available in `./finetune/data/NT` for splitting NT (revised), `./finetune/data/gener_task` for gener, `./finetune/data/species_classification` for splitting species sequences into 1k, 16k, and 32k chunks, `./finetune/data/DeepSTARR-enhancer-activity` for generating splitted datasets for regression and generation