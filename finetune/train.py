import os
import re
import sys
import math
import glob
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightning as pl
import transformers

from einops import repeat
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.trainer.states import RunningStage
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, root_mean_squared_error
from scipy.stats import pearsonr
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    CausalLMOutput,
    SequenceClassifierOutput
)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from utils import CosineLRSchedule, CheckpointCallback, CheckpointConfig, getLogging, process_json
from peft import LoraConfig, get_peft_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class PLModel(pl.LightningModule):
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.AutoTokenizer, modelName: str, args: argparse.Namespace):
        super(PLModel, self).__init__()

        self.save_hyperparameters(args)
        self.model = model
        self.tokenizer = tokenizer
        self.modelName = modelName
        
        self.evalCache = {}
        self.coreMetric = self.hparams.coreMetric
    
    @rank_zero_only
    def _custom_save_state_dict(self, dict_path: str, dtype: Optional[torch.dtype]=torch.float32):
        torch.save(self.model.to(dtype=dtype).state_dict(), dict_path)
    
    def _custom_load_state_dict(self, dict_path: Optional[str]="", prefix: Optional[str]="", state_dict: Optional[dict]=None):
        if dict_path != "":
            param = torch.load(dict_path, map_location='cpu', weights_only=True)
            if prefix != "": param = param[prefix]
        else:
            param = state_dict

        if dict_path != "": print(f"Try loading state dict from {dict_path}\n")
        try:
            self.model.load_state_dict(param, strict=True)
        except Exception as e:
            print(e)
            print("Model loading mismatch, try more flexible loading\n")
            self.model.load_state_dict(param, strict=False)
    
    @rank_zero_only
    def on_fit_start(self):
        self.finetuneSummaryDir = os.path.join(os.path.dirname(self.trainer.default_root_dir), "finetune.json")
        self.ckptSummaryDir = os.path.join(self.trainer.default_root_dir, "checkpoints.json")
        if os.path.exists(self.ckptSummaryDir): os.remove(self.ckptSummaryDir)
        self.testSummaryDir = os.path.join(os.path.dirname(self.trainer.default_root_dir), "test.json")

        if self.trainer.is_global_zero:
            prev_ckpt = glob.glob(os.path.join(self.trainer.default_root_dir, "*.pt"))
            for _dir in prev_ckpt: os.remove(_dir)

        self.textLog = getLogging(self.trainer.default_root_dir)

        self.textLog.info(f"--------------- Settings ---------------")
        for (k, v) in self.hparams.items(): self.textLog.info(f"{k}:\t {v}")
        self.textLog.info(f"------------- Architecture -------------")
        self.textLog.info(self.model)
        self.textLog.info(f"Total Param: {sum([p.numel() for p in self.model.parameters()])}")
        self.textLog.info(f"------------ Start Training ------------")
        self.__start = time.perf_counter()
    
    @rank_zero_only
    def on_fit_end(self):
        wall_clock = time.perf_counter() - self.__start
        self.textLog.info(f"Total wall-clock time: {wall_clock:.4f}")
        self.textLog.info("------------ End Training ------------")
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            self.hparams.maxLR,
            eps=1e-5,
            weight_decay=0.1,
            betas=(0.9, 0.98)
        )
        schedule = CosineLRSchedule(
            optimizer,
            warmup=self.hparams.warmup,
            maxLR=self.hparams.maxLR,
            minLR=self.hparams.minLR,
            endStep=self.hparams.totalStep
        )
        return [optimizer], [{"scheduler": schedule, "interval": "step", "frequency": 1}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
    
    def forward(self, data: Dict):
        if self.hparams.finetuneTask in ["classification", "regression"]:
            outputs: SequenceClassifierOutputWithPast = self.model(
                input_ids=data["input_ids"],
                labels=data["labels"],
                return_dict=True
            )
            return outputs
        elif self.hparams.finetuneTask == "embedding":
            outputs: SequenceClassifierOutputWithPast = self.model(
                input_ids=data["input_ids"],
                labels=None,
                return_dict=True,
                output_hidden_states=True
            )
            return outputs
        elif self.hparams.finetuneTask == "generation":
            if self.trainer.state.stage in [RunningStage.PREDICTING]:
                config = GenerationConfig(
                    max_new_tokens=self.hparams.maxGenerate,
                    num_beams=self.hparams.beamWidth,
                    temperature=self.hparams.temperature,
                    use_cache=True,
                    num_return_sequences=1,
                    repetition_penalty=1.2
                )
                inputs = data["input_ids"][:, :-1] # remove eos
                outputs: torch.LongTensor = self.model.generate(
                    inputs=inputs,
                    generation_config=config
                )
            else:
                outputs: CausalLMOutputWithPast = self.model(
                    input_ids=data["input_ids"],
                    labels=data["input_ids"],
                    return_dict=True,
                    prompt_idx=data["labels"]
                )
            return outputs
    
    def training_step(self, batch: Dict, batch_idx):
        res: SequenceClassifierOutputWithPast | CausalLMOutputWithPast = self(batch)

        record_dict = {}
        record_dict.update(
            loss=res.loss,
            seq=batch["input_ids"].size(-1),
            bsz=batch["input_ids"].size(0),
            lr=self.optimizers().optimizer.param_groups[0]["lr"]
        )
        self.log_dict(record_dict, prog_bar=True, batch_size=batch["input_ids"].size(0))

        return res.loss
    
    def validation_step(self, batch: Dict, batch_idx):
        res: SequenceClassifierOutputWithPast | CausalLMOutputWithPast | torch.LongTensor = self(batch)

        if isinstance(res, (SequenceClassifierOutputWithPast, SequenceClassifierOutput)):
            if "logits" not in self.evalCache: self.evalCache["logits"] = res.logits
            else: self.evalCache["logits"] = torch.cat([self.evalCache["logits"], res.logits], axis=0)
            
            if "label" not in self.evalCache: self.evalCache["label"] = batch["labels"]
            else: self.evalCache["label"] = torch.cat([self.evalCache["label"], batch["labels"]], axis=0)

            if res.hidden_states is not None and self.hparams.finetuneTask == "embedding":
                hidden_states = res.hidden_states[-1] if isinstance(res.hidden_states, Sequence) else res.hidden_states
                if hidden_states.dim() == 3: hidden_states = hidden_states.mean(dim=1)

                if "hidden_states" not in self.evalCache: self.evalCache["hidden_states"] = hidden_states
                else: self.evalCache["hidden_states"] = torch.cat([self.evalCache["hidden_states"], hidden_states], axis=0)
        
        elif isinstance(res, (CausalLMOutputWithPast, CausalLMOutput)):
            if "logits" not in self.evalCache: self.evalCache["logits"] = repeat(res.loss.exp().unsqueeze(0), "1 -> B", B=res.logits.size(0))
            else: self.evalCache["logits"] = torch.cat([self.evalCache["logits"], repeat(res.loss.exp().unsqueeze(0), "1 -> B", B=res.logits.size(0))], axis=0)

            if "label" not in self.evalCache: self.evalCache["label"] = batch["input_ids"][:, 1:]
            else: self.evalCache["label"] = torch.cat([self.evalCache["label"], batch["input_ids"][:, 1:]], axis=0)
        
        elif isinstance(res, torch.Tensor):
            res = self.tokenizer.batch_decode(res, skip_special_tokens=True)
            # remove prompt
            for i in range(len(res)): res[i] = res[i][len(batch["labels"][i]):]

            if "pred" not in self.evalCache: self.evalCache["pred"] = res
            else: self.evalCache["pred"] = self.evalCache["pred"] + res

            if "label" not in self.evalCache: self.evalCache["label"] = batch["labels"]
            else: self.evalCache["label"] = self.evalCache["label"] + batch["labels"]

        return None

    def on_validation_epoch_start(self):
        self.evalCache.clear()
    
    def on_validation_epoch_end(self):
        pred = self.evalCache["logits"].to(torch.float32) if self.hparams.finetuneTask in ["regression", "generation"] else self.evalCache["logits"].softmax(-1).argmax(-1)
        label = self.evalCache["label"]
        pred, label = self.all_gather(pred), self.all_gather(label)
        if self.trainer.num_devices > 1:
            pred, label = pred.flatten(0, 1), label.flatten(0, 1)

        self.evalCache.pop("logits")
        self.evalCache["pred"], self.evalCache["label"] = pred, label
        
        self._record_metrics()
    
    @rank_zero_only
    def _record_metrics(self):
        self.evalCache["pred"], self.evalCache["label"] = self.evalCache["pred"].cpu().numpy().flatten(), self.evalCache["label"].cpu().numpy().flatten()

        if self.hparams.finetuneTask == "classification":
            self.evalCache["Accuracy"] = accuracy_score(self.evalCache["label"], self.evalCache["pred"])
            self.evalCache["F1"] = f1_score(self.evalCache["label"], self.evalCache["pred"], average='macro', zero_division=0)
            self.evalCache["MCC"] = matthews_corrcoef(self.evalCache["label"], self.evalCache["pred"])
            self.evalCache["Precision"] = precision_score(self.evalCache["label"], self.evalCache["pred"], average='macro', zero_division=0)
            self.evalCache["Recall"] = recall_score(self.evalCache["label"], self.evalCache["pred"], average='macro', zero_division=0)
        
        elif self.hparams.finetuneTask == "regression":
            self.evalCache["RMSE"] = root_mean_squared_error(self.evalCache["label"], self.evalCache["pred"])
            self.evalCache["Pearson"] = float(pearsonr(self.evalCache["pred"], self.evalCache["label"])[0])
        
        elif self.hparams.finetuneTask == "generation":
            # self.evalCache["Accuracy"] = float(((self.evalCache["pred"] == self.evalCache["label"]).sum(-1) / self.evalCache["label"].shape[-1]).mean())
            self.evalCache["ppl"] = float(self.evalCache["pred"].mean())
                
        self.evalCache.pop("pred")
        self.evalCache.pop("label")
        
        if self.trainer.state.stage == RunningStage.VALIDATING:
            self.logger.log_metrics(self.evalCache, step=self.global_step)
            self.textLog.info(f"Step {self.global_step} eval finish, {self.hparams.coreMetric}: {self.evalCache[self.hparams.coreMetric]:.6f}")
            self.evalCache["step"] = self.global_step
    
        elif self.trainer.state.stage in [RunningStage.TESTING, RunningStage.PREDICTING]:
            metrics = []
            for (k, v) in self.evalCache.items(): metrics.extend([f"{k}: {v:.6f}"])
            metrics = ",".join(metrics)
            self.textLog.info(f"Model {self.modelName} test finish, {metrics}")
            process_json(self.testSummaryDir, [{"name": self.hparams.modelPath, "seed": self.hparams.seed, "metrics": self.evalCache}], mode="append")
            if self.finetuneSummaryDir != "":
                process_json(self.finetuneSummaryDir, [{"name": self.hparams.modelPath, "seed": self.hparams.seed, "dataset": self.hparams.dataName, "metrics": self.evalCache}], mode="append")
    
    def test_step(self, batch: Dict, batch_idx):
        self.validation_step(batch, batch_idx)
        return None
    
    def on_test_epoch_start(self):
        self.evalCache.clear()
        self.evalCache["wall_clock"] = time.perf_counter()
    
    def on_test_epoch_end(self):
        self.evalCache["wall_clock"] = time.perf_counter() - self.evalCache["wall_clock"]
        self.on_validation_epoch_end()
    
    def predict_step(self, batch: Dict, batch_idx):
        self.validation_step(batch, batch_idx)
        return None
    
    def on_predict_epoch_start(self):
        self.evalCache.clear()
        self.evalCache["wall_clock"] = time.perf_counter()

    def on_predict_epoch_end(self):
        self.evalCache["wall_clock"] = time.perf_counter() - self.evalCache["wall_clock"]

        # save emb to disk
        if self.hparams.finetuneTask == "embedding":
            pred, label = self.evalCache["hidden_states"], self.evalCache["label"]
            pred, label = self.all_gather(pred), self.all_gather(label)
            if self.trainer.num_devices > 1:
                pred, label = pred.flatten(0, 1), label.flatten(0, 1)

            self.evalCache["pred"], self.evalCache["label"] = pred.cpu(), label.cpu()

            record_dict = {}
            record_dict.update(
                wall_clock=self.evalCache["wall_clock"],
                pred=pred.cpu(),
                label=label.cpu()
            )
            torch.save(record_dict, os.path.join(self.trainer.default_root_dir, "embeddings.emb"))
        
        # save generated to disk
        elif self.hparams.finetuneTask == "generation":
            pred, label = self.evalCache["pred"], self.evalCache["label"]

            record_dict = {}
            record_dict.update(
                wall_clock=self.evalCache["wall_clock"],
                pred=pred,
                label=label
            )
            torch.save(record_dict, os.path.join(self.trainer.default_root_dir, "generation.emb"))


#########################################################
#                   --- dataset ---
#########################################################
class FinetuneData(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.AutoTokenizer, 
        split: str="", 
        num_seqs: int=1, 
        label_offset: int=0,
        args: argparse.Namespace=None
    ):
        super().__init__()

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.num_seqs = num_seqs
        self.num_labels = 0
        self.label_offset = label_offset

        self.args = args

        # loading
        if split != "":
            candidate = glob.glob(self.data_path + f"/{split}.*", recursive=True)
            tgt = [_ for _ in candidate if _.endswith((".csv", ".parquet"))][0]
            rawData = pd.read_csv(tgt) if tgt.endswith(".csv") else pd.read_parquet(tgt)
        else:
            rawData = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_parquet(data_path)

        data = []
        label = []
        for row in rawData.itertuples(index=False):
            seqs = []
            for _ in row[:self.num_seqs]:
                seq = re.sub(r"[^ATCG]", "", _.strip().upper())
                seqs.append(seq)
            data.append(tuple([*seqs, row[-1]]))
            label.append(row[-1])
        
        self.num_labels = max(self.num_labels, len(set(label)))
        self.valid_labels = list(set(label)) if isinstance(label[0], int) else None
        self.data = np.array(data, dtype=object)
    
    def process(self, datas: Sequence[Tuple]):
        texts, labels = tuple(tuple(_[:-1]) for _ in datas), tuple(_[-1] for _ in datas)

        pair_texts = None
        if len(texts[0]) > 1:
            texts, pair_texts = tuple([_[i] for _ in texts] for i in range(0, 2))
        else:
            texts = tuple(_[0] for _ in texts)
        
        output = self.tokenizer(
            texts, pair_texts,
            return_tensors="pt",
            padding="longest",
            max_length=int(1e6),
            truncation=True
        )
        return dict(
            input_ids=output["input_ids"],
            labels=list(labels) if isinstance(labels[0], str) else torch.tensor([_ + self.label_offset for _ in labels], dtype=torch.int64 if isinstance(labels[0], int) else torch.float32)
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    

class FinetuneDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_path: str, 
        tokenizer: transformers.AutoTokenizer, 
        num_seqs: int, 
        label_offset: int,
        args: argparse.Namespace, **kwargs
    ):
        super().__init__(**kwargs)

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.num_seqs = num_seqs
        self.label_offset = label_offset
        self.args = args

        self.data = {}
        self.data.update(
            train=FinetuneData(self.data_path, self.tokenizer, "train", self.num_seqs, self.label_offset, self.args),
            eval=FinetuneData(self.data_path, self.tokenizer, "dev", self.num_seqs, self.label_offset, self.args),
            test=FinetuneData(self.data_path, self.tokenizer, "test", self.num_seqs, self.label_offset, self.args)
        )
        self.num_labels = self.data["train"].num_labels
        if self.data["train"].valid_labels is not None:
            print(f"Valid labels: {",".join([str(_) for _ in self.data["train"].valid_labels])}")
    
    def setup(self, stage):
        pass
    
    def train_dataloader(self):
        return DataLoader(
            self.data["train"], shuffle=True,
            batch_size=self.args.trainBatchSize,
            collate_fn=self.data["train"].process,
            num_workers=self.args.numWorker,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data["eval"],
            batch_size=self.args.evalBatchSize,
            collate_fn=self.data["eval"].process,
            num_workers=self.args.numWorker
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.data["eval"],
            batch_size=self.args.evalBatchSize,
            collate_fn=self.data["eval"].process,
            num_workers=self.args.numWorker
        )


def main(args: argparse.Namespace):
    pl.seed_everything(args.seed, workers=True)
    startTime = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
    basePath = os.path.dirname(os.path.realpath(__file__))

    modelPath = args.modelPath
    modelName = modelPath.split("/")[-1]
    dataName = args.dataName.replace("/", "_").split(".")[0]
    dataPath = os.path.join(args.dataPath, args.dataName)

    args.modelPath = modelPath
    args.dataPath = dataPath
    args.dataName = dataName

    savePath = os.path.join(basePath, "result", modelName, dataName)

    os.makedirs(savePath, exist_ok=True)
    logPath = savePath if not os.path.exists("/root/tf-logs") else "/root/tf-logs"

    # load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(modelPath, model_max_length=int(1e6), padding_side="right", use_fast=True, trust_remote_code=True)
    modelConfig = transformers.AutoConfig.from_pretrained(modelPath, trust_remote_code=True)
    if "InstaDeepAI" in modelPath:
        tokenizer.eos_token = tokenizer.pad_token
    
    num_labels = 0
    label_offset = 0
    if "HGDNA" in args.modelPath: label_offset = modelConfig.class_ids_start
    if args.mode == "predict":
        predictData = FinetuneData(dataPath, tokenizer, "", 1, label_offset, args)
        predictDataset = DataLoader(
            predictData,
            batch_size=args.evalBatchSize,
            collate_fn=predictData.process,
            num_workers=args.numWorker
        )
        num_labels = 2
    else:
        trainDatamodule = FinetuneDataModule(dataPath, tokenizer, 1, label_offset, args)
        predictDataset = None
        num_labels = trainDatamodule.num_labels
    
    if args.finetuneTask in ["classification", "embedding", "regression"]:
        if "HGDNA" in args.modelPath:
            modelConfig.num_prompts = args.nPrompt
            modelConfig.prompt_size = args.dPrompt
            modelConfig.output_cls_states = False
            modelConfig.causal = False
        
        modelConfig.dropout_prob = 0.0 if args.finetuneTask == "regression" else args.drop
        if args.finetuneTask == "regression": modelConfig.problem_type = "regression"

        modelConfig.num_labels = 1 if args.finetuneTask == "regression" else num_labels
        model = transformers.AutoModelForSequenceClassification.from_pretrained(modelPath, trust_remote_code=True, config=modelConfig)
    
    elif args.finetuneTask in ["generation"]:
        if "HGDNA" in args.modelPath or "hyena" in args.modelPath:
            modelConfig.num_prompts = args.nPrompt
            modelConfig.prompt_size = args.dPrompt
            
            if "HGDNA" in args.modelPath:
                modelConfig.output_cls_states = False
                modelConfig.causal = True
        
        modelConfig.num_labels = num_labels
        modelConfig.dropout_prob = args.drop
        model = transformers.AutoModelForCausalLM.from_pretrained(modelPath, trust_remote_code=True, config=modelConfig)

    ckptCallback = CheckpointCallback(
        CheckpointConfig(
            coreMetric=args.coreMetric,
            topk=5,
            pt_prefix="",
            evalStart=args.evalStart,
            evalStep=args.evalStep,
            totalStep=args.totalStep,
            larger_is_bertter=not args.smallIsBetter,
            finalSave=args.finalSave
        )
    )

    trainer = pl.Trainer(
        precision=args.precision,
        logger=TensorBoardLogger(save_dir=logPath, name='_'.join([modelName, dataName, startTime])),
        max_steps=args.totalStep,
        default_root_dir=savePath,
        callbacks=[ckptCallback, TQDMProgressBar(leave=False)],
        accumulate_grad_batches=args.accumStep,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=65536,
        enable_checkpointing=False,
        reload_dataloaders_every_n_epochs=0,
        limit_val_batches=args.limitValBatch
    )

    if trainer.num_devices > 1:
        args.trainBatchSize = args.trainBatchSize // trainer.num_devices
        args.evalBatchSize = args.evalBatchSize // trainer.num_devices
    
    if trainer.accumulate_grad_batches > 1:
        args.trainBatchSize = args.trainBatchSize // trainer.accumulate_grad_batches
    
    if args.loraRank > 1:
        lora_config = LoraConfig(
            r=args.loraRank,
            lora_alpha=args.loraAlpha,
            target_modules="all-linear",
            lora_dropout=args.drop,
            bias="none",
            inference_mode=False
        )
        model = get_peft_model(model, lora_config)

    plmodel = PLModel(model, tokenizer, modelName, args)
    if args.ckptPath != "": plmodel._custom_load_state_dict(args.ckptPath)

    if args.mode == "predict":
        trainer.predict(
            plmodel,
            dataloaders=predictDataset
        )
    else:
        trainer.fit(
            plmodel,
            datamodule=trainDatamodule
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    dataParser = parser.add_argument_group("paths")
    dataParser.add_argument("--dataPath", type=str, help="Path to the dataset dir")
    dataParser.add_argument("--dataName", type=str, help="Name of the dataset")
    dataParser.add_argument("--modelPath", type=str, help="Path to the model")
    dataParser.add_argument("--ckptPath", type=str, default="", help="Path to the finetuned checkpoint")
    dataParser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "test", "predict"])
    dataParser.add_argument("--trainBatchSize", type=int, default=32)
    dataParser.add_argument("--evalBatchSize", type=int, default=32)
    dataParser.add_argument("--numWorker", type=int, default=4)

    # model Args:
    modelParser = parser.add_argument_group("model")
    modelParser.add_argument("--nPrompt", type=int, default=-1, help="Number of soft prompts")
    modelParser.add_argument("--dPrompt", type=int, default=64, help="Length of soft prompts")
    modelParser.add_argument("--drop", type=float, default=0.0, help="Dropout rate")

    # training Args:
    trainParser = parser.add_argument_group("train")
    trainParser.add_argument("--seed", type=int, default=17)
    trainParser.add_argument("--totalStep", type=int, default=20000)
    trainParser.add_argument("--evalStart", type=int, default=10000)
    trainParser.add_argument("--evalStep", type=int, default=1000)
    trainParser.add_argument("--accumStep", type=int, default=1)
    trainParser.add_argument("--precision", type=str, default="bf16-mixed")
    trainParser.add_argument("--maxLR", type=float, default=5e-4)
    trainParser.add_argument("--minLR", type=float, default=1e-6)
    trainParser.add_argument("--warmup", type=int, default=10000)
    trainParser.add_argument("--gradientCheckpoint", action="store_true")
    trainParser.add_argument("--limitValBatch", type=float, default=1.0)

    # task Args:
    taskParser = parser.add_argument_group("task")
    taskParser.add_argument("--coreMetric", type=str)
    taskParser.add_argument("--smallIsBetter", action="store_true")
    taskParser.add_argument("--finetuneTask", type=str, default="classification", help="classification or generation for finetuning")
    taskParser.add_argument("--finalSave", action="store_true", help="save the final ckpt in SFT")
    taskParser.add_argument("--ensembleOnly", action="store_true", help="only use ensemble ckpt as the final model")
    taskParser.add_argument("--maxGenerate", type=int, default=4096, help="maximum generated sequences length in inference")
    taskParser.add_argument("--beamWidth", type=int, default=3)
    taskParser.add_argument("--temperature", type=float, default=1.0)
    taskParser.add_argument("--ngramBLEU", type=int, default=8)
    taskParser.add_argument("--useCache", action="store_true")

    # lora Args:
    loraParser = parser.add_argument_group("lora")
    loraParser.add_argument("--loraRank", type=int, default=-1)
    loraParser.add_argument("--loraAlpha", type=int, default=16)
    loraParser.add_argument("--loraDropout", type=float, default=0.1)

    args = parser.parse_args()
    main(args)