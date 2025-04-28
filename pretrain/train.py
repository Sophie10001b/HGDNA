import os
import sys
import math
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataset
import lightning as pl
import transformers

from typing import Optional
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.trainer.states import RunningStage
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from nltk.translate.bleu_score import corpus_bleu
from transformers.generation import GenerationConfig

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from model.preprocess import PretrainDataModule, FinetuneDataModule, PackedData
from model.model import MLM, NTP
from model.linear_decoder import NTPConfig, NTPOutputWithPast
from model.utils import CosineLRSchedule, CheckpointCallback, getLogging, process_json
from model.preprocess import detokenizing

from peft import LoraConfig, get_peft_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class PLModel(pl.LightningModule):
    def __init__(self, model: nn.Module | NTP, ckptDir: str, args: argparse.Namespace, coreMetric: Optional[str]="MCC"):
        super(PLModel, self).__init__()

        self.save_hyperparameters(args)
        self.model = model

        self.finetuneSummaryDir = ""
        if ckptDir != "": self.finetuneSummaryDir = os.path.join(ckptDir, "finetune.json")
        
        self.evalCache = {}
        self.coreMetric = coreMetric

        if ckptDir != "":
            self._custom_load_state_dict(prefix=os.path.join(ckptDir, self.hparams.ckptName + ".pt"))
    
    @rank_zero_only
    def _custom_save_state_dict(self, prefix: str, dtype: Optional[torch.dtype]=torch.float32):
        torch.save({"model": self.model.to(dtype=dtype).state_dict()}, prefix)
    
    def _custom_load_state_dict(self, prefix: Optional[str]="", state_dict: Optional[dict]=None):
        if prefix != "":
            param = torch.load(prefix, map_location='cpu', weights_only=True)["model"]
        else:
            param = state_dict

        if prefix != "": print(f"Try loading state dict from {prefix}\n")
        try:
            self.model.load_state_dict(param, strict=True)
        except Exception as e:
            print(e)
            print("Model loading mismatch, try more flexible loading\n")
            self.model.load_state_dict(param, strict=False)
    
    @rank_zero_only
    def on_fit_start(self):
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
    
    def forward(self, data: PackedData):
        if self.hparams.trainMode == "pretrain":
            outputs: NTPOutputWithPast = self.model(
                input_ids=data.seq,
                labels=data.label,
                use_cache=False,
                return_dict=True,
                species_idx=data.speciesIdx,
                use_gradient_checkpoint=self.hparams.gradientCheckpoint
            )
            return {"Lmain": outputs.loss, "Lsub": outputs.subloss}
        elif self.hparams.finetuneTask == "classification":
            if self.trainer.state.stage == RunningStage.TRAINING:
                outputs: NTPOutputWithPast = self.model(
                    input_ids=data.seq,
                    labels=data.label,
                    use_cache=False,
                    return_dict=True,
                    prompt_idx=data.promptIdx,
                    augment=data.augment,
                    use_gradient_checkpoint=self.hparams.gradientCheckpoint
                )
                return {"Lmain": outputs.loss}
            else:
                outputs: NTPOutputWithPast = self.model(
                    input_ids=data.seq,
                    labels=data.label,
                    use_cache=False,
                    return_dict=True,
                    prompt_idx=data.promptIdx,
                    augment=data.augment,
                    use_gradient_checkpoint=False
                )
                return {"pred": outputs.pred, "label": outputs.label}
        elif self.hparams.finetuneTask == "generation":
            if self.trainer.state.stage in [RunningStage.TRAINING]:
                outputs: NTPOutputWithPast = self.model(
                    input_ids=torch.cat([data.seq, data.label], -1),
                    labels=data.label[:, 1:],
                    use_cache=False,
                    return_dict=True,
                    prompt_idx=data.promptIdx,
                    augment=data.augment,
                    use_gradient_checkpoint=self.hparams.gradientCheckpoint
                )
                return {"Lmain": outputs.loss}
            else:
                generate_config = GenerationConfig(
                    max_new_tokens=100 if self.trainer.state.stage == RunningStage.SANITY_CHECKING else self.hparams.maxGenerate,
                    min_new_tokens=100,
                    num_beams=1 if self.trainer.state.stage in [RunningStage.SANITY_CHECKING, RunningStage.VALIDATING] else self.hparams.beamWidth,
                    temperature=self.hparams.temperature,
                    use_cache=self.hparams.useCache,
                    num_return_sequences=1
                )
                outputs = self.model.generate(
                    torch.cat([data.seq, torch.full((data.seq.size(0), 1), self.model.config.bos_token_id, dtype=torch.int64, device=data.seq.device)], 1),
                    generation_config=generate_config,
                    prompt_idx=data.promptIdx
                )
                outputs = outputs[:, data.seq.size(1)+1:]
                padded_outputs = torch.full((outputs.size(0), self.hparams.maxGenerate), self.model.config.pad_token_id, dtype=torch.int64, device=outputs.device)
                padded_outputs[:, :outputs.size(1)] = outputs

                return {"pred": padded_outputs, "label": data.label[:, 1:]}
    
    def training_step(self, batch: PackedData, batch_idx):
        res: dict = self(batch)

        if "Lsub" in res and res["Lsub"] is None: res.pop("Lsub")

        res["seq"] = batch.seq.size(-1)
        res["bsz"] = batch.seq.size(0)
        res["lr"] = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log_dict(res, prog_bar=True, batch_size=len(batch))

        return res["Lmain"]
    
    def validation_step(self, batch: PackedData, batch_idx):
        res: dict = self(batch)
        if self.hparams.trainMode == "pretrain":
            if "Lmain" not in self.evalCache: self.evalCache["Lmain"] = torch.repeat_interleave(res["Lmain"].reshape(1), len(batch), 0)
            else: self.evalCache["Lmain"] = torch.cat([self.evalCache["Lmain"], torch.repeat_interleave(res["Lmain"].reshape(1), len(batch), 0)], 0)
        else:
            for key in res.keys():
                if key == "Lmain": continue
                if key not in self.evalCache: self.evalCache[key] = res[key]
                else: self.evalCache[key] = torch.cat([self.evalCache[key], res[key]], axis=0)

    def on_validation_epoch_start(self):
        self.evalCache.clear()
    
    @rank_zero_only
    def _record_metrics(self):
        if self.hparams.trainMode == "pretrain":
            self.evalCache["Lmain"] = self.evalCache["Lmain"].item()
            self.logger.log_metrics({"Lmain (eval)": self.evalCache["Lmain"]}, step=self.global_step)
            self.textLog.info(f"Step {self.global_step} eval finish, Lmain: {self.evalCache["Lmain"]:.6f}")
            self.evalCache["step"] = self.global_step

        elif self.hparams.trainMode == "finetune":
            self.evalCache["pred"], self.evalCache["label"] = self.evalCache["pred"].cpu().numpy(), self.evalCache["label"].cpu().numpy()

            if self.hparams.finetuneTask == "classification":
                self.evalCache["Accuracy"] = accuracy_score(self.evalCache["label"], self.evalCache["pred"])
                self.evalCache["F1"] = f1_score(self.evalCache["label"], self.evalCache["pred"], average='macro', zero_division=0)
                self.evalCache["MCC"] = matthews_corrcoef(self.evalCache["label"], self.evalCache["pred"])
                self.evalCache["Precision"] = precision_score(self.evalCache["label"], self.evalCache["pred"], average='macro', zero_division=0)
                self.evalCache["Recall"] = recall_score(self.evalCache["label"], self.evalCache["pred"], average='macro', zero_division=0)
            elif self.hparams.finetuneTask == "generation":
                pred = detokenizing(self.evalCache["pred"], self.hparams.tokenization, self.model.config.vocab, self.trainer.datamodule.data["train"].k, self.trainer.datamodule.data["train"].stride, self.trainer.datamodule.data["train"].tokenizeModel)
                label = detokenizing(self.evalCache["label"], self.hparams.tokenization, self.model.config.vocab, self.trainer.datamodule.data["train"].k, self.trainer.datamodule.data["train"].stride, self.trainer.datamodule.data["train"].tokenizeModel)

                pred, label = [list(_) for _ in pred], [[list(_)] for _ in label]
                self.evalCache["BLEU"] = corpus_bleu(label, pred, weights=tuple([1.0 / self.hparams.ngramBLEU for _ in range(self.hparams.ngramBLEU)]))
                    
            self.evalCache.pop("pred")
            self.evalCache.pop("label")
            
            if self.trainer.state.stage == RunningStage.VALIDATING:
                self.logger.log_metrics({self.coreMetric: self.evalCache[self.coreMetric]}, step=self.global_step)
                self.textLog.info(f"Step {self.global_step} eval finish, {self.coreMetric}: {self.evalCache[self.coreMetric]:.6f}")
                self.evalCache["step"] = self.global_step
        
            elif self.trainer.state.stage == RunningStage.TESTING:
                metrics = []
                for (k, v) in self.evalCache.items(): metrics.extend([f"{k}: {v:.6f}"])
                metrics = ",".join(metrics)
                self.textLog.info(f"Model {self.hparams.modelPrefix} test finish, {metrics}")
                process_json(self.testSummaryDir, [{"name": self.hparams.modelPrefix, "seed": self.hparams.seed, "metrics": self.evalCache}], mode="append")
                if self.finetuneSummaryDir != "":
                    process_json(self.finetuneSummaryDir, [{"name": self.hparams.modelPrefix, "seed": self.hparams.seed, "dataset": self.hparams.dataPath.replace('/', '_'), "metrics": self.evalCache}], mode="append")

    def on_validation_epoch_end(self):
        if self.hparams.trainMode == "pretrain":
            loss = self.evalCache["Lmain"]
            loss = self.all_gather(loss).mean()
            self.evalCache["Lmain"] = loss
        elif self.hparams.trainMode == "finetune":
            pred, label = self.evalCache["pred"], self.evalCache["label"]
            pred, label = self.all_gather(pred), self.all_gather(label)
            if self.trainer.num_devices > 1:
                pred, label = pred.flatten(0, 1), label.flatten(0, 1)

            self.evalCache["pred"], self.evalCache["label"] = pred, label
        
        self._record_metrics()
        
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
    
    def test_step(self, batch: PackedData, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def on_test_epoch_start(self):
        self.evalCache.clear()

    def on_train_epoch_start(self):
        if isinstance(self.trainer.datamodule, PretrainDataModule):
            if self.global_step > 0: self.trainer.datamodule.data["train"].getIndices()
            self.trainer.datamodule.data["train"]._step = self.global_step
    
    def on_train_epoch_end(self):
        if isinstance(self.trainer.datamodule, PretrainDataModule): self.trainer.datamodule._step = self.global_step


def main(args: argparse.Namespace):
    pl.seed_everything(args.seed, workers=True)
    startTime = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
    baseDir = os.path.dirname(os.path.realpath(__file__))
    modelName = args.modelPrefix
    ckptDir = "" if args.ckptPath == "" else os.path.join(baseDir, "result", args.ckptPath)
    saveDir = os.path.join(baseDir, "result", args.dataPath.replace("/", "_"), modelName)
    os.makedirs(saveDir, exist_ok=True)
    logDir = saveDir if not os.path.exists("/root/tf-logs") else "/root/tf-logs"
    dataDir = os.path.join(baseDir, "data", "pretrain", args.dataPath) if args.trainMode == "pretrain" else os.path.join(baseDir, "data", args.dataPath)
    
    coreMetric = "step" if args.trainMode == "pretrain" else "MCC"
    if "covid" in args.dataPath: coreMetric = "F1"
    elif args.finetuneTask == "generation": coreMetric = "BLEU"

    ckptCallback = CheckpointCallback(
        coreMetric=coreMetric,
        args=args,
        topk=5,
        larger_is_bertter=True,
        ensemble=False if args.trainMode == "pretrain" else True,
        finalTest=False if args.trainMode == "pretrain" else True,
        finalSave=False if args.trainMode == "finetune" else True
    )

    trainer = pl.Trainer(
        precision="bf16-mixed",
        logger=TensorBoardLogger(save_dir=logDir, name='_'.join([modelName, args.dataPath.replace("/", "_"), startTime])),
        max_steps=args.totalStep,
        default_root_dir=saveDir,
        callbacks=[ckptCallback, TQDMProgressBar(leave=False)],
        accumulate_grad_batches=args.accumStep,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=65536,
        enable_checkpointing=False,
        reload_dataloaders_every_n_epochs=1 if args.trainMode == "pretrain" else 0
    )

    args.generateBatchSize = args.batchSize if args.generateBatchSize < 1 else args.generateBatchSize

    if trainer.num_devices > 1:
        args.batchSize = args.batchSize // trainer.num_devices
        args.maxToken = args.maxToken // trainer.num_devices
        args.generateBatchSize = args.generateBatchSize // trainer.num_devices
    
    if trainer.accumulate_grad_batches > 1:
        args.batchSize = args.batchSize // trainer.accumulate_grad_batches
        args.maxToken = args.maxToken // trainer.accumulate_grad_batches
        args.seqLenWarmup *= trainer.accumulate_grad_batches

    dataModule = PretrainDataModule(dataDir, args) if args.trainMode == "pretrain" or args.trainMode == "finetune" else FinetuneDataModule(dataDir, args)

    if args.pretrainTask == "MLM":
        # model = MLM(args, dataModule.data["train"].classNum, dataModule.data["train"].vocab)
        raise ValueError("MLM Model is deprecated, please use NTP Model instead")
    elif args.pretrainTask == "NTP":
        model = NTP(NTPConfig(args, dataModule.vocab))
    
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

    plmodel = PLModel(model, ckptDir, args, coreMetric=coreMetric)

    trainer.fit(
        plmodel,
        datamodule=dataModule
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    dataParser = parser.add_argument_group("dataset")
    dataParser.add_argument("--dataPath", type=str, help="Path to the dataset")
    dataParser.add_argument("--modelPrefix", type=str, default="HybridGatedDeltaNet", help="Prefix of the model")
    dataParser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "test"])
    dataParser.add_argument("--numWorker", type=int, default=8, help="Number of workers for data loading")
    dataParser.add_argument("--batchSize", type=int, default=256, help="Batch size")
    dataParser.add_argument("--tokenization", type=str, default="base")
    dataParser.add_argument("--maxToken", type=int, default=102400, help="Maximum number of tokens each batch in pretrain")
    dataParser.add_argument("--seqLenWarmup", type=int, default=20000, help="Warmup step for reaching the 1st SLLength")
    dataParser.add_argument("--maxSeqLen", type=int, default=4096, help="Sequence length splitting in pretraining eval")
    dataParser.add_argument("--minSeqLen", type=int, default=80, help="Minimum sequence length splitting in pretraining eval")
    dataParser.add_argument("--maskRate", type=float, default=0.15, help="Mask rate for pretrain")
    dataParser.add_argument("--species", type=str, default="", help="Species for pretrain")
    dataParser.add_argument("--randomLenRate", type=float, default=0, help="Probobility for using random sequence length in pretraining")
    dataParser.add_argument("--speciesClassification", action="store_true", help="Species classification for pretrain")
    dataParser.add_argument("--pretrainTask", type=str, default="NTP", help="MLM or NTP for pretraining")
    dataParser.add_argument("--finetuneTask", type=str, default="classification", help="classification or generation for finetuning")
    dataParser.add_argument("--augment", action="store_true", help="Reverse complement augmentation")

    # model Args:
    modelParser = parser.add_argument_group("model")
    modelParser.add_argument("--dmodel", type=int, default=512, help="Dimension of the model")
    modelParser.add_argument("--ffnScale", type=int, default=4, help="Dimension of the feedforward layer (dmodel * ffnScale)")
    modelParser.add_argument("--nHead", type=int, default=8, help="Number of heads")
    modelParser.add_argument("--nLayer", type=int, default=8, help="Number of layers")
    modelParser.add_argument("--nPrompt", type=int, default=-1, help="Number of soft prompts")
    modelParser.add_argument("--dPrompt", type=int, default=64, help="Length of soft prompts")
    modelParser.add_argument("--drop", type=float, default=0.1, help="Dropout rate")
    modelParser.add_argument("--device", type=str, default='cuda:0', help="Device to use")
    modelParser.add_argument("--arch", type=str, default="full_attention")

    # training Args:
    trainParser = parser.add_argument_group("train")
    trainParser.add_argument("--saveName", type=str, default="")
    trainParser.add_argument("--trainMode", type=str, default="finetune", choices=["pretrain", "finetune"])
    trainParser.add_argument("--seed", type=int, default=17)
    trainParser.add_argument("--totalStep", type=int, default=20000)
    trainParser.add_argument("--evalStart", type=int, default=10000)
    trainParser.add_argument("--evalStep", type=int, default=1000)
    trainParser.add_argument("--accumStep", type=int, default=1)
    trainParser.add_argument("--maxLR", type=float, default=5e-4)
    trainParser.add_argument("--minLR", type=float, default=1e-6)
    trainParser.add_argument("--warmup", type=int, default=10000)
    trainParser.add_argument("--minLRStep", type=int, default=-1)
    trainParser.add_argument("--gradientCheckpoint", action="store_true")

    # evaling Args:
    evalParser = parser.add_argument_group("eval")
    evalParser.add_argument("--ckptPath", type=str, default="")
    evalParser.add_argument("--ckptName", type=str, default="EnsembleFinal")

    # task Args:
    taskParser = parser.add_argument_group("task")
    taskParser.add_argument("--epiTask", type=str, choices=["E2P", "P2E"], default="P2E")
    taskParser.add_argument("--generateBatchSize", type=int, default=0)
    taskParser.add_argument("--maxGenerate", type=int, default=4096, help="maximum generated sequences length in inference")
    taskParser.add_argument("--beamWidth", type=int, default=3)
    taskParser.add_argument("--temperature", type=float, default=1.0)
    taskParser.add_argument("--ngramBLEU", type=int, default=8)
    taskParser.add_argument("--useCache", action="store_true")

    args = parser.parse_args()

    main(args)
