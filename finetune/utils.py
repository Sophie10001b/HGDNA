import os
import argparse
import logging
import glob
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import lightning as pl
import transformers

from dataclasses import dataclass
from fla.modules import GatedMLP
from torch.optim.lr_scheduler import LRScheduler
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
from einops import rearrange
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.trainer.states import RunningStage

#########################################################
#            --- model trainer ---
#########################################################
class CosineLRSchedule(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup: Optional[int]=10000,
        maxLR: Optional[float]=3e-4,
        minLR: Optional[float]=1e-5,
        endStep: Optional[float]=150000
    ):
        self.warmup = warmup
        self.maxLR = maxLR
        self.minLR = minLR
        self.endStep = endStep

        super(CosineLRSchedule, self).__init__(optimizer)
    
    # override
    def get_lr(self) ->list[float]:
        step = max(1, self._step_count)
        if step <= self.warmup:
            scale = step / self.warmup
            return [min(lr * scale, self.maxLR) for lr in self.base_lrs]
        else:
            scale = (self.minLR + 0.5 * (self.maxLR - self.minLR) * \
                    (1.0 + math.cos(((step - self.warmup) / (max(self.endStep, step) - self.warmup)) * math.pi))) / self.maxLR
            if scale * self.maxLR < self.minLR:
                scale = self.minLR / self.maxLR
            return [min(lr * scale, self.maxLR) for lr in self.base_lrs]


#########################################################
#                 --- callback ---
#########################################################
def getLogging(logDir: str) -> logging.Logger:
    logger = logging.getLogger()
    logFile = logging.FileHandler(os.path.join(logDir, 'model.log'))
    logConsole = logging.StreamHandler()
    logFormat = logging.Formatter("%(name)-4s %(asctime)-1s: %(levelname)-4s %(message)s")
    logFormat.datefmt = "%m-%d %H:%M:%S"

    logFile.setFormatter(logFormat)
    logConsole.setFormatter(logFormat)

    logger.addHandler(logFile)
    logger.addHandler(logConsole)
    logger.setLevel(logging.INFO)
    return logger

@rank_zero_only
def process_json(jsonDir: str, tgt: Optional[list[dict]]=[], mode: Optional[str]="read"):
    if mode == "read":
        with open(jsonDir, 'r') as f: return json.load(f)
    elif mode == "write":
        with open(jsonDir, 'w') as f: f.write(json.dumps(tgt, indent=4))
    elif mode == "append":
        if not os.path.exists(jsonDir):
            with open(jsonDir, 'w') as f: f.write(json.dumps(tgt, indent=4))
        else:
            data = []
            with open(jsonDir, 'r') as f: data = json.load(f)
            data.extend(tgt)
            with open(jsonDir, 'w') as f: f.write(json.dumps(data, indent=4))

class CheckpointConfig:
    def __init__(
        self,
        coreMetric: str="MCC",
        topk: int=5,
        pt_prefix: str="",
        evalStart: int=200,
        evalStep: int=200,
        totalStep: int=1000,
        larger_is_bertter: bool=True,
        ensemble: bool=True,
        ensembleOnly: bool=False,
        finalTest: bool=True,
        finalSave: bool=True,
        **kwargs
    ):
        self.coreMetric = coreMetric
        self.topk = topk
        self.pt_prefix = pt_prefix
        self.evalStart = evalStart
        self.evalStep = evalStep
        self.totalStep = totalStep
        self.larger_is_bertter = larger_is_bertter
        self.ensemble = ensemble
        self.ensembleOnly = ensembleOnly
        self.finalTest = finalTest
        self.finalSave = finalSave


class CheckpointCallback(Callback):
    def __init__(self, config: CheckpointConfig, **kwargs):
        super().__init__(**kwargs)

        self.evalStart = config.evalStart
        self.evalStep = config.evalStep
        self.totalStep = config.totalStep

        assert config.topk > 0

        self.coreMetric = config.coreMetric
        self.topk = config.topk
        self.pt_prefix = config.pt_prefix
        self.larger_is_better = config.larger_is_bertter
        self.ensemble = config.ensemble
        self.ensembleOnly = config.ensembleOnly
        self.finalTest = config.finalTest
        self.finalSave = config.finalSave

        self.prev_step = -1
    
    def safe_eval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, process_stage: Optional[RunningStage]=RunningStage.VALIDATING):
        _first_loop_iter = trainer._logger_connector._first_loop_iter
        trainer.training = False
        pl_module.eval()
        stage = trainer.state.stage
        trainer.state.stage = process_stage

        if process_stage == RunningStage.TESTING: trainer.test_loop.run()
        else: trainer._run_stage()
        
        trainer.state.stage = stage
        trainer.training = True
        pl_module.train()
        trainer._logger_connector._epoch_end_reached = False
        trainer._logger_connector._first_loop_iter = _first_loop_iter
    
    @rank_zero_only
    def save(self, ckptName: str | int, trainer: pl.Trainer, pl_module: pl.LightningModule, record: Optional[bool]=True):
        pl_module._custom_save_state_dict(os.path.join(trainer.default_root_dir, f"{ckptName}.pt"))
        process_json(pl_module.ckptSummaryDir, [{"name": str(ckptName), "metrics": pl_module.evalCache}], mode="append")
    
    @rank_zero_only
    def _sort_and_remove(self, trainer: pl.Trainer, pl_module: pl.LightningModule, topk: Optional[int]=None, target: Optional[str]="", only_sort: Optional[bool]=False):
        current: list[dict] = process_json(pl_module.ckptSummaryDir, mode="read")
        if topk is None: topk = self.topk

        current = sorted(current, key=lambda x: x["metrics"][self.coreMetric], reverse=True if self.larger_is_better else False)
        if not only_sort:
            if target == "":
                if len(current) > topk:
                    for ckpt in current[topk:]:
                        removeTgt = os.path.join(trainer.default_root_dir, f"{ckpt["name"]}.pt")
                        if os.path.exists(removeTgt): os.remove(removeTgt)
                    
                    current = current[:topk]
                    process_json(pl_module.ckptSummaryDir, current, mode="write")
            else:
                tmp = []
                for ckpt in current:
                    if ckpt["name"] != target:
                        removeTgt = os.path.join(trainer.default_root_dir, f"{ckpt['name']}.pt")
                        if os.path.exists(removeTgt): os.remove(removeTgt)
                    else:
                        tmp.append(ckpt)
                current = tmp
                process_json(pl_module.ckptSummaryDir, current, mode="write")
            
        return current

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_step = trainer.global_step
        need_validate = (current_step >= self.evalStart and (current_step - self.evalStart) % self.evalStep == 0) or (current_step == self.totalStep)

        # for gradient accumulation
        need_validate = (current_step != self.prev_step) and need_validate
        self.prev_step = current_step

        if need_validate:
            self.safe_eval(trainer, pl_module)
            self.save(trainer.global_step, trainer, pl_module)
            pl_module.evalCache.clear()

            # check current checkpoints
            self._sort_and_remove(trainer, pl_module)
    
    def on_train_end(self, trainer, pl_module):
        # ensemble
        if self.ensemble:
            current: list[dict] = self._sort_and_remove(trainer, pl_module, only_sort=True)

            ensembleParam = None
            for i, _data in enumerate(current[:self.topk]):
                _name = _data["name"]
                modelPath = os.path.join(trainer.default_root_dir, f"{_name}.pt")
                param = torch.load(modelPath, map_location='cpu', weights_only=True)
                param = param[self.pt_prefix] if self.pt_prefix != "" else param

                if i == 0:
                    ensembleParam = param
                    for k, v in ensembleParam.items(): ensembleParam[k] = ensembleParam[k].float()
                else:
                    for k, v in ensembleParam.items(): ensembleParam[k].mul_(i).add_(param[k].float()).div_(i + 1)
            
            pl_module._custom_load_state_dict(state_dict=ensembleParam)
            self.safe_eval(trainer, pl_module)
            self.save("Ensemble", trainer, pl_module)
            pl_module.evalCache.clear()
        
        # get the best
        current: list[dict] = self._sort_and_remove(trainer, pl_module, topk=1, target="Ensemble" if self.ensembleOnly else "")
        
        finalCkpt = current[0]["name"]
        if trainer.is_global_zero: os.rename(os.path.join(trainer.default_root_dir, f"{finalCkpt}.pt"), os.path.join(trainer.default_root_dir, "final.pt"))
        current = {"name": "final", "metrics": current[0]["metrics"]}

        process_json(pl_module.ckptSummaryDir, [current], "write")

        # test set validation
        if self.finalTest:
            pl_module._custom_load_state_dict(os.path.join(trainer.default_root_dir, "final.pt"), self.pt_prefix)
            self.safe_eval(trainer, pl_module, RunningStage.TESTING)
            if not self.finalSave: os.remove(os.path.join(trainer.default_root_dir, "final.pt"))