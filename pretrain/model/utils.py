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
#            --- model initialize ---
#########################################################
def setSeed(seed: int) ->None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

@torch.no_grad()
def embeddingInit(embedding: nn.Embedding) ->None:
    fan_out = embedding.weight.size(1)
    std = 1.0 * math.sqrt(1.0 / float(fan_out))
    nn.init.normal_(embedding.weight, 0., std)
    if embedding.padding_idx is not None:
        embedding.weight[embedding.padding_idx].fill_(0)

@torch.no_grad()
def linearInit(
    linear: nn.Linear,
    distribution: Optional[str]='normal',
    zero_bias: Optional[bool]=False,
    gain: Optional[float]=1.0
) ->None:
    if distribution == 'normal':
        nn.init.xavier_normal_(linear.weight, gain=gain)
    elif distribution == 'uniform':
        nn.init.xavier_uniform_(linear.weight, gain=gain)
    if linear.bias is not None:
        if zero_bias:
            nn.init.zeros_(linear.bias)
        else:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(linear.bias, -bound, bound)

def getDeepNorm(nEncLayer: int, nDecLayer: int):
    encParam = {"gain": 1.0, "alpha": 1.0}
    decParam = {"gain": 1.0, "alpha": 1.0}

    encParam["gain"] = 0.87 * (((nEncLayer ** 4) * nDecLayer) ** (-1 / 16))
    encParam["alpha"] = 0.81 * (((nEncLayer ** 4) * nDecLayer) ** (1 / 16))
    decParam["gain"] = (12 * nDecLayer) ** (-1 / 4)
    decParam["alpha"] = (3 * nDecLayer) ** (1 / 4)
    return encParam, decParam

#########################################################
#            --- model architecture ---
#########################################################
class RMSNorm(nn.Module):
    def __init__(self, dmodel: int, transpose: Optional[bool]=False, eps: Optional[float]=1e-5):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.W = nn.Parameter(torch.ones((dmodel), dtype=torch.float32))
        self.transpose = transpose
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if self.transpose: src = rearrange(src, "B D L -> B L D")
        src = src * torch.rsqrt(torch.square(src).mean(-1, keepdim=True) + self.eps)
        return rearrange(src * self.W, "B L D -> B D L") if self.transpose else src * self.W

class GLU(nn.Module):
    def __init__(self, dmodel: int, dff: int, drop: Optional[float]=0.1):
        super(GLU, self).__init__()

        self.Win = nn.Linear(dmodel, dff*2, bias=True)
        self.Wout = nn.Linear(dff, dmodel, bias=True)
        self.glu = nn.GLU()
        self.drop = nn.Dropout(drop)

        self.initParam()
    
    def initParam(self):
        linearInit(self.Win)
        linearInit(self.Wout)
    
    def forward(self, src: torch.Tensor) ->torch.Tensor:
        uv = self.glu(self.Win(src))
        out = self.drop(uv)
        out = self.Wout(out)
        return out

# class GLU(nn.Module):
#     def __init__(self, dmodel: int, dff: int, drop: Optional[float]=0.1):
#         super(GLU, self).__init__()

#         self.gateMLP = GatedMLP(hidden_size=dmodel, intermediate_size=dff)
#         self.initParam()
    
#     def initParam(self):
#         for m in self.gateMLP.modules():
#             if isinstance(m, nn.Linear): linearInit(m, zero_bias=True)
    
#     def forward(self, src: torch.Tensor) ->torch.Tensor:
#         return self.gateMLP(src)

class CrossEntropy(nn.Module):
    def __init__(self, gamma: float=2.0):
        """
        input:
        src ->FloatTensor before softmax, size(batch, class)
        tgt ->LongTensor, size(batch,)
        """
        super(CrossEntropy, self).__init__()
        
        self.gamma = gamma
        self.critertion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) ->torch.Tensor:
        numClass = src.size(-1)
        srcSoftmax = torch.softmax(src, -1)
        loss = self.critertion(src, tgt)

        if self.gamma > 0.0:
            occurIdx = torch.arange(start=0, end=src.numel(), step=numClass, dtype=torch.int64, device=src.device) + tgt
            pOccur = srcSoftmax.flatten().index_select(-1, occurIdx).reshape_as(tgt)
            loss = loss * ((1 - pOccur) ** self.gamma)
            
        return loss.mean()

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


class CheckpointCallback(Callback):
    def __init__(self, coreMetric: str, args: argparse.Namespace, topk: Optional[int]=5, larger_is_bertter: Optional[bool]=True, ensemble: Optional[bool]=True, finalTest: Optional[bool]=True, finalSave: Optional[bool]=True):
        super(CheckpointCallback, self).__init__()

        self.evalStart = args.evalStart
        self.evalStep = args.evalStep
        self.totalStep = args.totalStep

        assert topk > 0

        self.coreMetric = coreMetric
        self.topk = topk
        self.larger_is_better = larger_is_bertter
        self.ensemble = ensemble
        self.finalTest = finalTest
        self.finalSave = finalSave

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
    def _sort_and_remove(self, trainer: pl.Trainer, pl_module: pl.LightningModule, topk: Optional[int]=None, only_sort: Optional[bool]=False):
        current: list[dict] = process_json(pl_module.ckptSummaryDir, mode="read")
        if topk is None: topk = self.topk

        current = sorted(current, key=lambda x: x["metrics"][self.coreMetric], reverse=True if self.larger_is_better else False)
        if not only_sort:
            if len(current) > topk:
                for ckpt in current[topk:]:
                    removeTgt = os.path.join(trainer.default_root_dir, f"{ckpt["name"]}.pt")
                    if os.path.exists(removeTgt): os.remove(removeTgt)
                
                current = current[:topk]
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
                param = torch.load(modelPath, map_location='cpu', weights_only=True)["model"]

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
        current: list[dict] = self._sort_and_remove(trainer, pl_module, topk=1)
        
        finalCkpt = current[0]["name"]
        if trainer.is_global_zero: os.rename(os.path.join(trainer.default_root_dir, f"{finalCkpt}.pt"), os.path.join(trainer.default_root_dir, "final.pt"))
        current = {"name": "final", "metrics": current[0]["metrics"]}

        process_json(pl_module.ckptSummaryDir, [current], "write")

        # test set validation
        if self.finalTest:
            pl_module._custom_load_state_dict(os.path.join(trainer.default_root_dir, "final.pt"))
            self.safe_eval(trainer, pl_module, RunningStage.TESTING)
            if not self.finalSave: os.remove(os.path.join(trainer.default_root_dir, "final.pt"))


#########################################################
#                 --- Generation ---
#########################################################
class Cache(transformers.cache_utils.Cache):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    **Input:**
        - recurrent_state: Cache for linear attention, size(bsz, nhead, k_dim, v_dim) for deltanet of size(bsz, nhead, dhead, dstate) for mamba2
        - attn_state: Cache for standard attention, tuple(size(bsz, k_len/v_len, dmodel) * 2)
        - conv_state: Cache for causal_conv1d, tuple(size(bsz, q_dim/k_dim/v_dim, kernel_size) * 3) for deltanet or size(bsz, q_dim/k_dim/v_dim, kernel_size) for mamba2
    """

    is_compileable = True

    def __init__(self, cache_position: int = 0):
        super().__init__()

        self.states: List[Dict[str, Any]] = []
        self._cache_position = cache_position # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states: yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        recurrent_state: torch.Tensor = None,
        attn_state: Tuple[torch.Tensor, torch.Tensor] = None,
        conv_state: Tuple[torch.Tensor] | torch.Tensor = None,
        ffn_state: torch.Tensor = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new `recurrent_state`/`attn_state`/`conv_state` for the layer `layer_idx`.

        Args:
            recurrent_state (`torch.Tensor`, `optional`):
                The new recurrent state to cache.
            attn_state (`Tuple[torch.Tensor, torch.Tensor]`, `optional`):
                The new attention key/value states to cache.
            conv_state (`Tuple[torch.Tensor]`, `optional`):
                The new convolution state to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, `optional`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """

        # Update the number of seen tokens
        if layer_idx == 0:
            self._cache_position += offset

        if attn_state is not None:
            input_size = attn_state[0].shape[-2]
            window_size = cache_kwargs.get('window_size', None)
            if not isinstance(attn_state, Tuple) or len(attn_state) != 2:
                raise ValueError("`attn_state` must be a tuple of two tensors for key/value states")
        if len(self.states) <= layer_idx:
            if attn_state is not None:
                if window_size is not None and input_size > window_size:
                    attn_state = (attn_state[0][..., -window_size:, :].contiguous(),
                                  attn_state[1][..., -window_size:, :].contiguous())
            state = dict(
                recurrent_state=recurrent_state,
                attn_state=attn_state,
                conv_state=conv_state,
                ffn_state=ffn_state
            )
            self.states.append(state)
        else:
            state = self.states[layer_idx]
            if recurrent_state is not None:
                state['recurrent_state'] = recurrent_state
            if attn_state is not None:
                if state['attn_state'] is None:
                    if window_size is not None and input_size > window_size:
                        attn_state = (attn_state[0][..., -window_size:, :].contiguous(),
                                      attn_state[1][..., -window_size:, :].contiguous())
                else:
                    key_state, value_state = state['attn_state']
                    if window_size is not None and key_state.shape[-2] == window_size:
                        # DO NOT allocate new memory if the cache is full
                        # roll the key/value states to the left by `input_size`
                        key_state = key_state.roll(-input_size, -2)
                        value_state = value_state.roll(-input_size, -2)
                        # replace the last `input_size` tokens with the new key/value states
                        key_state[..., -input_size:, :] = attn_state[0]
                        value_state[..., -input_size:, :] = attn_state[1]
                        attn_state = (key_state, value_state)
                    else:
                        attn_state = (torch.cat([key_state, attn_state[0]], -2),
                                      torch.cat([value_state, attn_state[1]], -2),)
                state['attn_state'] = attn_state
            if conv_state is not None:
                state['conv_state'] = conv_state
            if ffn_state is not None:
                state['ffn_state'] = ffn_state

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self._cache_position

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple:
        return tuple(self.states)
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.states)):
            for k in self.states[layer_idx].keys():
                if isinstance(self.states[layer_idx][k], torch.Tensor):
                    device = self.states[layer_idx][k].device
                    self.states[layer_idx][k] = self.states[layer_idx][k].index_select(0, beam_idx.to(device))
                elif isinstance(self.states[layer_idx][k], Tuple):
                    _temp = []
                    for i in range(len(self.states[layer_idx][k])):
                        device = self.states[layer_idx][k][i].device
                        _temp.append(self.states[layer_idx][k][i].index_select(0, beam_idx.to(device)))
                    self.states[layer_idx][k] = tuple(_temp)

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple] = None,
        cache_position: int = 0
    ):
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(cache_position)
        if isinstance(past_key_values, list):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache