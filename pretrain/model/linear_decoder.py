import time
import argparse
from attr import dataclass
from sympy import O
import torch
import torch.nn as nn

from typing import Optional, Dict, Tuple
from mamba_ssm import Mamba2
from fla import GatedDeltaNet
from fla.modules.layernorm import RMSNorm
from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func
from flash_attn.layers import rotary
from einops import rearrange
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from dataclasses import dataclass

from .utils import GLU, linearInit
from .utils import Cache

#########################################################
#                   --- config ---
#########################################################
class NTPConfig(PretrainedConfig):
    model_type = "NTP"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, args: argparse.Namespace, vocab: Dict[str, int], **kwargs):
        super().__init__(
            bos_token_id = vocab["<s>"],
            eos_token_id = vocab["</s>"],
            pad_token_id = vocab["<pad>"],
            **kwargs
        )

        self.dmodel = args.dmodel
        self.dff = self.dmodel * args.ffnScale
        self.nHead = args.nHead
        self.nLayer = args.nLayer
        self.nPrompt = args.nPrompt
        self.dPrompt = args.dPrompt
        self.drop = args.drop
        self.arch = args.arch
        self.trainMode = args.trainMode
        self.finetuneTask = args.finetuneTask

        self.vocab = vocab
        self.vocabSize = len(vocab)

@dataclass
class NTPOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    # custom
    subloss: Optional[torch.FloatTensor] = None
    pred: Optional[torch.LongTensor] = None
    label: Optional[torch.LongTensor] = None

#########################################################
#                   --- model ---
#########################################################
class SWA(nn.Module):
    def __init__(self, dmodel: int, nHead: int, window: int, drop: float, base: int, layer_idx: int):
        super(SWA, self).__init__()

        self._qkv = nn.Linear(dmodel, 3 * dmodel, bias=False)
        self._out = nn.Linear(dmodel, dmodel, bias=False)
        self.dmodel = dmodel
        self.nHead = nHead
        self.dHead = dmodel // nHead
        self.drop = drop
        self.window = window
        self.layer_idx = layer_idx

        self.rotary = rotary.RotaryEmbedding(dim=self.dHead, base=base)

        self.initParam()
    
    def initParam(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): linearInit(m, zero_bias=True)
    
    def forward(self, x: torch.Tensor, causal: bool=True, past_key_values: Cache=None):
        """
        x -> size(B, L, D)
        """
        if self.window > 0:
            window = (self.window - 1, 0) if causal else (self.window // 2, self.window // 2)
        else: window = (-1, -1)

        qkv: torch.Tensor = self._qkv(x)
        qkv = rearrange(qkv, "B L (C H D) -> B L C H D", C=3, H=self.nHead, D=self.dHead)

        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = qkv.size(1) + seqlen_offset
            qkv = self.rotary(qkv, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)

            q, k, v = qkv.unbind(dim=2)
            k, v = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q.size(1),
                cache_kwargs=dict(window_size=self.window) if self.window > 0 else dict()
            )["attn_state"]
            k, v = rearrange(k, "... (H D) -> ... H D", H=self.nHead, D=self.dHead), rearrange(v, "... (H D) -> ... H D", H=self.nHead, D=self.dHead)
        
            kv = torch.cat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
            out = flash_attn_kvpacked_func(q, kv, dropout_p=self.drop if self.training else 0, window_size=window, causal=causal)
        
        else:
            qkv = self.rotary(qkv)
            out = flash_attn_qkvpacked_func(qkv, dropout_p=self.drop if self.training else 0, window_size=window, causal=causal)

        return self._out(rearrange(out, "B L H D -> B L (H D)")), None, past_key_values


class LinearAttention(nn.Module):
    def __init__(
        self,
        dmodel: int,
        arch: str,
        layer_idx: int
    ):
        super(LinearAttention, self).__init__()
        self.arch = arch
        self.layer_idx = layer_idx

        if arch == "gated_delta_net":
            self.tokenMixer = GatedDeltaNet(
                hidden_size=dmodel,
                expand_v=2,
                head_dim=int(0.75 * dmodel / 6),
                layer_idx=layer_idx
            )
        elif arch == "mamba2":
            self.tokenMixer = Mamba2(d_model=dmodel, expand=2, layer_idx=layer_idx)
    
    def initParam(self):
        for m in self.tokenMixer.modules():
            if isinstance(m, nn.Linear):
                linearInit(m)
    
    def forward(self, x: torch.Tensor, past_key_values: Cache=None, offset_update: bool=True):
        """
        x -> size(B, L, D)
        """
        if isinstance(self.tokenMixer, GatedDeltaNet):
            out, _, past_key_values = self.tokenMixer(x, past_key_values=past_key_values, use_cache=False if past_key_values is None else True)
            
        elif isinstance(self.tokenMixer, Mamba2):
            if past_key_values is None: out = self.tokenMixer(x)
            else:
                if len(past_key_values) <= self.layer_idx:
                    conv_state, ssm_state = self.tokenMixer.allocate_inference_cache(x.size(0))
                    past_key_values.update(recurrent_state=ssm_state, conv_state=conv_state, layer_idx=self.layer_idx, offset=0)
                
                _temp = []
                for i in range(x.size(1)):
                    out, conv_state, ssm_state = self.tokenMixer(x[:, i:i+1], past_key_values[self.layer_idx]["conv_state"], past_key_values[self.layer_idx]["recurrent_state"])
                    past_key_values.update(recurrent_state=ssm_state, conv_state=conv_state, layer_idx=self.layer_idx)
                    _temp.append(out)
                out = torch.cat(_temp, dim=1)
        
        if (past_key_values is not None) and (not offset_update) and (self.layer_idx == 0): past_key_values._cache_position -= x.size(1)
        return out, None, past_key_values

#########################################################
#                   --- layer ---
#########################################################
class HybridLayer(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        nHead: int,
        window: int,
        drop: float,
        base: int,
        arch: str,
        layer_idx: int
    ):
        super(HybridLayer, self).__init__()

        self.tokenMixer = LinearAttention(dmodel, arch, layer_idx)
        self.tokenMixerNorm = RMSNorm(dmodel)
        self.ffn1 = GLU(
            dmodel=dmodel,
            dff=dff,
            drop=drop
        )
        self.ffn1Norm = RMSNorm(dmodel)
        self.attention = SWA(
            dmodel=dmodel,
            nHead=nHead,
            window=window,
            drop=drop,
            base=base,
            layer_idx=layer_idx
        )
        self.attentionNorm = RMSNorm(dmodel)
        self.ffn2 = GLU(
            dmodel=dmodel,
            dff=dff,
            drop=drop
        )
        self.ffn2Norm = RMSNorm(dmodel)
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor, causal: bool=True, past_key_values: Cache=None):
        """
        x -> size(B, L, D)
        """
        out, _, past_key_values = self.tokenMixer(self.tokenMixerNorm(x), past_key_values=past_key_values, offset_update=False)
        x = x + out
        x = x + self.ffn1(self.ffn1Norm(x))
        out, _, past_key_values = self.attention(self.attentionNorm(x), causal=causal, past_key_values=past_key_values)
        x = x + out
        x = x + self.ffn2(self.ffn2Norm(x))
        return x, None, past_key_values

class FullAttentionLayer(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        nHead: int,
        window: int,
        drop: float,
        base: int,
        layer_idx: int
    ):
        super(FullAttentionLayer, self).__init__()

        self.tokenMixer = SWA(dmodel, nHead, window, drop, base, layer_idx)
        self.ffn = GLU(dmodel, dff, drop)
        self.tokenMixerNorm = RMSNorm(dmodel)
        self.ffnNorm = RMSNorm(dmodel)
        self.layer_idx = layer_idx
    
    def forward(self, x: torch.Tensor, causal: bool=True, past_key_values: Cache=None):
        """
        x -> size(B, L, D)
        """
        out, _, past_key_values = self.tokenMixer(self.tokenMixerNorm(x), causal=causal, past_key_values=past_key_values)
        x = x + out
        x = x + self.ffn(self.ffnNorm(x))
        return x, None, past_key_values

class LinearAttentionLayer(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        drop: float,
        arch: str,
        layer_idx: int
    ):
        super(LinearAttentionLayer, self).__init__()

        self.tokenMixer = LinearAttention(dmodel, arch, layer_idx)
        self.ffn = GLU(dmodel, dff, drop)
        self.tokenMixerNorm = RMSNorm(dmodel)
        self.ffnNorm = RMSNorm(dmodel)
        self.layer_idx = layer_idx
    
    def forward(self, x: torch.Tensor, causal: bool=True, past_key_values: Cache=None):
        """
        x -> size(B, L, D)
        """
        out, _, past_key_values = self.tokenMixer(self.tokenMixerNorm(x), past_key_values=past_key_values)
        x = x + out
        x = x + self.ffn(self.ffnNorm(x))
        return x, None, past_key_values

#########################################################
#                   --- decoder ---
#########################################################
class HybridDecoder(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        nHead: int,
        nLayer: int,
        window: int,
        drop: float,
        base: int,
        arch: str
    ):
        super(HybridDecoder, self).__init__()

        self.layers = nn.ModuleList([
            HybridLayer(
                dmodel=dmodel,
                dff=dff,
                nHead=nHead,
                window=window,
                drop=drop,
                base=base,
                arch=arch,
                layer_idx=i
            )
            for i in range(0, nLayer//2)
        ])
        self.finalNorm = RMSNorm(dmodel)
    
    def forward(self, x: torch.Tensor, causal: bool=True, past_key_values: Cache=None, return_hidden_states: bool=False):
        """
        x -> size(B, L, D)
        """
        all_hidden_states = () if return_hidden_states else None
        for layer in self.layers:
            x, _, past_key_values = layer(x, causal=causal, past_key_values=past_key_values)
            if return_hidden_states: all_hidden_states += (x,)
        return BaseModelOutputWithPast(
            last_hidden_state=self.finalNorm(x),
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None
        )


class FullAttentionDecoder(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        nHead: int,
        nLayer: int,
        window: int,
        drop: float,
        base: int
    ):
        super(FullAttentionDecoder, self).__init__()
        self.layers = nn.ModuleList([
            FullAttentionLayer(
                dmodel=dmodel,
                dff=dff,
                nHead=nHead,
                window=window,
                drop=drop,
                base=base,
                layer_idx=i
            )
            for i in range(nLayer)
        ])
        self.finalNorm = RMSNorm(dmodel)
    
    def forward(self, x: torch.Tensor, causal: bool=True, past_key_values: Cache=None, return_hidden_states: bool=False):
        """
        x -> size(B, L, D)
        """
        all_hidden_states = () if return_hidden_states else None
        for layer in self.layers:
            x, _, past_key_values = layer(x, causal=causal, past_key_values=past_key_values)
            if return_hidden_states: all_hidden_states += (x,)
        return BaseModelOutputWithPast(
            last_hidden_state=self.finalNorm(x),
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None
        )


class LinearAttentionDecoder(nn.Module):
    def __init__(
        self,
        dmodel: int,
        dff: int,
        nLayer: int,
        drop: float,
        arch: str
    ):
        super(LinearAttentionDecoder, self).__init__()
        self.layers = nn.ModuleList([
            LinearAttentionLayer(
                dmodel=dmodel,
                dff=dff,
                drop=drop,
                arch=arch,
                layer_idx=i
            )
            for i in range(nLayer)
        ])
        self.finalNorm = RMSNorm(dmodel)
    
    def forward(self, x: torch.Tensor, causal: bool=True, past_key_values: Cache=None, return_hidden_states: bool=False):
        """
        x -> size(B, L, D)
        """
        all_hidden_states = () if return_hidden_states else None
        for layer in self.layers:
            x, _, past_key_values = layer(x, causal=causal, past_key_values=past_key_values)
            if return_hidden_states: all_hidden_states += (x,)
        return BaseModelOutputWithPast(
            last_hidden_state=self.finalNorm(x),
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None
        )