import math
import torch
import torch.nn as nn
import transformers

from typing import Optional, Dict, Tuple, List, Any

from fla import GatedDeltaNet
from fla.modules.layernorm import RMSNorm
from flash_attn import flash_attn_qkvpacked_func, flash_attn_kvpacked_func
from flash_attn.layers import rotary
from einops import rearrange
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

#########################################################
#            --- basic functions ---
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
        self._cache_position = [cache_position] # Used in `generate` to keep tally of how many tokens the cache has seen

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
        if len(self._cache_position) <= layer_idx:
            self._cache_position.append(0)

        self._cache_position[layer_idx] += offset

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
        return self._cache_position[layer_idx]

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


@torch.no_grad()
def embeddingInit(embedding: nn.Embedding) ->None:
    fan_out = embedding.weight.size(1)
    std = 1.0 * math.sqrt(1.0 / float(fan_out))
    nn.init.normal_(embedding.weight, 0., std)
    if embedding.padding_idx is not None:
        embedding.weight[embedding.padding_idx].fill_(0)


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
        layer_idx: int
    ):
        super(LinearAttention, self).__init__()
        self.layer_idx = layer_idx

        self.tokenMixer = GatedDeltaNet(
            hidden_size=dmodel,
            expand_v=2,
            head_dim=int(0.75 * dmodel / 6),
            layer_idx=layer_idx
        )
    
    def initParam(self):
        for m in self.tokenMixer.modules():
            if isinstance(m, nn.Linear):
                linearInit(m)
    
    def forward(self, x: torch.Tensor, past_key_values: Cache=None, offset_update: bool=True):
        """
        x -> size(B, L, D)
        """
        out, _, past_key_values = self.tokenMixer(x, past_key_values=past_key_values, use_cache=False if past_key_values is None else True)
        
        if (past_key_values is not None) and (not offset_update) and (len(past_key_values._cache_position) > self.layer_idx): past_key_values._cache_position[self.layer_idx] -= x.size(1)
        return out, None, past_key_values


class HybridLayer(nn.Module):
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
        super(HybridLayer, self).__init__()

        self.tokenMixer = LinearAttention(dmodel, layer_idx)
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