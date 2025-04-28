import random
import argparse
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from typing import Optional, Dict, List, Tuple, Union, Unpack
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from .preprocess import PackedData, CLASSLIST
from .linear_decoder import FullAttentionDecoder, LinearAttentionDecoder, HybridDecoder, NTPConfig, NTPOutputWithPast
from .utils import Cache, linearInit, embeddingInit

#########################################################
#          --- NTP Model (HF Generation) ---            #
#########################################################
class NTP(PreTrainedModel, GenerationMixin):
    config_class = NTPConfig
    supports_gradient_checkpointing = True
    # _tied_weights_keys = ["lmHead.weight"]
    _supports_cache_class = True

    def __init__(self, config: NTPConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(self.config.vocabSize, self.config.dmodel, padding_idx=self.config.pad_token_id)

        if self.config.arch.split('_')[0] == "hybrid":
            arch = self.config.arch.split('_')[1:]
            arch = "_".join(arch)
            self.decoder = HybridDecoder(self.config.dmodel, self.config.dff, self.config.nHead, self.config.nLayer, 1024, self.config.drop, int(1e6), arch)
        else:
            if self.config.arch == "full_attention":
                self.decoder = FullAttentionDecoder(self.config.dmodel, self.config.dff, self.config.nHead, self.config.nLayer, -1, self.config.drop, int(1e6))
            elif self.config.arch == "window_attention":
                self.decoder = FullAttentionDecoder(self.config.dmodel, self.config.dff, self.config.nHead, self.config.nLayer, -1, self.config.drop, int(1e6))
            else:
                self.decoder = LinearAttentionDecoder(self.config.dmodel, self.config.dff, self.config.nLayer, self.config.drop, self.config.arch)
        
        self.lmHead = nn.Linear(self.config.dmodel, self.config.vocabSize, bias=False)
        self.critertion = CrossEntropyLoss(ignore_index=self.config.pad_token_id)

        if self.config.nPrompt > 0 and self.config.trainMode == "finetune":
            self.softPrompt = nn.Parameter(torch.zeros(self.config.nPrompt, self.config.dPrompt, self.config.dmodel, dtype=torch.float32))
        else:
            self.softPrompt = None

        self.post_init()
    
    def _init_weights(self, module):
        for k, v in self.named_modules():
            if isinstance(v, nn.Linear):
                linearInit(v)
            elif isinstance(v, nn.Embedding):
                embeddingInit(v)

        if self.softPrompt is not None:
            indices = []
            candidate = [self.config.vocab[_] for _ in CLASSLIST][:self.config.dPrompt]
            for _ in range(self.config.nPrompt):
                random.shuffle(candidate)
                indices.extend(candidate)
            
            _weight = self.embedding.weight[indices].clone().detach()
            _weight = rearrange(_weight, "(N L) D -> N L D", N=self.config.nPrompt, L=self.config.dPrompt)
            self.softPrompt.data = _weight
    
    def get_input_embeddings(self):
        return self.embedding
    
    def set_input_embeddings(self, value):
        self.embedding = value
    
    def get_output_embeddings(self):
        return self.lmHead
    
    def set_output_embeddings(self, value):
        self.lmHead = value
    
    def set_decoder(self, value):
        self.decoder = value
    
    def get_decoder(self):
        return self.decoder
    
    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor=None, past_key_values: Cache=None, attention_mask=None, inputs_embeds=None, use_cache: bool=True, cache_position=None, logits_to_keep: int=None, prompt_idx: torch.Tensor=None, **kwargs
    ):
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids.contiguous()}
        
        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'prompt_idx': prompt_idx
        })
        return model_inputs
    
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        inputs_embeds: Optional[torch.Tensor]=None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]]=None,
        labels: Optional[torch.LongTensor]=None,
        use_cache: Optional[bool]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
        logits_to_keep: Optional[int]=0,
        **kwargs: Unpack[Dict]
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training: use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        cls_idx = None
        if inputs_embeds is None: inputs_embeds = self.embedding(input_ids)
        if use_cache and not isinstance(past_key_values, Cache): past_key_values = Cache.from_legacy_cache(past_key_values)

        if kwargs.get("use_gradient_checkpoint", False) is True and self.supports_gradient_checkpointing and self.training: self.gradient_checkpointing_enable()
        else: self.gradient_checkpointing = False

        if (self.softPrompt is not None) and (past_key_values is None or past_key_values.get_seq_length() == 0):
            soft_prompt = self.softPrompt.index_select(0, kwargs['prompt_idx'])
            inputs_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)
        
        if input_ids is not None and (self.config.trainMode == "pretrain" or self.config.finetuneTask in ["classification"]):
            cls_idx = torch.nonzero(input_ids.flatten() == self.config.eos_token_id).flatten()
            if self.softPrompt is not None:
                cls_idx += (torch.arange(1, inputs_embeds.size(0)+1, dtype=torch.int64, device=cls_idx.device) * soft_prompt.size(1)).repeat_interleave(cls_idx.size(0) // inputs_embeds.size(0), 0)
        
        fwd_func = lambda _x, _causal, _cache, _return_hidden: self.decoder(_x, _causal, _cache, _return_hidden) if not self.gradient_checkpointing else \
            checkpoint.checkpoint(self.decoder.__call__, _x, _causal, _cache, _return_hidden, use_reentrant=False)
        
        loss, subloss, pred, label = None, None, None, labels
        if self.config.trainMode == "pretrain":
            past_key_values = None
            outputs: BaseModelOutputWithPast = fwd_func(inputs_embeds, True, past_key_values, output_hidden_states)
            hidden_states = outputs.last_hidden_state[:, :-1]
            logits: torch.Tensor = self.lmHead(hidden_states)
            loss = self.critertion(rearrange(logits, "B L D -> (B L) D"), labels.flatten())

            if kwargs.get('species_idx', None) is not None:
                cls_idx -= torch.arange(0, cls_idx.size(0), dtype=torch.int64, device=cls_idx.device)
                cls_res = logits.flatten(0, 1).index_select(0, cls_idx)
                subloss = self.critertion(cls_res, kwargs['species_idx'].flatten())
        
        elif self.config.finetuneTask in ["classification"]:
            past_key_values = None
            outputs: BaseModelOutputWithPast = fwd_func(inputs_embeds, False, past_key_values, output_hidden_states)
            hidden_states = outputs.last_hidden_state.flatten(0, 1).index_select(0, cls_idx)
            logits: torch.Tensor = self.lmHead(hidden_states)
            if (not self.training):
                if kwargs.get('augment', False) is True:
                    _logits, _logits_reverse = logits.chunk(2, 0)
                    _logits, _logits_reverse = _logits.softmax(-1), _logits_reverse.softmax(-1)
                    _logits = (_logits + _logits_reverse) / 2
                    pred = _logits.argmax(-1)
                else: pred = logits.argmax(-1)
            else:
                loss = self.critertion(logits, labels.flatten())

        elif self.config.finetuneTask == "generation":
            if self.training: # training or evaling
                past_key_values = None
                outputs: BaseModelOutputWithPast = fwd_func(inputs_embeds, True, past_key_values, output_hidden_states)
                # assert all the prompt length is padded to the same
                tgt_idx = torch.nonzero(input_ids[0] == self.config.bos_token_id).flatten()[-1]
                
                hidden_states = outputs.last_hidden_state[:, tgt_idx+self.config.dPrompt:-1]
                logits: torch.Tensor = self.lmHead(hidden_states)
                loss = self.critertion(rearrange(logits, "B L D -> (B L) D"), labels.flatten())
            
            else: # testing in generation mode
                outputs: BaseModelOutputWithPast = fwd_func(inputs_embeds, True, past_key_values if use_cache else None, False)
                hidden_states = outputs.last_hidden_state
                logits: torch.Tensor = self.lmHead(hidden_states)
        
        return NTPOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            subloss=subloss,
            pred=pred,
            label=label
        )