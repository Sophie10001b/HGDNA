import random
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from typing import Optional, Dict, List, Tuple, Union, Unpack
from einops import rearrange, repeat
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from fla.modules.layernorm import RMSNorm
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast, SequenceClassifierOutputWithPast

from .configuration_hgdna import HGDNAConfig
from .hgdna_layer import Cache, HybridLayer, linearInit, embeddingInit

class HGDNAPretrainedModel(PreTrainedModel):
    config_class = HGDNAConfig
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _no_split_modules = ["HybridLayer"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            embeddingInit(module)
        elif isinstance(module, nn.Linear):
            linearInit(module, zero_bias=True)

class HGDNAModel(HGDNAPretrainedModel):
    def __init__(self, config: HGDNAConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=self.config.pad_token_id)

        self.layers = nn.ModuleList([
            HybridLayer(
                dmodel=config.hidden_size,
                dff=config.intermediate_size,
                nHead=config.num_attention_heads,
                window=config.window_size,
                drop=config.dropout_prob,
                base=config.rope_base,
                layer_idx=i
            )
            for i in range(0, config.num_hidden_layers//2)
        ])
        self.finalNorm = RMSNorm(config.hidden_size)

        self.post_init()
    
    def get_input_embeddings(self):
        return self.embedding
    
    def set_input_embeddings(self, value):
        self.embedding = value
    
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        inputs_embeds: Optional[torch.Tensor]=None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
        causal: Optional[bool]=True,
        soft_prompt: Optional[torch.FloatTensor]=None,
        **kwargs: Unpack[Dict]
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = kwargs.get("use_cache", None) if kwargs.get("use_cache", None) is not None else (self.config.use_cache if hasattr(self.config, "use_cache") and (not self.training) else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if kwargs.get("use_gradient_checkpoint", False) is True and self.supports_gradient_checkpointing and self.training: self.gradient_checkpointing_enable()
        else: self.gradient_checkpointing = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None: inputs_embeds = self.embedding(input_ids)
        if use_cache and not isinstance(past_key_values, Cache): past_key_values = Cache.from_legacy_cache(past_key_values)

        if soft_prompt is not None: inputs_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states: all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, _, past_key_values = checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    causal,
                    past_key_values
                )
            else:
                hidden_states, _, past_key_values = layer(hidden_states, causal=causal, past_key_values=past_key_values)
        
        hidden_states = self.finalNorm(hidden_states)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, past_key_values] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None
        )


class HGDNAForSequenceClassification(HGDNAPretrainedModel):
    _tied_weights_keys = []

    def __init__(self, config: HGDNAConfig, **kwargs):
        super().__init__(config, **kwargs)

        assert self.config.problem_type in ["single_label_classification", "multi_label_classification", "regression", "embedding"]

        self.decoder = HGDNAModel(config)
        
        self.lmHead = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.critertion = nn.MSELoss() if config.problem_type == "regression" else CrossEntropyLoss(ignore_index=self.config.pad_token_id)

        # maybe used
        if self.config.problem_type == "regression":
            self.score = nn.Linear(self.config.hidden_size, self.config.num_labels, bias=False)
        else:
            self.score = None

        if self.config.num_prompts > 0:
            self.softPrompt = nn.Parameter(torch.zeros(self.config.num_prompts, self.config.prompts_size, self.config.hidden_size, dtype=torch.float32))
        else:
            self.softPrompt = None

        self.post_init()
    
    def get_input_embeddings(self):
        return self.decoder.embedding
    
    def set_input_embeddings(self, value):
        self.decoder.embedding = value
    
    def _init_weights(self, module):
        for k, v in self.named_modules():
            if isinstance(v, nn.Linear):
                linearInit(v)
            elif isinstance(v, nn.Embedding):
                embeddingInit(v)

        if self.softPrompt is not None:
            indices = []
            candidate = list(range(self.config.class_ids_start, self.config.class_ids_start + min(self.config.class_ids_end, self.config.prompts_size)))
            for _ in range(self.config.num_prompts):
                random.shuffle(candidate)
                indices.extend(candidate)
            
            _weight = self.decoder.embedding.weight[indices].clone().detach()
            _weight = rearrange(_weight, "(N L) D -> N L D", N=self.config.num_prompts, L=self.config.prompts_size)
            self.softPrompt.data = _weight
    
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        inputs_embeds: Optional[torch.Tensor]=None,
        labels: Optional[torch.LongTensor|torch.FloatTensor]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
        **kwargs: Unpack[Dict]
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        cls_idx = None
        if inputs_embeds is None:
            batch_size = input_ids.size(0)
            device = input_ids.device
        elif input_ids is None:
            batch_size = inputs_embeds.size(0)
            device = inputs_embeds.device

        prompt_idx = kwargs.get("prompt_idx", torch.zeros((batch_size,), dtype=torch.int64, device=device))

        if self.softPrompt is not None:
            soft_prompt = self.softPrompt.index_select(0, prompt_idx)
        else: soft_prompt = None
        
        label_num = 1
        if labels is not None and labels.dim() > 1:
            label_num = labels.shape[1]

        # multi_label for "seq</s></s></s>...", where </s> denotes each label
        if input_ids is not None:
            cls_idx = torch.nonzero(input_ids.flatten() == self.config.eos_token_id).flatten().reshape(batch_size, label_num)
            if self.softPrompt is not None:
                cls_idx += (torch.arange(1, batch_size+1, dtype=torch.int64, device=device) * soft_prompt.size(1))[:, None]
            
            cls_idx = cls_idx.flatten()
        
        loss = None
        outputs: BaseModelOutputWithPast = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            causal=self.config.causal,
            soft_prompt=soft_prompt,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state.flatten(0, 1).index_select(0, cls_idx)

        if self.config.problem_type == "regression":
            logits: torch.Tensor = self.score(hidden_states)
            if labels is not None:
                loss = self.critertion(logits, labels.flatten().reshape(-1, self.config.num_labels))
        
        elif self.config.problem_type in ["single_label_classification", "multi_label_classification"]:
            logits: torch.Tensor = self.lmHead(hidden_states)
            if labels is not None:
                loss = self.critertion(logits, labels.flatten())
        
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states if self.config.output_cls_states else outputs.hidden_states,
            attentions=outputs.attentions
        )


class HGDNAForCausalLM(HGDNAPretrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config: HGDNAConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.decoder = HGDNAModel(config)
        
        self.lmHead = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.critertion = CrossEntropyLoss(ignore_index=self.config.pad_token_id)

        if self.config.num_prompts > 0:
            self.softPrompt = nn.Parameter(torch.zeros(self.config.num_prompts, self.config.prompts_size, self.config.hidden_size, dtype=torch.float32))
        else:
            self.softPrompt = None

        self.post_init()
    
    def get_input_embeddings(self):
        return self.decoder.embedding
    
    def set_input_embeddings(self, value):
        self.decoder.embedding = value
    
    def _init_weights(self, module):
        for k, v in self.named_modules():
            if isinstance(v, nn.Linear):
                linearInit(v)
            elif isinstance(v, nn.Embedding):
                embeddingInit(v)

        if self.softPrompt is not None:
            indices = []
            candidate = list(range(self.config.class_ids_start, self.config.class_ids_start + min(self.config.class_ids_end, self.config.prompts_size)))
            for _ in range(self.config.num_prompts):
                random.shuffle(candidate)
                indices.extend(candidate)
            
            _weight = self.decoder.embedding.weight[indices].clone().detach()
            _weight = rearrange(_weight, "(N L) D -> N L D", N=self.config.num_prompts, L=self.config.prompts_size)
            self.softPrompt.data = _weight
    
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
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
        logits_to_keep: Optional[int]=0,
        **kwargs: Unpack[Dict]
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (self.softPrompt is not None) and (past_key_values is None or past_key_values.get_seq_length() == 0):
            prompt_idx = kwargs.pop("prompt_idx", None)
            soft_prompt = self.softPrompt.index_select(0, prompt_idx) if prompt_idx is not None else None
        else: soft_prompt = None
        
        loss = None
        outputs: BaseModelOutputWithPast = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values if labels is None else None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            causal=True,
            soft_prompt=soft_prompt,
            **kwargs
        )

        if labels is not None:
            # assert all the prompt length is padded to the same
            tgt_idx = torch.nonzero(input_ids[0] == self.config.bos_token_id).flatten()[-1]
            
            hidden_states = outputs.last_hidden_state[:, tgt_idx+self.config.prompts_size:-1]
            logits: torch.Tensor = self.lmHead(hidden_states)
            loss = self.critertion(rearrange(logits, "B L D -> (B L) D"), labels[:, 1:].flatten())
        
        else: #generation mode
            hidden_states = outputs.last_hidden_state
            logits: torch.Tensor = self.lmHead(hidden_states)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions
        )