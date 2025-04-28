from typing import Dict, List, Tuple, Union, Any
from transformers.configuration_utils import PretrainedConfig

class HGDNAConfig(PretrainedConfig):
    model_type = "NTP"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=None,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout_prob=0.1,
        window_size=1024,
        rope_base=int(1e6),
        num_labels=1,
        num_prompts=64,
        prompts_size=64,
        class_ids_start=10,
        class_ids_end=1033,
        problem_type='single_label_classification',
        output_cls_states=False,
        causal=True,
        **kwargs
    ):
        super().__init__(
            bos_token_id = bos_token_id,
            eos_token_id = eos_token_id,
            pad_token_id = pad_token_id,
            **kwargs
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_prob = dropout_prob
        self.window_size = window_size
        self.rope_base = rope_base
        self.num_labels = num_labels
        self.num_prompts = num_prompts
        self.prompts_size = prompts_size
        self.class_ids_start = class_ids_start
        self.class_ids_end = class_ids_end
        self.problem_type = problem_type
        self.output_cls_states = output_cls_states
        self.causal = causal