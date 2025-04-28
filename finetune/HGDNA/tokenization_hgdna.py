import os
import re
import json

from typing import Dict, List, Tuple, Union, Any
from transformers import PreTrainedTokenizer

class HGDNATokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]

    def __init__(
        self,
        model_max_length: int=int(1e9),
        bos_token="<s>",
        eos_token="</s>",
        sep_token="|",
        cls_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        unk_token="<unk>",
        **kwargs
    ):
        self.model_max_length = model_max_length

        # load vocab
        __cur_dir__ = os.path.dirname(os.path.abspath(__file__))

        rawvocab = json.load(open(os.path.join(__cur_dir__, "vocab.json"), "r"))
        self.vocab: Dict = rawvocab["tokens"]
        self.ids_to_tokens: Dict = {v:k for k, v in self.vocab.items()}

        self._class_vocab: Dict = rawvocab["class_tokens"]
        self._extra_vocab: Dict = rawvocab["extra_tokens"]
        self._base_vocab: Dict = rawvocab["base_tokens"]

        self._class_token_ids = set([self.vocab[_] for _ in self._class_vocab.keys()])

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            model_max_length=model_max_length,
            padding_side=kwargs.pop("padding_side", "right"),
            **kwargs
        )

        self.add_special_tokens(
            special_tokens_dict={"additional_special_tokens": list(self._class_vocab.keys())}
        )

        self._build_special_token_pattern()
    
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab
    
    def _convert_token_to_id(self, token) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])
    
    def _convert_id_to_token(self, index) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)
    
    def _build_special_token_pattern(self):
        special_tokens = [re.escape(_) for _ in self.all_special_tokens]
        if special_tokens:
            self.special_token_pattern = re.compile(
                "(?:" + "|".join(special_tokens) + ")"
            )
        else:
            raise ValueError("No special tokens have been added")
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        if not text: return []

        split_texts = self.special_token_pattern.split(text)

        tokens = []
        for i, split_text in enumerate(split_texts):
            if self.special_token_pattern.match(split_text):
                tokens.append(split_text)
            else:
                # normal token, process overlapping tokenization
                if len(split_text) >= 6:
                    for pos in range(0, len(split_text) - 6 + 1, 1):
                        if len(split_text) - pos >= 6: tokens.append(split_text[pos:pos+6])
                else:
                    for pos in range(len(split_text)):
                        tokens.append(split_text[pos])
        
        return tokens
    
    def _decode(self, token_ids: List[int], skip_special_tokens=False, **kwargs) -> str:
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens)

        return "".join(tokens)

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool=False) -> Union[str, List[str]]:
        if isinstance(ids, int): return self._convert_id_to_token(ids)

        tokens = []
        chunk_len = 0
        for _id in ids:
            token = self._convert_id_to_token(_id)
            if token not in self.all_special_tokens:
                if chunk_len == 0 or token in self._base_vocab: tokens.append(token)
                else: tokens.append(token[-1])

                chunk_len += 1
            else:
                if not skip_special_tokens: tokens.append(token)
                chunk_len = 0

        return tokens
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.bos_token_id is not None:
            token_ids_0 = [self.bos_token_id] + token_ids_0
        if self.eos_token_id is not None:
            # check class_id
            if token_ids_0[-1] in self._class_token_ids:
                token_ids_0 = token_ids_0[:-1] + [self.eos_token_id] + [token_ids_0[-1]]
            else:
                token_ids_0 = token_ids_0 + [self.eos_token_id]
        
        if token_ids_1 is not None:
            if self.bos_token_id is not None:
                token_ids_1 = [self.bos_token_id] + token_ids_1
            if self.eos_token_id is not None:
                # check class_id
                if token_ids_1[-1] in self._class_token_ids:
                    token_ids_1 = token_ids_1[:-1] + [self.eos_token_id] + [token_ids_1[-1]]
                else:
                    token_ids_1 = token_ids_1 + [self.eos_token_id]

            return token_ids_0 + token_ids_1
        
        else:
            return token_ids_0

    def save_vocabulary(self, save_directory, filename_prefix = None):
        return ()