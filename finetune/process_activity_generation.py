import os
import glob
import torch
import numpy as np
from copy import deepcopy

import transformers
from transformers.generation import GenerationConfig
from lightning import seed_everything

from utils import process_json

if __name__ == "__main__":

    SEED = 17
    seed_everything(SEED)

    task = "developmental"
    modelName = "hyenadna-medium-160k-seqlen-hf"
    # modelName = "HGDNA"
    curDir = os.path.dirname(os.path.abspath(__file__))

    hfModelPath = os.path.join(curDir, f"{modelName}")
    ckptPath = os.path.join(curDir, "result", f"{modelName}", f"{task}_generation.pt")

    tokenizer = transformers.AutoTokenizer.from_pretrained(hfModelPath, model_max_length=int(1e6), padding_side="right", use_fast=True, trust_remote_code=True)
    modelConfig = transformers.AutoConfig.from_pretrained(hfModelPath, trust_remote_code=True)
    modelConfig.num_prompts = 10
    modelConfig.prompts_size = 64
    modelConfig.dropout_prob = 0.0

    if "hyena" in modelName:
        modelConfig.eos_token_id = 1
        modelConfig.pad_token_id = 4

    model = transformers.AutoModelForCausalLM.from_pretrained(hfModelPath, trust_remote_code=True, config=modelConfig)

    _state_dict = torch.load(ckptPath, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(_state_dict, strict=True)
    except Exception as e:
        print(e)
        print("Model loading mismatch, try more flexible loading\n")
        model.load_state_dict(_state_dict, strict=False)

    # load regression model
    hfJudgeModelPath = os.path.join(curDir, "HGDNA")
    judgeCkptPath = os.path.join(curDir, "result", "HGDNA", f"{task}_regression.pt")
    judgeTokenizer = transformers.AutoTokenizer.from_pretrained(hfJudgeModelPath, model_max_length=int(1e6), padding_side="right", use_fast=True, trust_remote_code=True)
    judgeConfig = transformers.AutoConfig.from_pretrained(hfJudgeModelPath, trust_remote_code=True)
    judgeConfig.problem_type = "regression"
    judgeConfig.num_labels = 1
    judgeConfig.num_prompts = 4
    judgeConfig.prompts_size = 64
    judgeConfig.dropout_prob = 0.0
    judgeConfig.causal = False

    judgeModel = transformers.AutoModelForSequenceClassification.from_pretrained(hfJudgeModelPath, trust_remote_code=True, config=judgeConfig)
    _state_dict = torch.load(judgeCkptPath, map_location="cpu", weights_only=True)
    try:
        judgeModel.load_state_dict(_state_dict, strict=True)
    except Exception as e:
        print(e)
        print("Model loading mismatch, try more flexible loading\n")
        judgeModel.load_state_dict(_state_dict, strict=False)

    model = model.to("cuda")
    judgeModel = judgeModel.to("cuda")

    genConfig = GenerationConfig(
        max_new_tokens=300,
        num_beams=40,
        num_beam_groups=20,
        temperature=1.0,
        use_cache=True,
        num_return_sequences=20,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        diversity_penalty=1.0
    )

    genCache = {}
    with torch.no_grad():
        for i in range(10):
            prompt_idx = torch.tensor([i], device="cuda:0", dtype=torch.int64)
            randomInput = torch.randint
            inputs = tokenizer([""], return_tensors="pt", padding="longest", max_length=int(1e6), truncation=True)["input_ids"].to("cuda")
            inputs = inputs[..., :-1]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model.generate(inputs, generation_config=genConfig, prompt_idx=prompt_idx)
            
            outputs[outputs >= tokenizer.vocab_size] = modelConfig.pad_token_id
            outputs = outputs.cpu()
            for batch_idx in range(outputs.size(0)):
                eos_pos = (outputs[batch_idx] == modelConfig.eos_token_id).nonzero()
                if len(eos_pos) > 0:
                    eos_pos = eos_pos[0].item()
                    if eos_pos + 1 < outputs.size(1): outputs[batch_idx, eos_pos+1:] = tokenizer.pad_token_id

            results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            judgeInputs = judgeTokenizer(results, return_tensors="pt", padding="longest", max_length=int(1e6), truncation=True)["input_ids"].to("cuda")
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                judgeRes = judgeModel(judgeInputs)["logits"]
            judgeRes = judgeRes.squeeze(-1).to(torch.float32).cpu().numpy()

            tmp = {_idx: (results[_idx], float(judgeRes[_idx])) for _idx in range(len(results))}

            genCache[i] = {
                "res": tmp,
                "avg_activity": float(judgeRes.mean()),
                "std_activity": float(judgeRes.std()),
                "min_activity": float(judgeRes.min()),
                "max_activity": float(judgeRes.max())
            }

    process_json(os.path.join(curDir, "result", f"{modelName}", f"{task}_generation_new.json"), genCache, "write")