# this code
## Only load the model architecture without loading the weight, we do the pretrain by ourself

import argparse
import torch 
import os 
import pathlib

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,

)


model_name = "Qwen/Qwen2.5-0.5B-Instruct" # Name on the model card
config = AutoConfig.from_pretrained(model_name)
print(config)
# we want to expand this 0.5B model to 1.5B so 
config.hidden_size = 1536 # 1.7 x
config.num_hidden_layer = 24 # remain same 
config.num_attention_heads = 24 #
config.num_key_value_heads = 4 # keep grouped ratio similar 
config.intermediate_size = 8352 # 5.43 X hidden size 

work_dir =pathlib.Path.cwd()
new_model_path = work_dir.joinpath("model")
print(work_dir)
print(new_model_path)

model = AutoModelForCausalLM.from_config(config,torch_dtype = torch.bfloat16,attn_implementation='flash_attention_2')
tokenizer = AutoTokenizer.from_pretrained(model_name)

# calculation of nb of params 
num_params = sum(p.numel() for p in model.parameters())
print(num_params)
# print(model)
# print(config)

