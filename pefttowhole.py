from transformers import MambaForCausalLM,AutoTokenizer
import json
from transformers import BartConfig,MambaForCausalLM,Seq2SeqTrainer,DataCollatorForSeq2Seq,AutoTokenizer,TrainingArguments,Seq2SeqTrainingArguments,BartTokenizer
from torch.utils.data import Dataset,DataLoader
import datasets
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataset import TestDataset
from tqdm import tqdm
from peft import PeftModelForCausalLM
# params = list(model.parameters())
# k = 0
# for i in params:
#     l = 1
#     print("该层的结构：" + str(list(i.size())))
#     for j in i.size():
#         l *= j
#     print("该层参数和：" + str(l))
#     k = k + l
# print("总参数数量和：" + str(k))

import json
with open('/data/ruanjh/best_training_method/iwslt17/test.json') as f:
    test_data=json.load(f)
    



def load_model_and_tokenizer(model_dir,peft_model_id):
    base_model = MambaForCausalLM.from_pretrained(model_dir,torch_dtype='auto')
    
    model = PeftModelForCausalLM.from_pretrained(base_model, peft_model_id)
    model.eval()
    # model=MambaForCausalLM.from_pretrained(model_dir,torch_dtype='auto',)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,padding_side='left')
    return model,tokenizer

model_dir='/data/ruanjh/mamba-2.8b-hf'
peft_model_id = "/data/ruanjh/mamba-translate-2.8b-lora/checkpoint-6800"
model,tokenizer=load_model_and_tokenizer(model_dir,peft_model_id)
model=model.merge_and_unload()
print(type(model))
model.base_model.save_pretrained('/data/ruanjh/mamba-translate-2.8b-ckpt6800lora',safe_serialization=False,max_shard_size='20GB')


# model_0,_=load_model_and_tokenizer('/data/ruanjh/mamba-2.8b-hf','/data/ruanjh/mamba-translate-2.8b-lora/checkpoint-1700')
# model_1,_=load_model_and_tokenizer('/data/ruanjh/mamba-translate-2.8b-lora/checkpoint-100','/data/ruanjh/mamba-translate-2.8b-lora/checkpoint-1700')
# model_0=model_0.merge_and_unload()
# model_1=model_1.merge_and_unload()
# # model_0=MambaForCausalLM.from_pretrained('/data/ruanjh/mamba-translate-2.8b-ckpt1700lora')
# model_1=MambaForCausalLM.from_pretrained('/data/ruanjh/mamba-2.8b-hf')
# # print(model_1)
# # model_1=MambaForCausalLM.from_pretrained('/data/ruanjh/mamba-translate-2.8b-ckpt900lora')
# same_params = set()
# different_params = set()
# for ((name_0, param_0), (name_1, param_1)) in zip(model_0.named_parameters(), model_1.named_parameters()):
#     assert name_0 == name_1
#     if not (param_0 == param_1).all():
#         different_params.add(name_0)
#     else:
#         same_params.add(name_0)

# print(same_params)
# print(different_params)