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
    



def load_model_and_tokenizer(device,model_dir):
    model=MambaForCausalLM.from_pretrained(model_dir,torch_dtype='auto',)
    model.to(device)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,padding_side='left')
    return model,tokenizer

def get_pred(rank,out_path,data,dict,model_dir):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(device,model_dir)
    dataset=TestDataset(data,tokenizer)
    collator= DataCollatorForSeq2Seq(tokenizer,model=model,padding=True)
    
    dataloader=tqdm(DataLoader(dataset,2,collate_fn=collator))
    result=[]
    for input in dataloader:
        input.to(device)
        output = model.generate(
                    input_ids=input['input_ids'],
                    attention_mask=input['attention_mask'],
                    num_beams=5,
                    do_sample=False,
                    temperature=1.0,
                    max_new_tokens=256,
                )

        temp_result=tokenizer.batch_decode(output,skip_special_tokens=True)
        print('temp_result',[temp_result])
        pred = [x.split('\nGerman: ')[-1] for x in temp_result]
        print(pred)
        result+=pred
    dict[f'{rank}']=result
    
    # dist.destroy_process_group()
    
def split_list(lst, n):
    avg = len(lst) / float(n)
    return [lst[int(avg * i):int(avg * (i + 1))] for i in range(n)]

if __name__=='__main__':

    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    data_all = [data_sample for data_sample in test_data]
    data_subsets = split_list(data_all,world_size)
    out_path='/data/ruanjh/best_training_method/iwslt17/mt_mamba-2_8b.de'
    model_dir='/data/ruanjh/mamba-translate-2.8b/checkpoint-850'
    processes = []
    manager = mp.Manager()
    dict = manager.dict()
    for rank in range(world_size):
        p = mp.Process(target=get_pred, args=(rank,out_path,data_subsets[rank],dict,model_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    with open(out_path, "w", encoding="utf-8") as f:
        for rank in range(world_size):
            for r in dict[f'{rank}']:
                f.write(r.replace('\n','\\n')+'\n')