import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import torch.multiprocessing as mp
from dataset import TestDataset,MyDataset
from torch.utils.data import DataLoader
import json

from transformers import DataCollatorForSeq2Seq
with open('/data/ruanjh/best_training_method/iwslt17/test.json') as f:
    test_data=json.load(f)
    



def load_model_and_tokenizer(device,model_dir):
    tokenizer = AutoTokenizer.from_pretrained("/data/ruanjh/mamba-chat")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    model = MambaLMHeadModel.from_pretrained(
        "/data/ruanjh/mamba-chat",device=device, dtype=torch.float16
    )
    return model,tokenizer

def get_pred(rank,data,dictt,model_dir):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(device,model_dir)
    dataset=TestDataset(data,tokenizer)
    collator= DataCollatorForSeq2Seq(tokenizer,model=model,padding=True)
    
    dataloader=tqdm(DataLoader(dataset,2,collate_fn=collator,pin_memory=True,num_workers=4))
    result=[]
    for input in dataloader:
        input.to(device)
        output = model.generate(
                    input_ids=input['input_ids'],
                    max_length=256,
                    temperature=0.9,
                    top_p=0.7,
                    eos_token_id=tokenizer.eos_token_id,
                    cg=True,
                )

        temp_result=tokenizer.batch_decode(output,skip_special_tokens=True)
        # print('temp_result',[temp_result])
        pred = [x.split('\nGerman: ')[-1] for x in temp_result]
        # print(pred)
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
    out_path='/data/ruanjh/best_training_method/iwslt17/mt_mamba_chat-lora2.8bckpt900.de'
    model_dir='mamba-2.8b-lorackpt900'
    processes = []
    manager = mp.Manager()
    dict = manager.dict()
    for rank in range(world_size):
        p = mp.Process(target=get_pred, args=(rank,data_subsets[rank],dict,model_dir))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    with open(out_path, "w", encoding="utf-8") as f:
        for rank in range(world_size):
            for r in dict[f'{rank}']:
                f.write(r+'\n')