import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import torch.multiprocessing as mp
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
    tokenizer = AutoTokenizer.from_pretrained("/data/ruanjh/mamba-chat")
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

    model = MambaLMHeadModel.from_pretrained(
        "/data/ruanjh/mamba-chat",device=device, dtype=torch.float16
    )
    return model,tokenizer

def get_pred(rank,data,dictt,model_dir):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(device,model_dir)
    en = [d["en"] for d in data]
    result=[]
    for e in tqdm(en):
        messages = []
        user_message = f"Translate this into German: {e}"
        messages.append(dict(role="user", content=user_message))
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(device)
        out = model.generate(
            input_ids=input_ids,
            max_length=512,
            temperature=0.9,
            top_p=0.7,
            eos_token_id=tokenizer.eos_token_id,
            # cg=True,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)[0].split("<|assistant|>\n")[-1]
        decoded=decoded.replace('\n','\\n')
        result.append(decoded)
    dictt[f'{rank}']=result
    
    # dist.destroy_process_group()
    
def split_list(lst, n):
    avg = len(lst) / float(n)
    return [lst[int(avg * i):int(avg * (i + 1))] for i in range(n)]

if __name__=='__main__':

    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    data_all = [data_sample for data_sample in test_data]
    data_subsets = split_list(data_all,world_size)
    out_path='/data/ruanjh/best_training_method/iwslt17/mt_mamba_chat.de'
    model_dir='/data/ruanjh/mamba-chat'
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