import json
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("/data/ruanjh/mamba-chat")
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

model = MambaLMHeadModel.from_pretrained(
    "/data/ruanjh/mamba-chat",device=device, dtype=torch.float16
)

# user_message = """
# Translate this into German: {input}
# """
with open("/data/ruanjh/best_training_method/iwslt17/test.json") as f:
    data = json.load(f)

en = [d["en"] for d in data]
result=[]
from tqdm import tqdm
with open('/data/ruanjh/best_training_method/iwslt17/mt_mamba_chat.de','w') as o:
    for e in tqdm(en):
        messages = []
        user_message = f"Translate this into German: {e}"
        messages.append(dict(role="user", content=user_message))
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to("cuda")
        out = model.generate(
            input_ids=input_ids,
            max_length=512,
            temperature=0.9,
            top_p=0.7,
            eos_token_id=tokenizer.eos_token_id,
            cg=True,
        )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)[0].split("<|assistant|>\n")[-1]
        messages.append(dict(role="assistant", content=decoded[0].split("<|assistant|>\n")[-1]))
        decoded=decoded.replace('\n','\\n')
        o.write(decoded+'\n')
    # result.append()

