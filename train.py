
from transformers import MambaForCausalLM,AutoTokenizer,Seq2SeqTrainer,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,Trainer,TrainingArguments
import torch
from dataset import MyDataset
import json
from plot import plot_loss
from peft import LoraConfig,get_peft_model
model_dir='/data/ruanjh/mamba-2.8b-hf/'
output_dir='./mamba-translate'
tokenizer=AutoTokenizer.from_pretrained(model_dir,padding_side='left')

model=MambaForCausalLM.from_pretrained(model_dir,torch_dtype=torch.bfloat16)

collator=DataCollatorForSeq2Seq(tokenizer,model)

with open('/data/ruanjh/best_training_method/iwslt17/train.json') as f:
    train_data=json.load(f)
train_dataset=MyDataset(train_data,tokenizer)

with open('/data/ruanjh/best_training_method/iwslt17/validation.json') as f:
    eval_data=json.load(f)
eval_dataset=MyDataset(eval_data,tokenizer)

# from torch.utils.data import DataLoader
# dataloader=DataLoader(dataset=eval_dataset,batch_size=2,collate_fn=collator)
# for d in dataloader:
#     print(d)
#     exit()

lora_config =  LoraConfig(
        r=64,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none",
        use_rslora=True,
)
# model.add_adapter(lora_config)
model = get_peft_model(model, lora_config)

trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            overwrite_output_dir =True,
            remove_unused_columns =False,
            gradient_accumulation_steps=16,
            # gradient_checkpointing=True,
            #------------------------------
            evaluation_strategy='steps',
            eval_delay=100,
            eval_steps =50,
            #-------------------------------
            save_strategy ='steps',
            save_steps = 50,
            save_total_limit =3,
            load_best_model_at_end=True,
            #--------------------------------
            dataloader_num_workers =10,
            learning_rate=2e-3,
            num_train_epochs=30,
            # auto_find_batch_size=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size =8,
            output_dir="/data/ruanjh/mamba-translate-2.8b-lora",
            logging_steps=5,
            bf16=True,
            prediction_loss_only=True,
            lr_scheduler_type="cosine",
            # torch_compile=True,
            # torch_compile_backend='inductor',
            # torch_compile_mode='max-autotune',
            optim='adamw_apex_fused',
            # save_safetensors =False,
        ),
        data_collator=collator,
    )

trainer.train()

plot_loss(output_dir)

