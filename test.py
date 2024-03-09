from transformers import MambaForCausalLM,AutoTokenizer,Seq2SeqTrainer,DataCollatorForSeq2Seq,Seq2SeqTrainingArguments,Trainer,TrainingArguments
import torch
from dataset import MyDataset
import json
from plot import plot_loss


def load_model_and_tokenizer(device,model_dir='/data/ruanjh/mamba-translate/checkpoint-3700'):
    model=BartForConditionalGeneration.from_pretrained(model_dir,torch_dtype='auto',)
    model.to(device)
    tokenizer=BartTokenizerFast.from_pretrained(model_dir)
    return model,tokenizer