from torch.utils.data import Dataset
import torch


# 这个dataset适合json文件的读取,其中键包含{'en','de'},只适用于decoder-only模型
class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        src = [d["en"] for d in data]
        tgt = [d["de"] for d in data]
        input_id = tokenizer(src)
        tgt_id = tokenizer(tgt)
        result_matrix = [row + [tokenizer.eos_token_id] for row in tgt_id.input_ids]
        self.data = {
            "input_ids": input_id.input_ids,
            "labels": result_matrix,
        }
        self.tokenizer=tokenizer

    def __getitem__(self, index):
        
        return {
            # 1是pad token,我拿来当源语言目标语言分隔符了,虽然我感觉用一些固定的描述也可以
            "input_ids": self.data["input_ids"][index]+[1]+self.data["labels"][index],
            "labels": (len(self.data["input_ids"][index])+1)*[-100]+self.data["labels"][index],
        }

    def __len__(self):
        # print(len(self.data['input_ids']))
        return len(self.data["input_ids"])

class TestDataset(Dataset):
    def __init__(self, data, tokenizer):
        src = [d["en"] for d in data]
        tgt = [d["de"] for d in data]
        input_id = tokenizer(src)
        self.data = {
            "input_ids": input_id.input_ids,
        }
        self.tokenizer=tokenizer

    def __getitem__(self, index):
        
        return {
            # 1是pad token,我拿来当源语言目标语言分隔符了,虽然我感觉用一些固定的描述也可以
            "input_ids": self.data["input_ids"][index]+[1],
        }

    def __len__(self):
        # print(len(self.data['input_ids']))
        return len(self.data["input_ids"])