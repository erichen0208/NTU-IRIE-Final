from typing import List, Dict
import torch
from torch.utils.data import Dataset
import random
import jieba

class LawDataset(Dataset):
    def __init__(self, data: List[Dict], provision_dict: Dict, tokenizer, max_length):
        self.data = data
        self.provisions = provision_dict
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def tokenize_jieba(self, s1, s2):
        s1_tokens = jieba.lcut(s1)
        s1_token_ids = self.tokenizer.convert_tokens_to_ids(s1_tokens)
        if len(s1_token_ids) > self.max_length // 2:
            s1_token_ids = s1_token_ids[:self.max_length // 2]

        s2_tokens = jieba.lcut(s2)
        s2_token_ids = self.tokenizer.convert_tokens_to_ids(s2_tokens)

        token_ids = [self.tokenizer.cls_token_id] + s1_token_ids + [self.tokenizer.sep_token_id] + s2_token_ids + [self.tokenizer.sep_token_id]

        # Pad to max_length if necessary
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
        else:
            token_ids = token_ids[:self.max_length]

        return torch.tensor(token_ids)
    
    def get_provision_content(self, label):
        if label not in self.provisions.keys():
            provision_content = label
        else:
            provision_content = self.provisions[label]["content"] 
            if self.provisions[label]["example"] != None:
                provision_content +=  "".join(self.provisions[label]["example"])

            provision_content = provision_content.replace("\n", "")
            provision_content = provision_content.replace("\r", "")
            provision_content = provision_content.replace(" ", "")

        return provision_content
    
    def __getitem__(self, idx):
        query =  "關鍵字:" + self.data[idx]["keyword"] + "標題:" + self.data[idx]["title"] + "問題:" + self.data[idx]["question"]
        provision_content = self.get_provision_content(self.data[idx]["label"])
        rel = self.data[idx]["rel"]

        # Tokenizer
        # query_input_ids = self.tokenize_jieba(query, provision_content)

        query_input_ids = self.tokenizer(query, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        provision_content_input_ids = self.tokenizer(provision_content, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        rel = torch.tensor(rel, dtype=torch.float)

        return query_input_ids['input_ids'].squeeze(0), provision_content_input_ids['input_ids'].squeeze(0), rel
    
