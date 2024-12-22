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

        # print(token_ids)
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

        # print(query)
        # print(provision_content)
        # print(rel)

        # Tokenizer
        # input_ids = self.tokenize_jieba(query, provision_content)
        # attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        # rel = torch.tensor(rel, dtype=torch.float)

        query_input_ids = self.tokenizer(query, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        provision_content_input_ids = self.tokenizer(provision_content, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        rel = torch.tensor(rel, dtype=torch.float)

        return query_input_ids['input_ids'].squeeze(0), provision_content_input_ids['input_ids'].squeeze(0), rel
    
    def custom_collate_fn(self, batch):
        input_ids, attention_masks, rel = zip(*batch)

        # Convert rel to a tensor
        rel = torch.tensor(rel, dtype=torch.float)

        return input_ids, attention_masks, rel

# class LawDataset(Dataset):
#     def __init__(self, data: List[Dict], provision_dict: Dict, tokenizer, max_length):
#         self.data = data
#         self.provisions = provision_dict
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def generate_provision_contents(self, labels):
#         contents = []
#         for label in labels:
#             if label in self.provisions.keys():
#                 content = self.provisions[label]['content']
#                 # examples = self.provisions[label]['example']
#                 concatenated_sentence = content
#                 # if (len(examples) > 0):
#                 #     concatenated_sentence += " [SEP] ".join(examples)
#                 contents.append(concatenated_sentence)
#         return contents

#     def generate_negative_labels(self, positive_labels):
#         negative_labels = []
#         for label in self.provisions.keys():
#             if label not in positive_labels:
#                 negative_labels.append(label)
#         return random.sample(negative_labels, len(positive_labels))
    
#     def __getitem__(self, idx):
#         title = self.data[idx]["title"] if self.data[idx]["title"] != None else ""
#         question = self.data[idx]["question"] if self.data[idx]["question"] != None else ""
#         query = title + '\n' + question
         
#         positive_labels = self.data[idx]["label"].split(",")  
#         positive_provision_contents = self.generate_provision_contents(positive_labels)
        
#         # Can't find provisions in dictionary
#         if len(positive_provision_contents) == 0: 
#             for label in positive_labels:
#                 positive_provision_contents.append(label)
        
#         # Hard negative sampling
#         negative_labels = self.data[idx]["neg_label"].split(",") 
#         negative_provision_contents = self.generate_provision_contents(negative_labels)\
        
#         # Can't find provisions in dictionary
#         if len(negative_provision_contents) == 0: 
#             for label in negative_labels:
#                 negative_provision_contents.append(label)
        
#         # Tokenizer
#         query_tokens = self.tokenizer(query, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
#         positive_provision_tokens = self.tokenizer(positive_provision_contents, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
#         negative_provision_tokens = self.tokenizer(negative_provision_contents, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")

#         return query_tokens['input_ids'].squeeze(), positive_provision_tokens['input_ids'], negative_provision_tokens['input_ids']
    
#     def custom_collate_fn(self, batch):
        queries, positive_provisions, negative_provisions = zip(*batch)

        # 1️. Pad and stack queries (shape: [batch_size, max_query_length])
        queries = torch.stack(queries)  
        
        # 2️. Pad and stack positive provisions (shape: [batch_size, max_num_positive, max_seq_len])
        max_num_pos = max(pos.shape[0] for pos in positive_provisions) 
        padded_positive_provisions = []
        for pos in positive_provisions:
            if pos.shape[0] < max_num_pos: 
                pad = torch.zeros(max_num_pos - pos.shape[0], pos.shape[1], dtype=pos.dtype)
                pos = torch.cat([pos, pad], dim=0) 
            padded_positive_provisions.append(pos)

        positive_provisions = torch.stack(padded_positive_provisions)  

        # 3️. Pad and stack negative provisions (shape: [batch_size, max_num_negative, max_seq_len])
        max_num_negative = max(negative.shape[0] for negative in negative_provisions) 
        padded_negative_provisions = []
        for negative in negative_provisions:
            if negative.shape[0] < max_num_negative:  
                pad = torch.zeros(max_num_negative - negative.shape[0], negative.shape[1], dtype=negative.dtype) 
                negative = torch.cat([negative, pad], dim=0) 
            padded_negative_provisions.append(negative)
        negative_provisions = torch.stack(padded_negative_provisions) 
        
        return queries, positive_provisions, negative_provisions

    