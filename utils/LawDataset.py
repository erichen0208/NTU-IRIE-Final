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
    
    def jieba_tokenize(self, text):
        if isinstance(text, list):
            return [self.jieba_tokenize(item) for item in text]
        else:
            words = jieba.cut(text)
            return " ".join(words) 

    def generate_provision_contents(self, labels):
        contents = []
        for label in labels:
            if label in self.provisions.keys():
                content = self.provisions[label]['content']
                examples = self.provisions[label]['example']
                concatenated_sentence = self.jieba_tokenize(content)
                if (len(examples) > 0):
                    concatenated_sentence += " [SEP] ".join(self.jieba_tokenize(examples))
                contents.append(concatenated_sentence)
        return contents

    def generate_negative_labels(self, positive_labels):
        negative_labels = []
        for label in self.provisions.keys():
            if label not in positive_labels:
                negative_labels.append(label)
        return random.sample(negative_labels, len(positive_labels))
    
    def __getitem__(self, idx):
        title = self.data[idx]["title"] if self.data[idx]["title"] != None else ""
        question = self.data[idx]["question"] if self.data[idx]["question"] != None else ""
        query = self.jieba_tokenize(title + '\n' + question)
         
        positive_labels = self.data[idx]["label"].split(",")  
        positive_provision_contents = self.generate_provision_contents(positive_labels)
        
        # Can't find provisions in dictionary
        if len(positive_provision_contents) == 0: 
            for label in positive_labels:
                positive_provision_contents.append(label)
            
        # Hard negative sampling
        negative_labels = self.generate_negative_labels(positive_labels)
        negative_provision_contents = self.generate_provision_contents(negative_labels)
        
        # Tokenizer
        query_tokens = self.tokenizer(query, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        positive_provision_tokens = self.tokenizer(positive_provision_contents, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        negative_provision_tokens = self.tokenizer(negative_provision_contents, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")

        return query_tokens['input_ids'].squeeze(), positive_provision_tokens['input_ids'], negative_provision_tokens['input_ids']
    
    def custom_collate_fn(self, batch):
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

    
