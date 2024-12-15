from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
import random
import itertools

class LawDataset(Dataset):
    def __init__(self, provisions, data, tokenizer, max_length):
        self.provisions = provisions
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title = self.data[idx]["title"] if self.data[idx]["title"] != None else ""
        question = self.data[idx]["question"] if self.data[idx]["question"] != None else ""
        query = title + '\n' + question

        labels = self.data[idx]["label"].split(",")  
        pos_provision_contents = []
        for label in labels:
            if label in self.provisions.keys():
                pos_provision_contents.append(self.provisions[label])
        
        if len(pos_provision_contents) == 0:
            for label in labels:
                pos_provision_contents.append(label)
            
        provision_keys = (label for label in self.provisions.keys() if label not in labels)
        limited_keys = list(itertools.islice(provision_keys, 1000))
        random_provisions = random.sample(limited_keys, 5)
        neg_provision_contents = [self.provisions[key] for key in random_provisions]
        # neg_provision_contents = [self.provisions[label] for label in self.provisions.keys() if label not in labels]
        
        query_tokens = self.tokenizer(query, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        pos_provision_tokens = self.tokenizer(pos_provision_contents, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        neg_provision_tokens = self.tokenizer(neg_provision_contents, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")

        return query_tokens['input_ids'].squeeze(), pos_provision_tokens['input_ids'], neg_provision_tokens['input_ids']
    
    def custom_collate_fn(self, batch, device):
        queries, pos_provisions, neg_provisions = zip(*batch)

        # 1️⃣ Pad and stack queries (shape: [batch_size, max_query_length])
        queries = torch.stack(queries)  # Each query has shape (seq_len,)
        
        # 2️⃣ Pad and stack positive provisions (shape: [batch_size, max_num_pos, max_seq_len])
        max_num_pos = max(pos.shape[0] for pos in pos_provisions)  # Find the maximum number of positive provisions in the batch
        padded_pos_provisions = []
        for pos in pos_provisions:
            if pos.shape[0] < max_num_pos:  # Pad provision list to max_num_pos
                pad = torch.zeros(max_num_pos - pos.shape[0], pos.shape[1], dtype=pos.dtype)  # (num_pad, seq_len)
                pos = torch.cat([pos, pad], dim=0)  # Pad to (max_num_pos, seq_len)
            padded_pos_provisions.append(pos)

        pos_provisions = torch.stack(padded_pos_provisions)  # (batch_size, max_num_pos, max_seq_len)

        # 3️⃣ Pad and stack negative provisions (shape: [batch_size, max_num_neg, max_seq_len])
        max_num_neg = max(neg.shape[0] for neg in neg_provisions)  # Find the maximum number of negative provisions in the batch
        padded_neg_provisions = []
        for neg in neg_provisions:
            if neg.shape[0] < max_num_neg:  # Pad provision list to max_num_neg
                pad = torch.zeros(max_num_neg - neg.shape[0], neg.shape[1], dtype=neg.dtype)  # (num_pad, seq_len)
                neg = torch.cat([neg, pad], dim=0)  # Pad to (max_num_neg, seq_len)
            padded_neg_provisions.append(neg)
        neg_provisions = torch.stack(padded_neg_provisions)  # (batch_size, max_num_neg, max_seq_len)
        
        return queries, pos_provisions, neg_provisions

    
