from torch.utils.data import Dataset
from transformers import AutoTokenizer
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
        
        provision_keys = (label for label in self.provisions.keys() if label not in labels)
        limited_keys = list(itertools.islice(provision_keys, 1000))
        random_provisions = random.sample(limited_keys, 5)
        neg_provision_contents = [self.provisions[key] for key in random_provisions]
        # neg_provision_contents = [self.provisions[label] for label in self.provisions.keys() if label not in labels]
        
        query_tokens = self.tokenizer(query, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        pos_provision_tokens = self.tokenizer(pos_provision_contents, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        neg_provision_tokens = self.tokenizer(neg_provision_contents, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")

        return query_tokens['input_ids'].squeeze(), pos_provision_tokens['input_ids'], neg_provision_tokens['input_ids']

    
