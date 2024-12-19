from torch.utils.data import Dataset

class ProvisionDataset(Dataset):
    def __init__(self, provisions, tokenizer, max_length):
        self.provisions = provisions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.provisions)

    def __getitem__(self, idx):
        provision_content = self.provisions[idx]['content']
        provision_examples = self.provisions[idx]['example']
        input_sentence = provision_content
        if len(provision_examples) > 0:
            input_sentence += " [SEP] ".join(provision_examples)

        provision_input = self.tokenizer(input_sentence, padding='max_length', max_length=self.max_length, truncation=True, return_tensors="pt")
        return provision_input['input_ids'].squeeze(0)
    