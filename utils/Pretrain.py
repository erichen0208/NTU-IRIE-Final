import torch
from transformers import (
    BertTokenizer, 
    BertForMaskedLM, 
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)
import jieba

class LegalCorpusPretrainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load base model and tokenizer
        self.config = config    
        self.tokenizer = BertTokenizer.from_pretrained(config["tokenizer_name"])
        self.model = BertForMaskedLM.from_pretrained(config["model_name"])
        
        # Corpus path
        self.legal_corpus_path = './data/pretrain_data.txt'
        self.output_dir = config["pretrained_model_path"]

    def jieba_tokenize(self, text):
        """
        Tokenize text using Jieba.
        Converts a string into a list of Chinese word tokens.
        """
        words = jieba.cut(text)
        return " ".join(words) 

    def prepare_dataset(self):
        # Create dataset from legal corpus
        # with open(self.legal_corpus_path, 'r', encoding='utf-8') as f:
        #     lines = f.readlines()

        # # Apply Jieba tokenization to each line
        # tokenized_lines = [self.jieba_tokenize(line.strip()) for line in lines]

        # # Save the tokenized corpus as a temporary file
        # tokenized_corpus_path = './data/tokenized_pretrain_data.txt'
        # with open(tokenized_corpus_path, 'w', encoding='utf-8') as f:
        #     f.write("\n".join(tokenized_lines))

        tokenized_corpus_path = './data/tokenized_pretrain_data.txt'
        # Create dataset from legal corpus
        dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=tokenized_corpus_path,
            block_size=128  # Adjust based on your document lengths
        )
        return dataset

    def prepare_data_collator(self):
        # Data collator for masked language modeling
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True,  # Masked Language Modeling
            mlm_probability=0.15  # Mask 15% of tokens
        )

    def train(self, epochs=3, batch_size=16, learning_rate=5e-5):
        # Prepare dataset and data collator
        train_dataset = self.prepare_dataset()
        data_collator = self.prepare_data_collator()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            no_cuda=False, 
            # device=self.device
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
        )

        # Start training
        trainer.train()

        # Save final model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(f"{self.output_dir}/tokenizer")

def pretrain(config):
    # Initialize and train
    pretrainer = LegalCorpusPretrainer(config)
    pretrainer.train()


if __name__ == '__main__':
    pretrain()