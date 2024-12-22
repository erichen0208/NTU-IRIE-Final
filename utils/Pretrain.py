import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BertTokenizer, 
    BertForMaskedLM, 
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments
)

class LegalCorpusPretrainer:
    def __init__(self, mode, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load base model and tokenizer
        self.mode = mode
        self.config = config 

        if mode == 'llm':
            self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
            self.model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        elif mode == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config["tokenizer_name"])
            self.model = BertForMaskedLM.from_pretrained(config["model_name"])

    def prepare_dataset(self):
        train_data_path = self.config["train_data_path"]
        # Create dataset from legal corpus
        dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path=train_data_path,
            block_size=self.config["max_length"]
        )
        return dataset

    def prepare_data_collator(self):
        # Data collator for masked language modeling
        if self.mode == 'bert':
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=True,  # Masked Language Modeling
                mlm_probability=0.15  # Mask 15% of tokens
            )
        elif self.mode == 'llm':
            return DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False
            )

    def train(self):
        # Training parameters
        epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        learning_rate = self.config["learning_rate"]
        output_dir = self.config["model_save_path"]

        # Prepare dataset and data collator
        train_dataset = self.prepare_dataset()
        data_collator = self.prepare_data_collator()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
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
            fp16=True,  
            gradient_accumulation_steps=8,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
        )

        # Start training
        print("Starting training...")
        trainer.train()

        # Save final model
        self.model.save_pretrained(self.config["model_save_path"])
        self.tokenizer.save_pretrained(self.config["model_save_path"])
        print("Training completed.")

def pretrain(mode, config):
    # Initialize and train
    pretrainer = LegalCorpusPretrainer(mode, config)
    pretrainer.train()

if __name__ == '__main__':
    pretrain()