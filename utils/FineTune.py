import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def load_data(file_path: str):
    """
    Loads and processes the dataset.
    
    Args:
        file_path (str): Path to the dataset file.
        
    Returns:
        DatasetDict: Loaded dataset object.
    """
    dataset = load_dataset("json", data_files=file_path)  # Assuming the dataset is in JSON format
    return dataset


def tokenize_function(examples, tokenizer):
    """
    Tokenizes the input examples using the specified tokenizer.
    
    Args:
        examples (dict): Batch of text samples.
        tokenizer (AutoTokenizer): Tokenizer to convert text to tokens.
        
    Returns:
        dict: Tokenized examples.
    """
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


def prepare_data(dataset, tokenizer):
    """
    Tokenizes and prepares the dataset for training.
    
    Args:
        dataset (DatasetDict): The raw dataset to process.
        tokenizer (AutoTokenizer): The tokenizer to use.
        
    Returns:
        DatasetDict: Tokenized dataset ready for training.
    """
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    return tokenized_dataset


def fine_tune(file_path: str, output_dir: str, epochs: int = 3, batch_size: int = 8, logging_steps: int = 500):
    """
    Fine-tunes the TAIDE-LX-7B model on a custom dataset.
    
    Args:
        file_path (str): Path to the dataset file.
        output_dir (str): Directory where the model will be saved.
        epochs (int, optional): Number of epochs to train. Defaults to 3.
        batch_size (int, optional): Batch size for training. Defaults to 8.
        logging_steps (int, optional): Steps after which training logs will be printed. Defaults to 500.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1")
    model = AutoModelForCausalLM.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1")
    
    # Load and prepare the dataset
    dataset = load_data(file_path)
    tokenized_dataset = prepare_data(dataset, tokenizer)
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # Set to True if you want to use Masked Language Modeling instead
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=1000,
        save_total_limit=2,  # Only keep last 2 model checkpoints
        logging_dir=f'{output_dir}/logs',
        logging_steps=logging_steps,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True if torch.cuda.is_available() else False,  # Use mixed precision if CUDA is available
        push_to_hub=False  # Set to True if you want to push the model to the Hugging Face hub
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),  # Optional if validation set is available
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    FILE_PATH = "path/to/your/dataset.json"  # Replace with the path to your dataset
    OUTPUT_DIR = "./taide_finetuned_model"  # Directory where the fine-tuned model will be saved
    EPOCHS = 3  # Number of epochs
    BATCH_SIZE = 8  # Batch size
    
    fine_tune(
        file_path=FILE_PATH, 
        output_dir=OUTPUT_DIR, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )
