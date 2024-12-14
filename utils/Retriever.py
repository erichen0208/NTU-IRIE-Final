import os
import pickle
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import (
  BertTokenizerFast,
  AutoModel,
)
from tqdm import tqdm

from typing import List, Dict, Tuple
import numpy as np
from utils.LawDataset import LawDataset
from utils.Loss import Loss
from utils.DualBertEncoder import DualBERTEncoder

class Retriever:
    def __init__(self, config):
        """
        Initialize Retriever with model and tokenizer
        """
        self.tokenizer = BertTokenizerFast.from_pretrained(config['tokenizer_name'])
        self.model = AutoModel.from_pretrained(config['model_name'])
        self.embeddings = {}
        self.config = config
        self.cosine_similarity = Loss()
    
    def retrieve_relevant_laws(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant laws for a given query.
        """
        processed_query = self.preprocess_chinese_text(query)
        query_embedding = self.get_bert_embedding(processed_query)

        similarities = []
        for law_id, law_embedding in self.embeddings.items():
            similarity = self.cosine_similarity(query_embedding, law_embedding)[0][0]
            similarities.append((law_id, similarity))

        # Sort by similarity and return top_k results
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Generate BERT embedding for a given text.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config['max_length'],
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token embedding as the representation
        return outputs.last_hidden_state[:, 0, :]

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


    def train(self, provisions: Dict, training_data: List[Dict]):
        print("Training started...")

        # parameters
        learning_rate = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        max_length = self.config['max_length']
        model_save_path = self.config['model_save_path']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = self.model.to(device) # DualBERTEncoder(self.model).to(device) 
        tokenizer = self.tokenizer

        dataset = LawDataset(provisions, training_data, tokenizer, max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=lambda batch: self.custom_collate_fn(batch, device)
        )

        loss_fn = Loss()
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in tqdm(range(epochs), desc="Epoch Progress", unit="epoch"):
            total_loss = 0
            
            # Add a progress bar for the batch loop
            batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False)
            for batch in batch_bar:
                query, pos_provision, neg_provision = batch
                query, pos_provision, neg_provision = query.to(device), pos_provision.to(device), neg_provision.to(device)
                
                # query_embedding, pos_provision_embeddings = model(query, pos_provision)
                # _, neg_provision_embeddings = model(query, neg_provision)

                query_outputs = model(query, attention_mask=(query > 0))
                query_embedding = query_outputs.last_hidden_state[:, 0, :]

                pos_provision_embeddings = []
                for i in range(pos_provision.shape[1]):
                    provision_input = pos_provision[:, i, :]  # Shape: [batch_size, seq_length]
                    attention_mask = (provision_input > 0).long() # Shape [batch_size, seq_length]
                    pos_provision_outputs = model(provision_input, attention_mask=attention_mask)
                    pos_provision_embeddings.append(pos_provision_outputs.last_hidden_state[:, 0, :])  # Shape: [batch_size, hidden_size]
                pos_provision_embeddings = torch.stack(pos_provision_embeddings, dim=1)  # Shape: [batch_size, num_provisions, hidden_size]

                neg_provision_embeddings = []
                for i in range(neg_provision.shape[1]):
                    provision_input = neg_provision[:, i, :]
                    attention_mask = (provision_input > 0).long()
                    neg_provision_outputs = model(provision_input, attention_mask=attention_mask)
                    neg_provision_embeddings.append(neg_provision_outputs.last_hidden_state[:, 0, :])
                neg_provision_embeddings = torch.stack(neg_provision_embeddings, dim=1)

                # Compute loss
                loss = loss_fn(query_embedding, pos_provision_embeddings, neg_provision_embeddings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

                # Update batch bar with loss info
                batch_bar.set_postfix(loss=loss.item())

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

        model.save_pretrained(model_save_path)                 
        print("Training completed!")

    def evaluate(self):
        print("Evaluating the model...")
        # Placeholder for evaluation logic
    
