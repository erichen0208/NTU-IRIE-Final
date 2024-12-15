import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
  BertTokenizerFast,
  AutoModel,
  AutoConfig,
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from typing import List, Dict, Tuple
import numpy as np
from utils.LawDataset import LawDataset
from utils.Loss import Loss

import os
import faiss
from utils.law_list import law_list
from utils.ProvisionDataset import ProvisionDataset

class Retriever:
    def __init__(self, config):
        """
        Initialize Retriever with model and tokenizer
        """
        self.config = config
        # self.tokenizer = BertTokenizerFast.from_pretrained(config['tokenizer_name'])
        # self.model = AutoModel.from_pretrained(config['model_name'])
        
    def load_model(self, pretrained = False):
        if pretrained:
            self.model = AutoModel.from_pretrained(self.config['model_name'])
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config['tokenizer_name'])
        else:
            config = AutoConfig.from_pretrained(self.config['model_save_path'])
            self.model = AutoModel.from_pretrained(self.config['model_save_path'], config=config)
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config['tokenizer_name'])

    def get_embeddings(self, batch, model, device):
        query, pos_provision, neg_provision = batch
        query, pos_provision, neg_provision = query.to(device), pos_provision.to(device), neg_provision.to(device)

        query_outputs = model(query, attention_mask=(query > 0))
        query_embeddings = query_outputs.last_hidden_state[:, 0, :]
        # print(query_embeddings.shape)

        pos_provision_embeddings = []
        for i in range(pos_provision.shape[1]):
            provision_input = pos_provision[:, i, :]  # Shape: [batch_size, seq_length]
            attention_mask = (provision_input > 0).long().to(device) # Shape [batch_size, seq_length]
            pos_provision_outputs = model(provision_input, attention_mask=attention_mask)
            pos_provision_embeddings.append(pos_provision_outputs.last_hidden_state[:, 0, :])  # Shape: [batch_size, hidden_size]
        pos_provision_embeddings = torch.stack(pos_provision_embeddings, dim=1)  # Shape: [batch_size, num_provisions, hidden_size]
        # print(pos_provision_embeddings.shape)

        neg_provision_embeddings = []
        for i in range(neg_provision.shape[1]):
            provision_input = neg_provision[:, i, :]
            attention_mask = (provision_input > 0).long().to(device)
            neg_provision_outputs = model(provision_input, attention_mask=attention_mask)
            neg_provision_embeddings.append(neg_provision_outputs.last_hidden_state[:, 0, :])
        neg_provision_embeddings = torch.stack(neg_provision_embeddings, dim=1)

        return query_embeddings, pos_provision_embeddings, neg_provision_embeddings

    def train(self, provisions: Dict, training_data: List[Dict]):
        # Hyper parameters
        learning_rate = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        max_length = self.config['max_length']
        model_save_path = self.config['model_save_path']

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.load_model(pretrained=True)
        model = self.model.to(device)
        tokenizer = self.tokenizer

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_bytes = num_params * 4  
        model_size_gb = model_size_bytes / (1024 ** 3)
        print(f"Model size: {model_size_gb:.4f} GB")

        train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)

        train_dataset = LawDataset(provisions, train_data, tokenizer, max_length)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=lambda batch: train_dataset.custom_collate_fn(batch, device)
        )

        val_dataset = LawDataset(provisions, val_data, tokenizer, max_length)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: val_dataset.custom_collate_fn(batch, device)
        )

        loss_fn = Loss()
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        for epoch in tqdm(range(epochs), desc="Epoch Progress", unit="epoch"):
            # -----------------------------
            # Training Phase
            # -----------------------------
            model.train()
            total_train_loss = 0
            batch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", unit="batch", leave=False)

            for batch in batch_bar:
                # Get embeddings
                query_embeddings, pos_provision_embeddings, neg_provision_embeddings = self.get_embeddings(batch, model, device)

                # Compute loss
                loss = loss_fn(query_embeddings, pos_provision_embeddings, neg_provision_embeddings)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                batch_bar.set_postfix(loss=loss.item())
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

            # -----------------------------
            # Validation Phase
            # -----------------------------
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                batch_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", unit="batch", leave=False)
                for batch in batch_bar:
                    # Get embeddings
                    query_embeddings, pos_provision_embeddings, neg_provision_embeddings = self.get_embeddings(batch, model, device)
                    
                    # Compute loss
                    loss = loss_fn(query_embeddings, pos_provision_embeddings, neg_provision_embeddings)
                    total_val_loss += loss.item()
                    batch_bar.set_postfix(loss=loss.item())

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

        model.save_pretrained(model_save_path)                 
        print("Training completed!")

    def generate_provision_embeddings(self):
        config = self.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.load_model(pretrained=False)
        model = self.model.to(device)
        tokenizer = self.tokenizer

        provision_dataset = ProvisionDataset(law_list, tokenizer, config['max_length'])
        provision_dataloader = DataLoader(provision_dataset, batch_size=8, shuffle=False)

        model = model.to(device)
        provision_embeddings = []
        for provision_inputs in tqdm(provision_dataloader, desc="Processing Provisions", unit="batch"):
            provision_inputs = provision_inputs.to(device)
            attention_mask = (provision_inputs > 0).to(device)

            with torch.no_grad():  
                provision_outputs = model(provision_inputs, attention_mask=attention_mask)

            provision_embedding = provision_outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            provision_embeddings.extend(provision_embedding)

        # FAISS: Initialize the index
        provision_embeddings = np.vstack(provision_embeddings)  
        provision_embeddings = self.normalize_embeddings(provision_embeddings)
        dim = provision_embeddings.shape[1]
        print(f"Embedding dimension: {dim}, Number of provisions: {provision_embeddings.shape[0]}")
        
        embedding_save_path = config["embeddings_save_path"]
        directory = os.path.dirname(embedding_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        faiss_index = faiss.IndexFlatL2(dim) 
        faiss_index.add(provision_embeddings) 
        faiss.write_index(faiss_index, config["embeddings_save_path"])

        print("Provision embeddings generated and saved!")

    def load_provision_embeddings(self):
        index_path = self.config['embeddings_save_path']
        index = faiss.read_index(index_path)
        return index
    
    def generate_query_embeddings(self, queries: List[str]):
        config = self.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        self.load_model(pretrained=True)
        model = self.model.to(device)
        tokenizer = self.tokenizer

        query_embeddings = []
        for query in tqdm(queries, desc="Processing Querys", unit="query"):
            query_input = tokenizer(query, padding='max_length', max_length=config['max_length'], truncation=True, return_tensors="pt")
            query_input = query_input['input_ids'].to(device)
            attention_mask = (query_input > 0).to(device)

            with torch.no_grad():  
                query_outputs = model(query_input, attention_mask=attention_mask)

            query_embedding = query_outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().flatten()
            query_embeddings.append(query_embedding)

        query_embeddings = np.vstack(query_embeddings)
        return self.normalize_embeddings(query_embeddings)
    
    def test(self, queries, k = 5):
        config = self.config
        if os.path.exists(config['embeddings_save_path']) == False:
            print("Please generate provision embeddings first!")
            return
        
        # Load provision embeddings
        index = self.load_provision_embeddings()
        res = faiss.StandardGpuResources() 
        index = faiss.index_cpu_to_gpu(res, 0, index)

        query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)
        distances, indices = index.search(query_embeddings, k) # Shape (N, k)

        for i, query in enumerate(queries):
            print(f"\nQuery {i+1} query\n")
            print(f"Top {k} nearest provisions:")
            for j in range(k):
                provision_idx = indices[i][j]  # Index of the provision
                distance = distances[i][j]  # Distance to this provision
                provision_name = law_list[provision_idx]['provision']  # Retrieve the content from the law_list
                print(f"  Provision: {provision_name}, Distance: {distance:.4f}")

        print("Testing completed!")
    
    def inference(self, queries, k = 5):
        config = self.config
        if os.path.exists(config['embeddings_save_path']) == False:
            print("Please generate provision embeddings first!")
            return

        index = self.load_provision_embeddings()
        res = faiss.StandardGpuResources() 
        index = faiss.index_cpu_to_gpu(res, 0, index)

        query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)
        distances, indices = index.search(query_embeddings, k) # Shape (N, k)

        provision_list = []
        for i, index in enumerate(indices):
            best_result_distance = distances[i][0]
            provisions = []
            for j in range(k):
                if distances[i][j] - best_result_distance > 0.003:
                    break
                provision_idx = indices[i][j]  # Index of the provision
                provision_name = law_list[provision_idx]['provision'] 
                provisions.append(provision_name)
            provision_list.append(provisions)
        
        self.write_submission_csv(provision_list)

    def normalize_embeddings(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        return normalized_embeddings
    
    def calculate_f1_score(self, ground_truth, predictions):
        batch_f1_scores = []

        for i in range(ground_truth.shape[0]):  # Loop over each query
            true_positives = torch.sum((predictions[i] == 1) & (ground_truth[i] == 1))
            false_positives = torch.sum((predictions[i] == 1) & (ground_truth[i] == 0))
            false_negatives = torch.sum((predictions[i] == 0) & (ground_truth[i] == 1))
            
            if true_positives == 0:
                precision, recall, f1 = 0.0, 0.0, 0.0
            else:
                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
            
            batch_f1_scores.append(f1)
        
        f1_score = torch.mean(torch.tensor(batch_f1_scores))  # Average F1 score across queries
        return f1_score
    
    def write_submission_csv(self, provisions_list):
        config = self.config
        with open(config['output_csv_path'], "w") as f:
            f.write("id,TARGET\n")
            for i, provisions in enumerate(provisions_list):
                provisions_text = ",".join(provisions)
                f.write(f"test_{i},\"{provisions_text}\"\n")

        print("Submission CSV file created!") 
