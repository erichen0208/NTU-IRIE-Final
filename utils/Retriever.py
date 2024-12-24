import os
import json
import numpy as np
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
  BertTokenizerFast,
)

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.BertModel import DenseModel
from utils.ProvisionDataset import ProvisionDataset
from utils.LawDataset import LawDataset
from utils.Loss import ContrastiveLoss
from utils.bm25 import BM25
from utils.Score import Score

class Retriever:
    def __init__(self, config):
        """
        Initialize Retriever with model and tokenizer
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.is_load_model = False
        self.provision_embeddings = None
        
    def load_model(self, mode='inference'):
        self.model = DenseModel(self.config, self.device, mode) # BertModel(self.config, self.device, mode)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config['model_path'])
        self.is_load_model = True

    def generate_provision_embeddings(self, provision_list):
        config = self.config
        device = self.device

        # Check embeddings file path
        embedding_save_path = config["embeddings_save_path"]
        directory = os.path.dirname(embedding_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.load_model()
        model = self.model.to(device)
        tokenizer = self.tokenizer

        provision_embeddings_dict = {}
        for item in tqdm(provision_list, desc="Processing Provisions", unit="genere"):
            genere = item['genere']
            provisions = item['provisions']
            provision_dataset = ProvisionDataset(provisions, tokenizer, config['max_length'])
            provision_dataloader = DataLoader(provision_dataset, batch_size=64, shuffle=False)

            provision_embeddings = []
            for batch in provision_dataloader:
                provision_names, provision_inputs = batch

                provision_inputs = provision_inputs.to(device)
                attention_mask = (provision_inputs > 0).to(device)

                with torch.no_grad():  
                    provision_embedding = model(provision_inputs, attention_mask=attention_mask)

                # provision_embeddings.extend(provision_embedding)
                for name, embedding in zip(provision_names, provision_embedding):
                    provision_embeddings_dict[name] = embedding.cpu().numpy()

        np.save(config["embeddings_save_path"], provision_embeddings_dict)

        print("Provision embeddings generated and saved!")

    def load_candidate_provision_embeddings(self, candidate_list: List[str]):
        if self.provision_embeddings is None:
            self.provision_embeddings = np.load(self.config["embeddings_save_path"], allow_pickle=True).item()
        
        return {
            name: embedding
            for name, embedding in self.provision_embeddings.items()
            if name in candidate_list
        }
        
    def generate_query_embeddings(self, queries: List[str]):
        config = self.config
        device = self.device

        if self.is_load_model == False:
            self.load_model()
        model = self.model
        tokenizer = self.tokenizer

        query_embeddings = []
        for query in tqdm(queries, desc="Processing Queries", unit="query"):
            query_input = tokenizer(query, padding='max_length', max_length=config['max_length'], truncation=True, return_tensors="pt")
            query_input = query_input['input_ids'].to(device)
            attention_mask = (query_input > 0).to(device)

            with torch.no_grad():  
                query_outputs = model(query_input, attention_mask=attention_mask)
                query_embedding = query_outputs.detach().cpu().numpy()

            query_embeddings.append(query_embedding)

        return query_embeddings

    def finetune(self, train_list: List[Dict], provision_dict: Dict):
        # Hyper parameters
        learning_rate = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        max_length = self.config['max_length']
        pth_save_path = self.config['pth_save_path']
        device = self.device

        self.load_model(mode='finetune')
        model = self.model
        tokenizer = self.tokenizer

        train_data, val_data = train_test_split(train_list, test_size=0.2, random_state=42)

        train_dataset = LawDataset(train_data, provision_dict, tokenizer, max_length)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
        )

        val_dataset = LawDataset(val_data, provision_dict, tokenizer, max_length)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        loss_fn = ContrastiveLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate,  weight_decay=0.01)
        train_losses = []
        val_losses = []
        
        for epoch in tqdm(range(epochs), desc="Epoch Progress", unit="epoch"):
            # -----------------------------
            # Training Phase
            # -----------------------------
            model.train()
            total_train_loss = 0
            batch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", unit="batch", leave=False)

            for batch in batch_bar:
                # Get embeddings
                query_ids, provision_ids, rel = batch
                query_ids = query_ids.to(device)
                query_attention_mask = (query_ids > 0)
                provision_ids = provision_ids.to(device)
                provision_attention_mask = (provision_ids > 0)
                rel = rel.to(device)

                optimizer.zero_grad()

                query_embeddings = model(query_ids, query_attention_mask)
                provision_embeddings = model(provision_ids, provision_attention_mask)

                # Compute loss
                loss = loss_fn(query_embeddings, provision_embeddings, rel)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                batch_bar.set_postfix(loss=loss.item())
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_losses.append(avg_train_loss)
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
                    query_ids, provision_ids, rel = batch
                    query_ids = query_ids.to(device)
                    provision_ids = provision_ids.to(device)
                    rel = rel.to(device)
                        
                    query_embeddings = model(query_ids, attention_mask=(query_ids > 0))
                    provision_embeddings = model(provision_ids, attention_mask=(provision_ids > 0))

                    # Compute loss
                    loss = loss_fn(query_embeddings, provision_embeddings, rel)

                    total_val_loss += loss.item()
                    batch_bar.set_postfix(loss=loss.item())

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

        model.save_model(pth_save_path, epoch, optimizer, avg_val_loss)                 
        print("Finetune completed!")

        # -----------------------------
        # Plot and Save Loss Curve
        # -----------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('loss_curve.png')
        plt.show()

    def test(self, val_list):
        print(f"There are {len(val_list)} testcases")

        config = self.config
        if os.path.exists(config['embeddings_save_path']) == False:
            print("Please generate provision embeddings first!")
            return
        
        score = Score()

        queries, labels = [], []
        for data in val_list:
            query = "關鍵字:" + data["keyword"] + "標題:" + data["title"] + ("問題:" + data["question"] if data["question"] != None else "")
            queries.append(query)
            labels.append(data['label'].split(','))
        
        top_k = config["first_retrieval_topk"]
        # First retrieval using BM25
        if os.path.exists(config["first_retrieval_cache"]):
            from data.py.first_retrieval_cache import candidate_label_with_score
            candidate_labels = [[item['name'] for item in candidate] for candidate in candidate_label_with_score]
        else:      
            bm25 = BM25(top_k=top_k)
            candidate_labels = []
            candidate_label_with_score = []
            for query in tqdm(queries, desc="First retrieval (BM25)", unit=" query", total=len(queries)):
                retrieved_labels = bm25.retrieve_provisions(query) # [{"name": str, "score": float}]
                candidate_labels.append([item['name'] for item in retrieved_labels])
                candidate_label_with_score.append(retrieved_labels)
                
            precision = score.precision(candidate_labels, labels)
            recall = score.recall(candidate_labels, labels)
            f1 = score.f1(candidate_labels, labels)
            print(f"[BM25] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            with open(config["first_retrieval_cache"], 'w', encoding='utf-8') as py_file:
                py_file.write("candidate_label_with_score = ")
                py_file.write(json.dumps(candidate_label_with_score, ensure_ascii=False, indent=4))
        
        # first_retrieval_results = candidate_label_with_score
        first_retrieval_results = [] 
        for testcase in candidate_label_with_score:
            score_dict = {}
            for item in testcase:
                score_dict[item['name']] = item['score']
            first_retrieval_results.append(score_dict)
        

        # Second retrieval using bert embeddings (using candidate labels from BM25)
        query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)
        second_retrieval_results = []
        for i, [query_embedding, label] in tqdm(enumerate(zip(query_embeddings, labels)), desc="Second retrieval (BERT)", unit=" query", total=len(queries)):
  
            provisions_dict = self.load_candidate_provision_embeddings(candidate_labels[i])
            similarities = []
            embedding_space = []
            for provision_name, provision_embedding in provisions_dict.items():
                # Cosine similarity: 1 - cosine distance
                query_embedding = query_embedding.squeeze()
                similarity = 1 - cosine(query_embedding, provision_embedding)
                similarities.append({"name": provision_name, "score": similarity})
                if provision_name in label:
                    embedding_space.append([provision_embedding, 1])
                else:
                    embedding_space.append([provision_embedding, 0])

            # Sort by similarity
            sorted_provisions = sorted(similarities, key=lambda x: x['score'], reverse=True) # [(name, score)]
            second_retrieval_results.append(sorted_provisions[:top_k])

            # self.plot_embeddings(query_embedding, embedding_space, save_path=f"./plots/{i+1}.png")
        
        # Hybrid score
        retrieve_num = config["final_retrieval_topk"]
        bm25_weight = config["bm25_weight"]
        bert_weight = 1 - bm25_weight
        results = []
        for i, item in enumerate(second_retrieval_results):
            name_and_scores = []
            for j, provision in enumerate(item):
                provision_name = provision['name']
                bert_score = provision['score']
                bm25_score = first_retrieval_results[i][provision_name]
                weighted_score = bm25_weight * bm25_score + bert_weight * bert_score
                name_and_scores.append({"name": provision_name, "score": weighted_score})

            name_and_scores = sorted(name_and_scores, key=lambda x: x['score'], reverse=True)[:retrieve_num]
            results.append([item['name'] for item in name_and_scores])

        precision = score.precision(results, labels)
        recall = score.recall(results, labels)
        f1 = score.f1(results, labels)

        print(f"Retrieval number: {retrieve_num}, BM25 weight: {bm25_weight}, BERT weight: {bert_weight}")
        print(f"[Hybrid] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        print("Testing completed!")
    
    def inference(self, test_list, top_k = 100):
        print(f"There are {len(test_list)} testcases")

        config = self.config
        if os.path.exists(config['embeddings_save_path']) == False:
            print("Please generate provision embeddings first!")
            return

        queries = []
        for data in test_list:
            query = "關鍵字:" + data["keyword"] + "標題:" + data["title"] + ("問題:" + data["question"] if data["question"] != None else "")
            queries.append(query)
        
        # First retrieval using BM25
        if os.path.exists(config["first_retrieval_cache"]):
            from data.py.first_retrieval_cache.py import candidate_label_with_score
            candidate_labels = [[item['name'] for item in candidate] for candidate in candidate_label_with_score]
        else:      
            bm25 = BM25(top_k=top_k)
            candidate_labels = []
            candidate_label_with_score = []
            for query in tqdm(queries, desc="First retrieval (BM25)", unit=" query", total=len(queries)):
                retrieved_labels = bm25.retrieve_provisions(query) # [{"name": str, "score": float}]
                candidate_labels.append([item['name'] for item in retrieved_labels])
                candidate_label_with_score.append(retrieved_labels)
                
            with open(config["first_retrieval_cache"], 'w', encoding='utf-8') as py_file:
                py_file.write("candidate_label_with_score = ")
                py_file.write(json.dumps(candidate_label_with_score, ensure_ascii=False, indent=4))
        
        # make first_retrieval_results into dict
        first_retrieval_results = [] 
        for testcase in candidate_label_with_score:
            score_dict = {}
            for item in testcase:
                score_dict[item['name']] = item['score']
            first_retrieval_results.append(score_dict)

        # Second retrieval using bert embeddings (using candidate labels from BM25)
        query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)
        second_retrieval_results = []
        for i, query_embedding in tqdm(enumerate(query_embeddings), desc="Second retrieval (BERT)", unit=" query", total=len(queries)):
  
            provisions_dict = self.load_candidate_provision_embeddings(candidate_labels[i])
            similarities = []
            embedding_space = []
            for provision_name, provision_embedding in provisions_dict.items():
                # Cosine similarity: 1 - cosine distance
                query_embedding = query_embedding.squeeze()
                similarity = 1 - cosine(query_embedding, provision_embedding)
                similarities.append({"name": provision_name, "score": similarity})
                embedding_space.append([provision_embedding, None])

            # Sort by similarity
            sorted_provisions = sorted(similarities, key=lambda x: x['score'], reverse=True) # [(name, score)]
            second_retrieval_results.append(sorted_provisions[:top_k])

        retrieve_num = 3
        bm25_weight = 0.2
        bert_weight = 1 - bm25_weight
        results = []
        for i, item in enumerate(second_retrieval_results):
            name_and_scores = []
            for j, provision in enumerate(item):
                provision_name = provision['name']
                bert_score = provision['score']
                bm25_score = first_retrieval_results[i][provision_name]
                weighted_score = bm25_weight * bm25_score + bert_weight * bert_score
                name_and_scores.append({"name": provision_name, "score": weighted_score})

            name_and_scores = sorted(name_and_scores, key=lambda x: x['score'], reverse=True)[:retrieve_num]
            results.append([item['name'] for item in name_and_scores])

        self.write_submission_csv(results)
    
    def write_submission_csv(self, final_list):
        config = self.config
        with open(config['output_csv_path'], "w") as f:
            f.write("id,TARGET\n")
            for i, provisions in enumerate(final_list):
                provisions_text = ",".join(provisions)
                f.write(f"test_{i},\"{provisions_text}\"\n")

        print("Submission CSV file created!") 

    def plot_embeddings(self, query_embedding, embedding_space, save_path="embedding_plot.png", title="Query vs Label vs Other Provisions Embeddings"):
        label_embeddings = []
        other_embeddings = []
        labels = []

        for item in embedding_space:
            provision_embedding, label = item[0], item[1]
            # print(label, provision_embedding)
            if label == 1:
                label_embeddings.append(provision_embedding)
            else:
                other_embeddings.append(provision_embedding)
            labels.append(label)

        # Combine the query and embeddings for PCA
        all_embeddings = [query_embedding] + label_embeddings + other_embeddings
        all_labels = [1] + [1] * len(label_embeddings) + [0] * len(other_embeddings)  # 1 for label matches, 0 for others

        # Reduce dimensionality for plotting (using PCA for 2D visualization)
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(all_embeddings)

        # Create scatter plot
        plt.figure(figsize=(8, 6))

        for i, emb in enumerate(other_embeddings):
            plt.scatter(reduced_embeddings[len(label_embeddings)+1 + i][0], reduced_embeddings[len(label_embeddings)+1 + i][1], color='green', label='Other' if i == 0 else "", s=30, marker='o')  # Other points
        # Highlight query (in red), label matches (in green), and others (in blue)
        plt.scatter(reduced_embeddings[0][0], reduced_embeddings[0][1], color='red', label='Query', s=100, marker='o')  # Query point
        for i, emb in enumerate(label_embeddings):
            plt.scatter(reduced_embeddings[i+1][0], reduced_embeddings[i+1][1], color='blue', label='Label' if i == 0 else "", s=100, marker='x')  # Label matches
        
        # Add labels and title
        plt.title(title)
        plt.legend()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Save the plot as a .png file
        plt.savefig(save_path)
        plt.close()  # Close the figure after saving it to avoid display