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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams

from typing import List, Dict
from annoy import AnnoyIndex
import numpy as np

import os
from scipy.stats import trim_mean
# import faiss

from utils.ProvisionDataset import ProvisionDataset
from utils.LawDataset import LawDataset
from utils.Loss import Loss

class Retriever:
    def __init__(self, config):
        """
        Initialize Retriever with model and tokenizer
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # self.tokenizer = BertTokenizerFast.from_pretrained(config['tokenizer_name'])
        # self.model = AutoModel.from_pretrained(config['model_name'])
        
    def load_model(self, pretrained = False):
        if pretrained:
            self.model = AutoModel.from_pretrained(self.config['model_name'])
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config['tokenizer_name'])
        else:
            config = AutoConfig.from_pretrained(self.config['pretrained_model_path'])
            self.model = AutoModel.from_pretrained(self.config['pretrained_model_path'], config=config)
            self.tokenizer = BertTokenizerFast.from_pretrained(f"{self.config['pretrained_model_path']}/tokenizer")
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_bytes = num_params * 4  
        model_size_gb = model_size_bytes / (1024 ** 3)
        print(f"Model size: {model_size_gb:.4f} GB")

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

    def generate_provision_embeddings(self, provision_list):
        config = self.config
        device = self.device

        self.load_model(pretrained=False)
        model = self.model.to(device)
        tokenizer = self.tokenizer

        genere_list = [] # [299]
        genere_embeddings = [] # [299, 768]
        embedding_space = [] # [299, num_of_provisions, 768]
        for item in tqdm(provision_list, desc="Processing Provisions", unit="genere"):
            genere = item['genere']
            provisions = item['provisions']
            provision_dataset = ProvisionDataset(provisions, tokenizer, config['max_length'])
            provision_dataloader = DataLoader(provision_dataset, batch_size=64, shuffle=False)

            provision_embeddings = []
            for provision_inputs in provision_dataloader:
                provision_inputs = provision_inputs.to(device)
                attention_mask = (provision_inputs > 0).to(device)

                with torch.no_grad():  
                    provision_outputs = model(provision_inputs, attention_mask=attention_mask)

                provision_embedding = provision_outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                provision_embeddings.extend(provision_embedding)

            genere_list.append(genere)
            genere_embeddings.append(trim_mean(provision_embeddings, proportiontocut=0.2, axis=0))
            embedding_space.append(provision_embeddings)

        
        # Check embeddings file path
        embedding_save_path = config["embeddings_save_path"]
        directory = os.path.dirname(embedding_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        dim = self.config['embedding_dim']

        # Global embedding space for 299 generes
        genere_embeddings = np.vstack(genere_embeddings)
        genere_embeddings = self.normalize_embeddings(genere_embeddings)

        genere_index = AnnoyIndex(dim, 'angular')  
        for i, embedding in enumerate(provision_embeddings):
            genere_index.add_item(i, embedding)

        num_trees = 10  
        genere_index.build(num_trees)
        genere_index.save(f"{embedding_save_path}/global.ann")

        # Plot
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(genere_embeddings)
        plt.figure(figsize=(10, 7))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=10, alpha=0.7)

        for i, (x, y) in enumerate(embeddings_2d):
            plt.text(x + 0.01, y + 0.01, i, fontsize=9, color='red')

        plt.title('PCA of Annoy Embedding Space')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        plt.savefig('global.png', dpi=300)
        plt.close()

        # Local embedding space for each genere
        for genere, genere_embedding_list in zip(genere_list ,embedding_space):
            genere_embedding_list = np.vstack(genere_embedding_list)
            genere_embedding_list = self.normalize_embeddings(genere_embedding_list)

            this_genere_index = AnnoyIndex(dim, 'angular')
            for i, embedding in enumerate(genere_embedding_list):
                this_genere_index.add_item(i, embedding)

            num_trees = 10
            this_genere_index.build(num_trees)
            this_genere_index.save(f"{embedding_save_path}/{genere}.ann")

        print("Provision embeddings generated and saved!")

    def load_provision_embeddings(self, genere = "global"):
        index_path = self.config['embeddings_save_path']
        index = AnnoyIndex(self.config['embedding_dim'], 'angular')  # 'angular' for cosine similarity
        index.load(f"{index_path}/{genere}.ann") 
        return index
    
    def generate_query_embeddings(self, queries: List[str]):
        config = self.config
        device = self.device

        self.load_model(pretrained=False)
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

    def train(self, train_list: List[Dict], provision_dict: Dict):
        # Hyper parameters
        learning_rate = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['epochs']
        max_length = self.config['max_length']
        model_save_path = self.config['model_save_path']
        device = self.device

        self.load_model(pretrained=False)
        model = self.model.to(device)
        tokenizer = self.tokenizer

        train_data, val_data = train_test_split(train_list, test_size=0.2, random_state=42)

        train_dataset = LawDataset(train_data, provision_dict, tokenizer, max_length)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=lambda batch: train_dataset.custom_collate_fn(batch, device)
        )

        val_dataset = LawDataset(val_data, provision_dict, tokenizer, max_length)
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

    def test(self, provision_list: List[Dict], queries: List, labels: List[List], k = 20):
        config = self.config
        if os.path.exists(config['embeddings_save_path']) == False:
            print("Please generate provision embeddings first!")
            return
        
        index = self.load_provision_embeddings()
        query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)

        for i, [query_embedding, label] in enumerate(zip(query_embeddings, labels)):
            print(f"\nQuery {i+1} {queries[i]}\n")
            print(f"Top {k} nearest provisions:")

            # Annoy search for top k nearest neighbors (search returns indices and distances)
            indices, distances = index.get_nns_by_vector(query_embedding, k, include_distances=True)

            for j in range(k):
                provision_idx = indices[j]  # Index of the provision
                distance = distances[j]  # Distance to this provision
                provision_name = provision_list[provision_idx]['genere']  # Retrieve the provision name from the law list
                print(f"  Provision: {provision_name}, Distance: {distance:.4f}")
                
                provision_genere_idx = indices[j]
                provision_genere = provision_list[provision_genere_idx]['genere'] 
                index_genere = self.load_provision_embeddings(genere=provision_genere)
                provision_indices, provision_distances = index_genere.get_nns_by_vector(query_embedding, k, include_distances=True)
                for x in range(min(k // 4, len(provision_indices))):
                    provision_idx = provision_indices[x]
                    provision_distance = provision_distances[x]
                    provision_name = provision_list[provision_genere_idx]['provisions'][provision_idx]['name']
                    print(f"  Provision: {provision_name}, Distance: {provision_distance:.4f}")
            print(f"  Ground Truth: {label}")
            print("----------------------------------")

        print("Testing completed!")
    
    def inference(self, provision_list: List[Dict], queries: List, k = 20):
        config = self.config
        if os.path.exists(config['embeddings_save_path']) == False:
            print("Please generate provision embeddings first!")
            return
        
        index = self.load_provision_embeddings(genere="global")
        query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)

        final_list = []
        for i, query_embedding in enumerate(query_embeddings):
            # Annoy search for top k nearest neighbors (search returns indices and distances)
            indices, distances = index.get_nns_by_vector(query_embedding, k, include_distances=True)
            candidate_provisions = []
            for j in range(k):
                provision_genere_idx = indices[j] 
                provision_genere = provision_list[provision_genere_idx]['genere'] 
                index_genere = self.load_provision_embeddings(genere=provision_genere)
                provision_indices, provision_distances = index_genere.get_nns_by_vector(query_embedding, k, include_distances=True)
                for x in range(min(k // 4, len(provision_indices))):
                    provision_idx = provision_indices[x]
                    provision_distance = provision_distances[x]
                    provision_name = provision_list[provision_genere_idx]['provisions'][provision_idx]['name']
                    candidate_provisions.append([provision_name, provision_distance]) # name, distance
                
            candidate_provisions = sorted(candidate_provisions, key=lambda x: x[1])
            end = 1
            while end < 5 and candidate_provisions[end][1] - candidate_provisions[0][1] < 0.05:
                end += 1
            final_list.append([element[0] for element in candidate_provisions[:end]])
            
        self.write_submission_csv(final_list)

    def normalize_embeddings(self, embeddings): 
        # embeddings: [num, 786]
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
    
    def write_submission_csv(self, final_list):
        config = self.config
        with open(config['output_csv_path'], "w") as f:
            f.write("id,TARGET\n")
            for i, provisions in enumerate(final_list):
                provisions_text = ",".join(provisions)
                f.write(f"test_{i},\"{provisions_text}\"\n")

        print("Submission CSV file created!") 

    def test_embeddings(self, k = 10):
        from data.py.provision_list import provision_list
        
        genere = "民法"
        for i in range(len(provision_list)):
            if provision_list[i]['genere'] == genere:
                genere_index = i
        index = self.load_provision_embeddings(genere=genere)

        # queries = [
        #     "為加強道路交通管理。維護交通秩序。確保交通安全。制定本條例。",
        #     "道路交通管理、處罰。依本條例規定。本條例未規定者。依其他法律規定。",
        #     "本條例用詞。定義如下。\n一、道路。指公路、街道、巷衖、廣場、騎樓、走廊或其他供公眾通行之地方。\n二、車道。指以劃分島、護欄或標線劃定道路之部分。及其他供車輛行駛之道路。\n三、人行道。指為專供行人通行之騎樓、走廊。及劃設供行人行走之地面道路。與人行天橋及人行地下道。\n四、行人穿越道。指在道路上以標線劃設。供行人穿越道路之地方。\n五、標誌。指管制道路交通。表示警告、禁制、指示。而以文字或圖案繪製之標牌。\n六、標線。指管制道路交通。表示警告、禁制、指示。而在路面或其他設施上劃設之線條、圖形或文字。\n七、號誌。指管制道路交通。表示行進、注意、停止。而以手勢、光色、音響、文字等指示之訊號。\n八、車輛。指非依軌道電力架設。而以原動機行駛之汽車（包括機車）、慢車及其他行駛於道路之動力車輛。\n九、大眾捷運系統車輛。指大眾捷運法所定大眾捷運系統使用之專用動力車輛。\n十、臨時停車。指車輛因上、下人、客。裝卸物品。其停止時間未滿三分鐘。保持立即行駛之狀態。\n十一、停車。指車輛停放於道路兩側或停車場所。而不立即行駛。",
        # ]
        # queries = [
        #     "我們制定這個規定，主要是為了加強道路管理，保持交通秩序，並確保大家的行車安全。",
        #     "道路交通管理和處罰會依照這條規定來執行。如果這條規定沒說到的，會按照其他法律來處理。",
        #     "這條規定裡的詞語定義是這樣的：\n\n1. 道路：指的是像公路、街道、巷子、廣場、騎樓、走廊等，任何供大家通行的地方。\n2. 車道：就是那種被分隔島、護欄或標線劃分開來的車行道。\n3. 人行道：專門給行人走的地方，包括騎樓、走廊和地面道路，也包含人行天橋和地下道。\n4. 行人穿越道：就是路面上標出來的行人過馬路的地方。\n5. 標誌：指的是道路交通管制的標牌，上面會有文字或圖案，提醒、警告或者給出指示。\n6. 標線：就是那些在路面或設施上劃出來的線條、圖形或文字，通常是用來表示警告、指示或管制交通的。\n7. 號誌：就是那些交通指示信號，可以是手勢、燈號、聲音或者文字。\n8. 車輛：指的是那些用原動機推動的車子，比如汽車（包括機車）、慢車或者其他交通工具。\n9. 大眾捷運系統車輛：是指大眾捷運法規定的大眾捷運車輛。\n10. 臨時停車：就是車輛因為上下人或裝卸東西，停在路邊，但時間不超過三分鐘，隨時可以開走。\n11. 停車：指車輛停在路邊或者停車場，並且不會馬上開走。",
        # ]

        queries = [
            "【時事】假房東與無權處分法律問題\n兩名男子在去年12月在臉書上找到一間位於台北市大安區的公寓，以每月二萬元的租金與洪姓女子簽訂租約，並一次付清八個月的房租和四萬元的押金，共計二十萬元。\n今年七月，兩名男子發現洪姓女子已經失聯，而且有一對自稱是房屋代管的男女來到門口，指控他們侵入住居，並要求他們立即搬離。\n兩名男子向警方報案，警方調查發現，洪姓女子是假房東，她使用偽造的權狀來欺騙房客，並收取不當利益。警方已掌握洪姓女子的身分資料，正傳喚她到案說明，全案朝詐欺、偽造文書等罪嫌偵辦中。\n請問有關民法上無權處分與善意取得的法律效果：在假房東案件中，真房東、假房東和房客各自的權利和義務是什麼？他們之間的法律關係如何判斷？\n如果房客在租屋時知道或應該知道假房東沒有出租的權利，他們還能否主張善意取得房屋使用權？為什麼？\n如果真房東在假房東出租房屋後，承認了假房東對房客的處分，那麼房客是否能繼續居住房屋？為什麼？",
            "兩名男子於去年12月，透過臉書平台找到位於台北市大安區的一處公寓，並以每月新台幣20,000元的租金與洪姓女子訂立租賃契約，另一次性支付八個月租金及四萬元押金，共計20萬元。然而，今年7月，兩名男子發現洪姓女子已失聯，且有一對自稱為房屋代管人的男女前來該公寓，指控兩人非法居住並要求即刻搬遷。兩名男子隨後報案處理，警方經調查後發現，洪姓女子並非該房屋的所有人，其以偽造之權狀對外進行租賃，並向房客收取不正當利益。警方已掌握洪姓女子的身份資料，並正召喚其到案說明，案件已依詐欺罪及偽造文書罪等罪名展開調查。"
        ]

        query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)
        
        for i, query_embedding in enumerate(query_embeddings):
            print(f"\nQuery {i+1} {queries[i]}\n")
            print(f"Top {k} nearest provisions:")

            # Annoy search for top k nearest neighbors (search returns indices and distances)
            indices, distances = index.get_nns_by_vector(query_embedding, k, include_distances=True)
            for j in range(k):
                provision_idx = indices[j]  # Index of the provision
                distance = distances[j]  # Distance to this provision
                provision_name = provision_list[genere_index]["provisions"][provision_idx]["name"]  # Retrieve the provision name from the law list
                print(f"  Provision: {provision_name}, Distance: {distance:.4f}")
