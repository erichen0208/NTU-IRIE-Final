import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
  BertTokenizerFast,
  AutoModel,
  AutoConfig,
  AutoTokenizer, 
  AutoModelForCausalLM
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import rcParams

from typing import List, Dict
from annoy import AnnoyIndex
import numpy as np
import jieba
from scipy.spatial.distance import cosine

import os
from scipy.stats import trim_mean
# import faiss

from utils.BertModel import BertModel, DenseModel
from utils.ProvisionDataset import ProvisionDataset
from utils.LawDataset import LawDataset
from utils.Loss import BCELoss, ContrastiveLoss

class Retriever:
    def __init__(self, config):
        """
        Initialize Retriever with model and tokenizer
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.is_load_model = False
        # self.tokenizer = BertTokenizerFast.from_pretrained(config['tokenizer_name'])
        # self.model = AutoModel.from_pretrained(config['model_name'])
        
    def load_model(self, mode='inference'):
        self.model = DenseModel(self.config, self.device, mode) # BertModel(self.config, self.device, mode)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config['model_path'])
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_bytes = num_params * 4  
        model_size_gb = model_size_bytes / (1024 ** 3)
        print(f"Model size: {model_size_gb:.4f} GB")
        self.is_load_model = True

    def get_embeddings(self, batch, model, device):
        """
        Args:
            batch: Tuple containing query, positive provisions, and negative provisions.
                Shapes: 
                - query: [batch_size, max_length]
                - pos_provision: [batch_size, num_pos, max_length]
                - neg_provision: [batch_size, num_neg, max_length]        
        Returns:
            Tuple of embeddings:
            - query_embeddings: [batch_size, hidden_size]
            - pos_provision_embeddings: [batch_size, num_pos, hidden_size]
            - neg_provision_embeddings: [batch_size, num_neg, hidden_size]
        """
        
        query, pos_provision, neg_provision = batch
        query, pos_provision, neg_provision = query.to(device), pos_provision.to(device), neg_provision.to(device)

        query_embeddings = model(query, attention_mask=(query > 0))
        # query_embeddings = query_outputs.last_hidden_state[:, 0, :]
        # print(query_embeddings.shape)

        pos_provision_embeddings = []
        for i in range(pos_provision.shape[1]):
            provision_input = pos_provision[:, i, :]  # Shape: [batch_size, seq_length]
            attention_mask = (provision_input > 0).long().to(device) # Shape [batch_size, seq_length]
            embedding = model(provision_input, attention_mask=attention_mask)
            pos_provision_embeddings.append(embedding)
            # pos_provision_embeddings.append(pos_provision_outputs.last_hidden_state[:, 0, :])  # Shape: [batch_size, hidden_size]
        pos_provision_embeddings = torch.stack(pos_provision_embeddings, dim=1)  # Shape: [batch_size, num_provisions, hidden_size]
        # print(pos_provision_embeddings.shape)

        neg_provision_embeddings = []
        for i in range(neg_provision.shape[1]):
            provision_input = neg_provision[:, i, :]
            attention_mask = (provision_input > 0).long().to(device)
            embedding = model(provision_input, attention_mask=attention_mask)
            neg_provision_embeddings.append(embedding)
        neg_provision_embeddings = torch.stack(neg_provision_embeddings, dim=1)

        return query_embeddings, pos_provision_embeddings, neg_provision_embeddings

    def generate_provision_embeddings(self, provision_list):
        config = self.config
        device = self.device

        # Check embeddings file path
        embedding_save_path = config["embeddings_save_path"]
        directory = os.path.dirname(embedding_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        dim = self.config['embedding_dim']

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


            # this_genere_index = AnnoyIndex(dim, 'angular')
            # for i, embedding in enumerate(provision_embeddings):
            #     this_genere_index.add_item(i, embedding)

            # num_trees = 50
            # this_genere_index.build(num_trees)
            # this_genere_index.save(f"{embedding_save_path}/{genere}.ann")
        np.save('./embeddings/provision_embeddings.npy', provision_embeddings_dict)

        print("Provision embeddings generated and saved!")

        # provision_dataset = ProvisionDataset(provision_list, tokenizer, config['max_length'])
        # provision_dataloader = DataLoader(provision_dataset, batch_size=128, shuffle=False)
        # provision_embeddings = []
        # for batch in tqdm(provision_dataloader, desc=f"Processing Provisions", unit="batch", leave=False):
        #     provision_inputs = batch
        #     provision_inputs = provision_inputs.to(device)
        #     attention_mask = (provision_inputs > 0).to(device)

        #     with torch.no_grad():  
        #         provision_embedding = model(provision_inputs, attention_mask=attention_mask)
        #         provision_embedding = provision_embedding.detach().cpu().numpy()

        #     # provision_embedding = provision_outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        #     provision_embeddings.extend(provision_embedding)
        
        # # print(len(provision_embeddings))
        # # Check embeddings file path
        # embedding_save_path = self.config["embeddings_save_path"]
        # directory = os.path.dirname(embedding_save_path)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)

        # dim = self.config['embedding_dim']
        # index = AnnoyIndex(dim, 'angular')  
        # for i, embedding in enumerate(provision_embeddings):
        #     index.add_item(i, embedding)

        # num_trees = 10  
        # index.build(num_trees)
        # index.save(f"{embedding_save_path}/global.ann")

        # from data.py.provision_list import provision_list as law_list
        # each_len = [len(law['provisions']) for law in law_list] # [299, each_number_of_law]
        
        # # Perform PCA
        # pca = PCA(n_components=2)
        # embeddings_2d = pca.fit_transform(provision_embeddings)  # Reduced to (20000+, 2)

        # # Plot with unique colors per class
        # plt.figure(figsize=(12, 8))
        # num_classes = len(each_len)
        # colors = plt.cm.tab10(np.linspace(0, 1, num_classes))  # Define distinct colors for classes

        # start = 0
        # for i, l in enumerate(each_len):
        #     end = start + l
        #     # Scatter plot for each class
        #     plt.scatter(
        #         embeddings_2d[start:end, 0],
        #         embeddings_2d[start:end, 1],
        #         s=10,
        #         alpha=0.7,
        #         color=colors[i],
        #         label=f'Class {i}'
        #     )
        #     start = end

        # plt.title('PCA of Law Provision Embeddings')
        # plt.xlabel('PCA Component 1')
        # plt.ylabel('PCA Component 2')
        # plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), title='Law Classes')
        # plt.tight_layout()
        # plt.savefig('global_colored_by_class.png', dpi=300)
        # plt.close()

        print("Provision embeddings generated and saved!")

    def load_provision_embeddings(self, genere = "global"):
        index_path = self.config['embeddings_save_path']
        index = AnnoyIndex(self.config['embedding_dim'], 'angular')  # 'angular' for cosine similarity
        index.load(f"{index_path}/{genere}.ann") 
        return index
    
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
                query_embedding = query_outputs.detach().cpu().numpy().flatten()

            query_embeddings.append(query_embedding)

        query_embeddings = np.vstack(query_embeddings)
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
        
        for epoch in tqdm(range(epochs), desc="Epoch Progress", unit="epoch"):
            # -----------------------------
            # Training Phase
            # -----------------------------
            model.train()
            total_train_loss = 0
            batch_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", unit="batch", leave=False)

            for batch in batch_bar:
                # Get embeddings
                # query_embeddings, pos_provision_embeddings, neg_provision_embeddings = self.get_embeddings(batch, model, device)

                query_ids, provision_ids, rel = batch
                query_ids = query_ids.to(device)
                query_attention_mask = (query_ids > 0)
                provision_ids = provision_ids.to(device)
                provision_attention_mask = (provision_ids > 0)
                rel = rel.to(device)

                optimizer.zero_grad()

                # print(query_ids.shape, query_attention_mask.shape, provision_ids.shape, rel.shape)
                query_embeddings = model(query_ids, query_attention_mask)
                provision_embeddings = model(provision_ids, provision_attention_mask)

                # input_ids, attention_mask, rel = input_ids.to(device), attention_mask.to(device), rel.to(device)
                # predicted_scores = model(input_ids, attention_mask)
                # predicted_scores = predicted_scores.squeeze(1)

                # Compute loss
                # loss = loss_fn(predicted_scores, rel)
                loss = loss_fn(query_embeddings, provision_embeddings, rel)
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
                    # query_embeddings, pos_provision_embeddings, neg_provision_embeddings = self.get_embeddings(batch, model, device)
                    # input_ids, attention_mask, rel = batch
                    # input_ids, attention_mask, rel = input_ids.to(device), attention_mask.to(device), rel.to(device)
                    # predicted_scores = model(input_ids, attention_mask)
                    # predicted_scores = predicted_scores.squeeze(1)

                    query_ids, provision_ids, rel = batch
                    query_ids = query_ids.to(device)
                    provision_ids = provision_ids.to(device)
                    rel = rel.to(device)
                        
                    query_embeddings = model(query_ids, attention_mask=(query_ids > 0))
                    provision_embeddings = model(provision_ids, attention_mask=(provision_ids > 0))

                    # Compute loss
                    # loss = loss_fn(predicted_scores, rel)
                    loss = loss_fn(query_embeddings, provision_embeddings, rel)

                    # loss, similarities = loss_fn(query_embeddings, pos_provision_embeddings, neg_provision_embeddings)
                    total_val_loss += loss.item()
                    batch_bar.set_postfix(loss=loss.item())

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

        model.save_model(pth_save_path, epoch, optimizer, avg_val_loss)                 
        print("Finetune completed!")

    def test(self, provision_list: List[Dict], queries: List, labels: List[List], k = 30):
        config = self.config
        if os.path.exists(config['embeddings_save_path']) == False:
            print("Please generate provision embeddings first!")
            return
        
        genere = "民法"
        provision_embeddings = np.load('./embeddings/provision_embeddings.npy', allow_pickle=True).item()

        input_datas = [
            # {"id": 21, "title": "醫生如何合法修改病例，怎樣是違法竄改？醫療糾紛中應該由誰負舉證責任？", "question": "看到新聞說醫療糾紛占民事訴訟中很大一塊，病例常常是訴訟中雙方爭執的事情。\n1.醫師發現病例記載有疏漏，可否事後修改？要遵守哪些規定才合法？\n2.如果病人認為醫生違法竄改病歷，是病人要證明「有竄改的事實」？還是病人只要提出，反過來由醫院證明「沒有竄改」？\n3.如果醫生違法竄改病歷，法律上要負什麼責任呢？", "label": "中華民國刑法第215條,醫師法第12條,醫療法第104條,醫療法第67條,醫療法第68條,醫療法第74條", "keyword": ": 醫師、病歷、醫療糾紛、訴訟、責任、糾紛、雙方。"}
            # {"id": 24, "title": "公司工作規則加班是以半小時為單位，未足半小時就不記加班，這樣是否違反勞基法呢？", "question": "加班的情況應該是依照事情處理進度來看，很難正好滿足整點、整點半。\n1.法令有規定計算加班的最小單位一定要是小時、半小時、分﹒.﹒嗎？\n2.公司工作規則可以說「加班是以半小時為單位，未足半小時不記入加班」嗎？是否違背了「實際上有加班就應該獲得加班費」的精神？", "label": "勞動基準法第1條,勞動基準法第24條,勞動基準法第84條之1", "keyword": ": 加班、計時、半小時、小時、分、勞基法。"}
            {"id": 32, "title": "無法對未成年者提起刑事附帶民事賠償，對被告的家長提起民事賠償可以嗎？", "question": "您好，因為刑事判決已經定案，但因無法對未成年者提起刑事附帶民事賠償，想請問，對被告的家長提起民事賠償是成立的嗎？\n且想請問，若是對方同意賠償金額，是以匯款／還是需要到法院面對面呢？\n再者，若是對方不同意賠償金額，是否也需要到法院與被告面對面談判呢？", "label": "刑事訴訟法第487條,少年事件處理法第2條,民法第184條,民法第187條,公證法第13條,民事訴訟法第406條之1,民事訴訟法第407條之1,民事訴訟法第415條之1、民事訴訟法第417條、民事訴訟法第418條,強制執行法第4條,民事訴訟法第406條", "keyword": ": 未成年、刑事、民事、匯款、法院、匯款。"}
        ]
        queries, labels = [], []
        for data in input_datas:
            query = "關鍵字:" + data["keyword"] + "標題:" + data["title"] + "問題:" + data["question"]
            queries.append(query)
            labels.append(data['label'].split(','))

        query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)
        
        for i, [query_embedding, label] in enumerate(zip(query_embeddings, labels)):
            print(f"\nQuery {i+1} {queries[i]}\n")
            print(f"Top nearest provisions:")

            provisions_dict = {k: v for k, v in provision_embeddings.items() if k.startswith(genere)}
            similarities = {}
            embedding_space = []
            for provision_name, provision_embedding in provisions_dict.items():
                # Cosine similarity: 1 - cosine distance
                similarity = 1 - cosine(query_embedding, provision_embedding)
                similarities[provision_name] = similarity
                if provision_name in label:
                    embedding_space.append([provision_embedding, 1])
                else:
                    embedding_space.append([provision_embedding, 0])

            # Sort by similarity
            sorted_provisions = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            for j, (provision_name, similarity) in enumerate(sorted_provisions[:10]):
                print(f"  {j+1}. {provision_name} (Similarity: {similarity:.4f})")
                
            print(f"  Ground Truth: {label}")
            print("----------------------------------")
            self.plot_embeddings(query_embedding, embedding_space)

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

    def test_embeddings(self, k = 20):
        import random
        from data.py.provision_list_1d import provision_list_1d as provision_list
        
        index = self.load_provision_embeddings(genere='global')

        items = [random.choice(provision_list) for _ in range(10)]
        names = [item['name'] for item in items]
        queries = [item['content'] for item in items]
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

        # queries = [
        #     "【時事】假房東與無權處分法律問題\n兩名男子在去年12月在臉書上找到一間位於台北市大安區的公寓，以每月二萬元的租金與洪姓女子簽訂租約，並一次付清八個月的房租和四萬元的押金，共計二十萬元。\n今年七月，兩名男子發現洪姓女子已經失聯，而且有一對自稱是房屋代管的男女來到門口，指控他們侵入住居，並要求他們立即搬離。\n兩名男子向警方報案，警方調查發現，洪姓女子是假房東，她使用偽造的權狀來欺騙房客，並收取不當利益。警方已掌握洪姓女子的身分資料，正傳喚她到案說明，全案朝詐欺、偽造文書等罪嫌偵辦中。\n請問有關民法上無權處分與善意取得的法律效果：在假房東案件中，真房東、假房東和房客各自的權利和義務是什麼？他們之間的法律關係如何判斷？\n如果房客在租屋時知道或應該知道假房東沒有出租的權利，他們還能否主張善意取得房屋使用權？為什麼？\n如果真房東在假房東出租房屋後，承認了假房東對房客的處分，那麼房客是否能繼續居住房屋？為什麼？",
        #     "兩名男子於去年12月，透過臉書平台找到位於台北市大安區的一處公寓，並以每月新台幣20,000元的租金與洪姓女子訂立租賃契約，另一次性支付八個月租金及四萬元押金，共計20萬元。然而，今年7月，兩名男子發現洪姓女子已失聯，且有一對自稱為房屋代管人的男女前來該公寓，指控兩人非法居住並要求即刻搬遷。兩名男子隨後報案處理，警方經調查後發現，洪姓女子並非該房屋的所有人，其以偽造之權狀對外進行租賃，並向房客收取不正當利益。警方已掌握洪姓女子的身份資料，並正召喚其到案說明，案件已依詐欺罪及偽造文書罪等罪名展開調查。"
        # ]

        query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)
        
        for i, query_embedding in enumerate(query_embeddings):
            print(f"\nQuery {i+1} {names[i]}\n")
            print(f"Top {k} nearest provisions:")

            # Annoy search for top k nearest neighbors (search returns indices and distances)
            indices, distances = index.get_nns_by_vector(query_embedding, k, include_distances=True)
            for j in range(k):
                provision_idx = indices[j]  # Index of the provision
                distance = distances[j]  # Distance to this provision
                provision_name = provision_list[provision_idx]["name"]  # Retrieve the provision name from the law list
                print(f"  Provision: {provision_name}, Distance: {distance:.4f}")

    def generate_neg_labels(self):
        from data.py.train_list import train_list
        from data.py.provision_list_1d import provision_list_1d as provision_list
        import json
        with open("./data/json/provision_dict.json", 'r', encoding='utf-8') as f:
            provision_dict = json.load(f)
 
        index = self.load_provision_embeddings(genere='global')

        new_train_list = []
        for i, item in tqdm(enumerate(train_list), desc="Processing Train List", unit="item"):
            names = item['label'].split(',')
            # contents = [provision_dict[name]['content'] for name in names]
            contents = []
            for name in names:
                if name in provision_dict.keys():
                    contents.append(provision_dict[name]['content'])
            if len(contents) == 0:
                contents = names

            query_embeddings = self.generate_query_embeddings(contents) # [num, dim]

            for j, query_embedding in enumerate(query_embeddings):
                indices, distances = index.get_nns_by_vector(query_embedding, 5000, include_distances=True)
                # print(f"\nQuery {i+1} {names[j]}\n")
                neg_names = []
                for k in range(4980, 4980 + len(names)):
                    provision_idx = indices[k]
                    provision_name = provision_list[provision_idx]['name']
                    neg_names.append(provision_name)
            neg_names = ",".join(neg_names)
            new_train_list.append({'id': item['id'], 'title': item['title'], 'question': item['question'], 'label': item['label'], 'neg_label': neg_names})
            # break

        with open("./data/py/train_list_with_neg.py", 'w', encoding='utf-8') as py_file:
            py_file.write("train_list = ")
            py_file.write(json.dumps(new_train_list, ensure_ascii=False, indent=4))

        #     for j in range(45, 50):
        #         provision_idx = indices[j]
        #         provision_name = provision_list[provision_idx]['name']
        #         print(f"  Provision: {provision_name}, Distance: {distances[j]:.4f}")
        #     print()
        # query_embeddings = self.generate_query_embeddings(queries) # Shape (N, D)
        
        # for i, query_embedding in enumerate(query_embeddings):
        #     print(f"\nQuery {i+1} {names[i]}\n")
        #     print(f"Top {k} nearest provisions:")

        #     # Annoy search for top k nearest neighbors (search returns indices and distances)
        #     indices, distances = index.get_nns_by_vector(query_embedding, k, include_distances=True)
        #     for j in range(k):
        #         provision_idx = indices[j]  # Index of the provision
        #         distance = distances[j]  # Distance to this provision
        #         provision_name = provision_list[provision_idx]["name"]  # Retrieve the provision name from the law list
        #         print(f"  Provision: {provision_name}, Distance: {distance:.4f}")

    def interaction(self):
 
        # tokenizer = AutoTokenizer.from_pretrained("lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct")
        # model = AutoModelForCausalLM.from_pretrained("lianghsun/Llama-3.2-Taiwan-Legal-3B-Instruct")
        model = AutoModel.from_pretrained("QuantFactory/Llama-3.2-Taiwan-Legal-3B-Instruct-GGUF")
        model.to(self.device)

        title_and_keyword = {"id": 21, "title": "醫生如何合法修改病例，怎樣是違法竄改？醫療糾紛中應該由誰負舉證責任？", "question": "看到新聞說醫療糾紛占民事訴訟中很大一塊，病例常常是訴訟中雙方爭執的事情。\n1.醫師發現病例記載有疏漏，可否事後修改？要遵守哪些規定才合法？\n2.如果病人認為醫生違法竄改病歷，是病人要證明「有竄改的事實」？還是病人只要提出，反過來由醫院證明「沒有竄改」？\n3.如果醫生違法竄改病歷，法律上要負什麼責任呢？", "label": "中華民國刑法第215條,醫師法第12條,醫療法第104條,醫療法第67條,醫療法第68條,醫療法第74條", "keyword": ": 醫師、病歷、醫療糾紛、訴訟、責任、糾紛、雙方。"}
        query = "關鍵字:" + title_and_keyword["keyword"] + "標題:" + title_and_keyword["title"] + "問題:" + title_and_keyword["question"]

        outputs = model.generate(query, max_length=512, num_return_sequences=1, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        print(outputs)

        # input_ids = tokenizer.encode(query, return_tensors="pt")
        # print(input_ids.shape)
        # outputs = model.generate(input_ids, max_length=512, num_return_sequences=1, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        # print(output.shape)

        # for i, output in enumerate(outputs):
        #     print(f"Response {i+1}: {tokenizer.decode(output, skip_special_tokens=True)}")


        # from data.py.provision_list import provision_list

        # self.load_model()
        # model = self.model
        # device = self.device

        # for item in provision_list:
        #     if item['genere'] == "醫療法":
        #         provision_list = item['provisions']
        #         break
        
        # title_and_keyword = {"id": 21, "title": "醫生如何合法修改病例，怎樣是違法竄改？醫療糾紛中應該由誰負舉證責任？", "question": "看到新聞說醫療糾紛占民事訴訟中很大一塊，病例常常是訴訟中雙方爭執的事情。\n1.醫師發現病例記載有疏漏，可否事後修改？要遵守哪些規定才合法？\n2.如果病人認為醫生違法竄改病歷，是病人要證明「有竄改的事實」？還是病人只要提出，反過來由醫院證明「沒有竄改」？\n3.如果醫生違法竄改病歷，法律上要負什麼責任呢？", "label": "中華民國刑法第215條,醫師法第12條,醫療法第104條,醫療法第67條,醫療法第68條,醫療法第74條", "keyword": ": 醫師、病歷、醫療糾紛、訴訟、責任、糾紛、雙方。"}

        # scores = []
        # query = "關鍵字:" + title_and_keyword["keyword"] + "標題:" + title_and_keyword["title"] + "問題:" + title_and_keyword["question"]
        # for provision in provision_list:
        #     name = provision['name']
        #     content = provision['content']
        #     example = provision['example']

        #     provision_content = content + "".join(example)

        #     query = query.replace("\n", "")
        #     query = query.replace("\r", "")
        #     query = query.replace(" ", "")
        #     provision_content = provision_content.replace("\n", "")
        #     provision_content = provision_content.replace("\r", "")
        #     provision_content = provision_content.replace(" ", "")

        #     input_ids = self.tokenize_jieba(query, provision_content)
        #     # print(f"Provision: {name}, Input IDs: {input_ids}")
            
        #     input_ids = input_ids.unsqueeze(0).to(device)
        #     attention_mask = (input_ids > 0).to(device)
        #     predicted_scores = model(input_ids, attention_mask)
        #     predicted_scores = predicted_scores.squeeze(1)

        #     scores.append([name, predicted_scores.item()])
        
        # scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # print(f"Top 50 nearest provisions:")
        # for i in range(50):
        #     print(f"  Provision: {scores[i][0]}, Score: {scores[i][1]:.4f}")
        # print(f"Ground Truth: {title_and_keyword['label']}")

    def tokenize_jieba(self, s1, s2):
        s1_tokens = jieba.lcut(s1)
        s1_token_ids = self.tokenizer.convert_tokens_to_ids(s1_tokens)
        s2_tokens = jieba.lcut(s2)
        s2_token_ids = self.tokenizer.convert_tokens_to_ids(s2_tokens)

        token_ids = [self.tokenizer.cls_token_id] + s1_token_ids + [self.tokenizer.sep_token_id] + s2_token_ids + [self.tokenizer.sep_token_id]

        # Pad to max_length if necessary
        padding_length = self.config["max_length"] - len(token_ids)
        if padding_length > 0:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length
        else:
            token_ids = token_ids[:self.config["max_length"]]

        return torch.tensor(token_ids)
    
    def plot_embeddings(self, query_embedding, embedding_space, save_path="embedding_plot.png"):
        label_embeddings = []
        other_embeddings = []
        labels = []

        # print(type(embedding_space))
        # print(embedding_space[0])
        # print(embedding_space.shape)


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
            plt.scatter(reduced_embeddings[i+1][0], reduced_embeddings[i+1][1], color='blue', label='Label Match' if i == 0 else "", s=100, marker='x')  # Label matches
        
        # Add labels and title
        plt.title('Query vs Label vs Other Provisions Embeddings')
        plt.legend()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Save the plot as a .png file
        plt.savefig(save_path)
        plt.close()  # Close the figure after saving it to avoid display