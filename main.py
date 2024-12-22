import argparse
import json
from utils.Retriever import Retriever
import os
from utils.preprocess.data_preprocess import generate_keyword_train_list
import numpy as np

def main():

    # generate_keyword_train_list()
    # return
    parser = argparse.ArgumentParser(description="Legal Document Retrieval System")
    parser.add_argument('--mode', type=str, default='llm', choices=['llm', 'bert', 'finetune', 'generate', 'inference', 'test', 'test_embeddings', 'interaction'], help='Mode to run: train, inference, test')

    args = parser.parse_args()

    if args.mode == 'llm':
        print("Starting continue learning for llm mode...")

        from utils.Pretrain import pretrain
        with open('pretrain_llm_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        pretrain(mode='llm', config=config)
        return
    
    elif args.mode == 'bert':
        print("Starting continue learning for bert mode...")

        from utils.Pretrain import pretrain
        with open('pretrain_bert_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        pretrain(mode='bert', config=config)
        return

    # Load configuration
    with open('finetune_bert_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)

    retriever = Retriever(config)
    
    if args.mode == 'finetune':
        print("Starting finetine bert model mode...")

        # from data.py.train_list import train_list
        from data.py.train_list_new import train_list
        with open("./data/json/provision_dict.json", 'r', encoding='utf-8') as f:
            provision_dict = json.load(f)

        retriever.finetune(train_list, provision_dict)

    elif args.mode == 'generate':
        print("Starting provision embeddings generation mode...")

        # from data.py.provision_list import provision_list
        from data.py.provision_list import provision_list
        retriever.generate_provision_embeddings(provision_list)

    elif args.mode == 'inference':
        print("Starting inference mode...")
        
        # Load queries (list of text)
        queries = []
        with open(config['test_data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                title = item["title"] if item["title"] != None else ""
                question = item["question"] if item["question"] != None else ""
                query = title + '\n' + question
                queries.append(query)    
        from data.py.provision_list import provision_list

        retriever.inference(provision_list, queries)

    elif args.mode == 'test':
        print("Starting test mode...")
        from data.py.val_list import val_list
        from data.py.provision_list import provision_list
        
        queries = [] # list of text
        labels = [] # list of [list of labels]
        for val_data in val_list[:10]:
            title = val_data["title"] if val_data["title"] != None else ""
            question = val_data["question"] if val_data["question"] != None else ""
            query = title + '\n' + question
            queries.append(query)
            labels.append(val_data["label"].split(','))

        for i in range(len(provision_list)):
            if provision_list[i]["genere"] == "中華民國刑法":
                provision_list = provision_list[i]["provisions"]
                break

        queries = ["勞工、雇主、特休假、勞動契約、預告期、抵充、交接",
                   "妨礙自由、強硬填入、貸款、資料",
                   "醫師、病歷、醫療糾紛、訴訟、責任、糾紛、雙方"
                ]
        labels = [["勞動基準法第15條", "勞動基準法第38條"],
                  ["中華民國刑法第302條"],
                  ["中華民國刑法第215條","醫師法第12條"",醫療法第104條","醫療法第67條","醫療法第68條","醫療法第74條"]]
        retriever.test(provision_list, queries, labels)
    
    elif args.mode == 'test_embeddings':
        print("Starting test embeddings mode...")

        retriever.test_embeddings()
        return

    else:  
        retriever.interaction()

if __name__ == '__main__':
    main()