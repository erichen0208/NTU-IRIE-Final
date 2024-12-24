import argparse
import json
from utils.Retriever import Retriever
import os
from utils.preprocess.data_preprocess import generate_keyword_train_list
import numpy as np
from sklearn.model_selection import train_test_split

def main():

    # generate_keyword_train_list()
    # return
    parser = argparse.ArgumentParser(description="Legal Document Retrieval System")
    parser.add_argument('--mode', type=str, default='llm', choices=['llm', 'bert', 'finetune', 'generate', 'inference', 'test', 'test_embeddings', 'interaction'], help='Mode to run: train, inference, test')

    args = parser.parse_args()

    # Continue learning for llm or bert
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
    
    # Finetine / Generate / Inference / Test
    if args.mode == 'finetune':
        print("Starting finetine bert model mode...")

        # Load configuration
        with open('finetune_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        # from data.py.train_list import train_list
        from data.py.train_list_with_keyword_and_rel import train_list
        with open(config['provision_dict_path'], 'r', encoding='utf-8') as f:
            provision_dict = json.load(f)

        retriever = Retriever(config)
        retriever.finetune(train_list, provision_dict)

    elif args.mode == 'generate':
        print("Starting provision embeddings generation mode...")

        # Load configuration
        with open('finetune_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        from data.py.provision_list import provision_list

        retriever = Retriever(config)
        retriever.generate_provision_embeddings(provision_list)

    elif args.mode == 'inference':
        print("Starting inference mode...")

        # Load configuration
        with open('inference_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        from data.py.provision_list import provision_list
        
        test_list = []
        with open(config['test_data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                test_list.append(item)

        retriever = Retriever(config)
        retriever.inference(test_list)

    elif args.mode == 'test':
        print("Starting test mode...")

        # Load configuration
        with open('inference_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        train_list = []
        with open(config["train_data_path"], 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                train_list.append(item)

        _, val_list = train_test_split(train_list, test_size=0.2, random_state=42)

        retriever = Retriever(config)
        retriever.test(val_list)
    
    elif args.mode == 'test_embeddings':
        print("Starting test embeddings mode...")

        # Load configuration
        with open('inference_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        retriever = Retriever(config)
        retriever.test_embeddings()
        
    else:  
        retriever.interaction()

if __name__ == '__main__':
    main()