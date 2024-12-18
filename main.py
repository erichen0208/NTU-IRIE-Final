import argparse
import json
from utils.Retriever import Retriever
import os

def main():
    parser = argparse.ArgumentParser(description="Legal Document Retrieval System")
    parser.add_argument('--mode', type=str, default='train', choices=['pretrain', 'train', 'generate', 'inference', 'test'], help='Mode to run: train, inference, test')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')

    args = parser.parse_args()

    custom_cache_path = os.path.expanduser('~/.cache/jieba/')
    os.makedirs(custom_cache_path, exist_ok=True)
    os.environ['JIEBA_CACHE_DIR'] = custom_cache_path

    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    if args.mode == 'pretrain':
        print("Starting pretrain mode...")

        from utils.Pretrain import pretrain      
        pretrain(config)
        return

    retriever = Retriever(config)

    if args.mode == 'train':
        print("Starting train mode...")

        from data.py.train_list import train_list
        with open("./data/json/provision_dict.json", 'r', encoding='utf-8') as f:
            provision_dict = json.load(f)

        retriever.train(train_list, provision_dict)

    elif args.mode == 'generate':
        print("Starting provision embeddings generation mode...")

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

        retriever.inference(queries)

    elif args.mode == 'test':
        print("Starting test mode...")
        from data.py.val_list import val_list
        
        queries = [] # list of text
        labels = [] # list of [list of labels]
        for val_data in val_list[:3]:
            title = val_data["title"] if val_data["title"] != None else ""
            question = val_data["question"] if val_data["question"] != None else ""
            query = title + '\n' + question
            queries.append(query)
            labels.append(val_data["label"].split(','))

        retriever.test(queries, labels)

if __name__ == '__main__':
    main()
