import argparse
import json
import os
from utils.Retriever import Retriever

def main():
    parser = argparse.ArgumentParser(description="Legal Document Retrieval System")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'test'], help='Mode to run: train, eval, test')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    retriever = Retriever(config)

    if args.mode == 'train':
        print("Starting training mode...")

        # Load law documents
        provisions = {}
        with open(config['provision_path'], 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                provisions[item['provision']] = item['content']

        with open(config['train_data_path'], 'r', encoding='utf-8') as f:
            training_data = [json.loads(line) for line in f]

        retriever.train(provisions, training_data)
        # retriever.save_model(config['model_save_path'])
        # retriever.save_embeddings(config['doc_embeddings_save_path'])

    elif args.mode == 'eval':
        print("Starting evaluation mode...")
        retriever.load_model(config['model_save_path'])
        retriever.load_embeddings(config['embeddings_save_path'])
        retriever.evaluate()

    elif args.mode == 'test':
        print("Starting test mode...")
        retriever.load_model(config['model_save_path'])
        retriever.load_embeddings(config['embeddings_save_path'])

        queries = config['test_queries']

        for idx, query in enumerate(queries):
            relevant_laws = retriever.retrieve_relevant_laws(query)
            print(f"Query {idx+1}: {query}")
            for law_id, similarity in relevant_laws:
                print(f"  Law: {law_id}, Similarity: {similarity:.4f}")

if __name__ == '__main__':
    main()
