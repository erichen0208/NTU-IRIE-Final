import argparse
import json
from utils.Retriever import Retriever

def main():
    parser = argparse.ArgumentParser(description="Legal Document Retrieval System")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate', 'inference', 'test'], help='Mode to run: train, inference, test')
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
                provisions[item['provision']] = item['content'] # {"法律名": "法律内容"}

        with open(config['train_data_path'], 'r', encoding='utf-8') as f:
            training_data = [json.loads(line) for line in f]

        retriever.train(provisions, training_data)

    elif args.mode == 'generate':
        print("Starting provision embeddings generation mode...")

        retriever.generate_provision_embeddings()

    elif args.mode == 'inference':
        print("Starting inference mode...")
        
        # Load queries
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
        
        queries = [
            "帶手指虎打傷人，手指虎會被「沒收」還是「沒入」？ 知道沒收是刑法，沒入是行政法。\n\r\n單純持有違禁品（手指虎）會遭到沒收，\r\n但用違禁品傷人，是會被「沒收」還是「沒入」呢？",
            "請問，如果我們在網路上看到有人販賣違禁品，我們應該通報哪個單位？",
            "終止收養後繼承問題\n出生後三個月被收養，國中時因對方搬到同一區域，經收養家庭允許見過原生家庭並了解對方情況，經濟能力差且可能有負債，但再無聯絡交集，成年後因個人因素終止收養回歸本姓。\r\n過了二十幾年現在想到這件事，對方若往生我也不知情，怎麼來得及辦理拋棄繼承？\r\n到目前的人生為止就見過那次面再無其他連結，這能成為保障自己的主張嗎，萬分感謝。"
        ]
        retriever.test(queries)

if __name__ == '__main__':
    main()
