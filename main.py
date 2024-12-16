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
        print("Starting train mode...")

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
            "帶手指虎打傷人，手指虎會被「沒收」還是「沒入」？\n知道沒收是刑法，沒入是行政法。\n\r\n單純持有違禁品（手指虎）會遭到沒收，\r\n但用違禁品傷人，是會被「沒收」還是「沒入」呢？",
            "雇用工讀生\n公司辦展覽欲雇用工讀生看顧器材，分為兩個時段各四個小時\n\r\nQ1, 若工讀生自願連續工作八個小時不休息，這樣是否違反勞基法？ 那若雙方協議可以嗎？ 如果不行，休息時間是否需要照算薪水？（例如來上班八個小時，安排休息一個小時，薪水是算八小時還是七小時）\n\r\nQ2, 工讀生若在國定假日上班，薪水也是否給雙倍？\n\r\nQ3, 如果要跟工讀生訂定工作契約來確保工讀生須對設備保管責任，哪類相關契約可以參考？",
            "百貨公司購物，家電尚未出貨\n在某百貨公司購買家電，櫃姐強調刷卡購買會給所有的折扣，會比母親節檔期更優惠，所以當下基於信任就刷卡了，\r\n但實際上比其他百貨賣的還貴，而該樓管說妳到櫃上看過摸過也體驗過，就算商品尚未出貨，也無法退刷，請問該如何處理？"
            ]
        retriever.test(queries)

if __name__ == '__main__':
    main()
