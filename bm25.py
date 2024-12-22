import os
import json
import jieba
import argparse
from rank_bm25 import BM25Okapi

# 加載文本資料
def load_categories(document_folder): #299個分類
    documents = []
    file_map = {}
    for file_name in os.listdir(document_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(document_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                words = list(jieba.cut(content))  # 使用 jieba 進行中文分詞
                documents.append(words)  # 添加分詞後的結果
                file_map[len(documents) - 1] = file_name  # 建立文件索引對應
    return documents, file_map

def load_laws(document_file): #20000條法律
    documents = []
    file_map = {}
    with open(document_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                doc_json = json.loads(line.strip())
                content = doc_json.get("content", "")
                example = doc_json.get("example", "")
                combined_text = f"{content}{example}"
                words = list(jieba.cut(combined_text))  # 使用 jieba 進行中文分詞
                documents.append(words)  # 添加分詞後的結果
                file_map[idx] = doc_json["provision"]  # 將索引映射到 "provision"
    return documents, file_map

# 建構 BM25 系統
def build_bm25(documents):
    bm25 = BM25Okapi(documents)
    return bm25

# 檢索文件
def retrieve_documents(title, keyword, bm25, file_map, top_k=5):
    cleaned_keyword = keyword.replace(": ", "").replace("。", "")
    if "、" in cleaned_keyword:  # 判斷是否包含「、」
        query = cleaned_keyword.replace("、", "").replace(",", "")
        query = list(jieba.cut(query))
        
    else:
        cleaned_title = title.replace("？", "").replace("，", "")
        query = list(jieba.cut(cleaned_title))
    print(query)
    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = [{"file": file_map[idx], "score": scores[idx]} for idx in top_indices]
    return results

# 主程式邏輯
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Legal Document Retrieval System")
    parser.add_argument('--mode', type=str, default='inference', choices=['categorize', 'inference'])
    args = parser.parse_args()
    query_file = "./data/json/test_with_keyword.json" 
    top_k = 5

    if args.mode == "categorize":
        document_folder = "./data/laws_by_category"
        output_file = "results.jsonl"
        documents, file_map = load_categories(document_folder)

    if args.mode == "inference":
        document_file = "./data/json/law_with_examples.jsonl"
        output_file = "results.jsonl"  # 檢索結果輸出檔案
        documents, file_map = load_laws(document_file)
    
    bm25 = build_bm25(documents)

    # 處理每一行查詢並進行檢索
    results = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 確保不是空行
                try:
                    query_json = json.loads(line.strip())  # 確保每行轉換為字典
                    keyword = query_json["keyword"]  # 提取關鍵字
                    title = query_json["title"]
                    retrieved_docs = retrieve_documents(title, keyword, bm25, file_map, top_k)
                    results.append({"query_id": query_json["id"], "retrieved": retrieved_docs})
                except json.JSONDecodeError:
                    print(f"跳過無效的 JSON 行: {line.strip()}")

    # 將檢索結果寫入輸出檔案
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
