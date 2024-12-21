import os
import json
import jieba
from rank_bm25 import BM25Okapi
from pathlib import Path


# 1. 加載文本資料
def load_documents(document_folder):
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

def build_bm25(documents):
    bm25 = BM25Okapi(documents)
    return bm25

def retrieve_documents(keyword, bm25, file_map, top_k=10):
    if "、" in keyword:  # 判斷是否包含「、」
        query = keyword.split("、")  # 使用「、」分割關鍵詞
    else:
        query = list(jieba.cut(keyword))
    scores = bm25.get_scores(query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = [{"file": file_map[idx], "score": scores[idx]} for idx in top_indices]
    return results

if __name__ == "__main__":
    document_folder = "./data/laws_by_category"  # 文檔資料夾
    query_file = "./data/json/test_with_keyword.json"  # 查詢檔案
    output_file = "result.json"
    top_k = 10  # 檢索返回的文件數量

    documents, file_map = load_documents(document_folder)
    bm25 = build_bm25(documents)

    results = []
    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                query_json = json.loads(line.strip())
                keyword = query_json["keyword"]
                retrieved_docs = retrieve_documents(keyword, bm25, file_map, top_k)
                results.append({"query_id": query_json["id"], "retrieved": retrieved_docs})

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
