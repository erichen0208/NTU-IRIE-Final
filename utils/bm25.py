import json
import jieba
from rank_bm25 import BM25Okapi

class BM25:
    def __init__(self, top_k = 10, document_file = "./data/json/law_with_example.jsonl"):
        self.top_k = top_k
        self.documents, self.file_map = self.load_laws(document_file)
        self.bm25 = self.build_bm25(self.documents)

    def load_laws(self, document_file): # 20000+ laws
        documents = []
        file_map = {}
        with open(document_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if line.strip():
                    doc_json = json.loads(line.strip())
                    content = doc_json.get("content", "")
                    example = doc_json.get("example", "")
                    combined_text = f"{content}{example}"
                    words = list(jieba.cut(combined_text))  
                    documents.append(words)  
                    file_map[idx] = doc_json["provision"]  

        return documents, file_map

    def build_bm25(self, documents):
        bm25 = BM25Okapi(documents)
        return bm25

    def retrieve_provisions(self, query): # Return top-k provisions: [{"provision": str, "score": float}]
        query = list(jieba.cut(query))

        # Get Scores
        scores = self.bm25.get_scores(query)
        min_score = min(scores)
        max_score = max(scores)
        if max_score != min_score:  # Avoid division by zero
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        else:
            normalized_scores = [0.5] * len(scores)
        scores = normalized_scores

        # Get Top-K
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        results = [{"name": self.file_map[idx], "score": scores[idx]} for idx in top_indices]
        return results
