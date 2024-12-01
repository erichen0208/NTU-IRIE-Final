import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import jieba
import re
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple

class ChineseLegalRetriever:
    def __init__(self):
        # Initialize BERT model specifically trained for Chinese
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.model = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        # Initialize jieba with legal dictionary
        self.initialize_jieba()
        
    def initialize_jieba(self):
        """Initialize jieba with legal terms"""
        # Add legal-specific terms to jieba
        legal_terms = ['勞基法', '特休假', '工作時間', '加班費', '勞動基準法']
        for term in legal_terms:
            jieba.add_word(term)
    
    def preprocess_chinese_text(self, text: str) -> str:
        """
        Preprocess Chinese text with specific handling for legal documents
        """
        # Remove unnecessary spaces and normalize punctuation
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('：', ':').replace('；', ';')
        
        # Remove parentheses numbers like (一) (二) often found in legal texts
        text = re.sub(r'（[一二三四五六七八九十]+）', '', text)
        
        # Segment using jieba
        words = jieba.cut(text)
        return ' '.join(words)
    
    def parse_law_document(self, content: str) -> List[Dict]:
        """
        Parse a law document into structured format
        """
        # Split into articles
        articles = []
        current_article = ''
        current_number = ''
        
        for line in content.split('\n'):
            if '第' in line and '條' in line:
                if current_article:
                    articles.append({
                        'article_number': current_number,
                        'content': current_article.strip()
                    })
                current_number = re.search(r'第\s*(\d+)\s*條', line).group(1)
                current_article = line
            else:
                current_article += ' ' + line.strip()
                
        # Add the last article
        if current_article:
            articles.append({
                'article_number': current_number,
                'content': current_article.strip()
            })
            
        return articles

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding for Chinese text
        """
        # Tokenize and get BERT embedding
        inputs = self.tokenizer(text, return_tensors='pt', 
                              max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use [CLS] token embedding as text representation
        return outputs.last_hidden_state[:, 0, :].numpy()
    
    def create_article_embeddings(self, laws: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Create embeddings for all articles in the laws
        """
        embeddings = {}
        for law in laws:
            articles = self.parse_law_document(law['content'])
            for article in articles:
                key = f"{law['title']}_第{article['article_number']}條"
                processed_text = self.preprocess_chinese_text(article['content'])
                embeddings[key] = self.get_bert_embedding(processed_text)
        return embeddings

    def retrieve_relevant_laws(self, 
                             query: str, 
                             laws: List[Dict], 
                             embeddings: Dict[str, np.ndarray],
                             top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant laws for a query
        """
        # Preprocess query
        processed_query = self.preprocess_chinese_text(query)
        query_embedding = self.get_bert_embedding(processed_query)
        
        # Calculate similarities
        similarities = []
        for law_id, law_embedding in embeddings.items():
            similarity = cosine_similarity(query_embedding, law_embedding)[0][0]
            similarities.append((law_id, similarity))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

def main():
    # Example usage
    retriever = ChineseLegalRetriever()
    
    # Load laws and queries
    with open('laws.json', 'r', encoding='utf-8') as f:
        laws = json.load(f)
    
    with open('queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    # Create embeddings for all articles (can be cached)
    embeddings = retriever.create_article_embeddings(laws)
    
    # Example query
    query = queries[0]['question']
    relevant_laws = retriever.retrieve_relevant_laws(query, laws, embeddings)
    
    return relevant_laws