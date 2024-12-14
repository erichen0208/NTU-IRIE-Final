import torch
from torch import nn

class Loss(nn.Module):
    def __init__(self, margin=0.5):
        super(Loss, self).__init__()
        self.margin = margin
    
    def forward(self, query_embedding, positive_embeddings, negative_embeddings):
        positive_similarities = torch.stack([
            torch.nn.functional.cosine_similarity(query_embedding, pos_emb)
            for pos_emb in positive_embeddings
        ], dim=1)  # Shape: [batch_size, num_positives]
        
        # Calculate cosine similarity for each negative embedding
        negative_similarities = torch.stack([
            torch.nn.functional.cosine_similarity(query_embedding, neg_emb)
            for neg_emb in negative_embeddings
        ], dim=1)  # Shape: [batch_size, num_negatives]
        
        # Calculate the mean similarity for positive and negative embeddings
        positive_distance = positive_similarities.mean(dim=1)  # Shape: [batch_size]
        negative_distance = negative_similarities.mean(dim=1)  # Shape: [batch_size]
        
        # Contrastive loss calculation (using ReLU to apply the margin)
        loss = torch.mean(torch.relu(self.margin + negative_distance - positive_distance))  # Shape: scalar loss
        return loss
