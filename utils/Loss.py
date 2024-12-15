import torch
from torch import nn

class Loss(nn.Module):
    def __init__(self, margin=0.5):
        super(Loss, self).__init__()
        self.margin = margin
    
    def forward(self, query_embedding, positive_embeddings, negative_embeddings):
        """
        query_embedding: [batch_size, hidden_dim] 
        positive_embeddings: [batch_size, num_positives, hidden_dim] 
        negative_embeddings: [batch_size, num_negatives, hidden_dim]
        """
        
        # Step 1: Normalize all embeddings
        query_embedding = nn.functional.normalize(query_embedding, p=2, dim=1)  # Shape: [batch_size, hidden_dim]
        positive_embeddings = nn.functional.normalize(positive_embeddings, p=2, dim=2)  # Shape: [batch_size, num_positives, hidden_dim]
        negative_embeddings = nn.functional.normalize(negative_embeddings, p=2, dim=2)  # Shape: [batch_size, num_negatives, hidden_dim]
        
        # Step 2: Compute cosine similarity
        positive_similarities = torch.bmm(positive_embeddings, query_embedding.unsqueeze(2)).squeeze(2)  # [batch_size, num_positives]
        negative_similarities = torch.bmm(negative_embeddings, query_embedding.unsqueeze(2)).squeeze(2)  # [batch_size, num_negatives]
        
        # Step 3: Calculate the mean similarity for positive and negative embeddings
        positive_distance = positive_similarities.mean(dim=1)  # Shape: [batch_size]
        negative_distance = negative_similarities.mean(dim=1)  # Shape: [batch_size]
        
        # Step 4: Contrastive loss (ReLU to apply margin)
        loss = torch.mean(torch.relu(negative_distance - positive_distance + self.margin))  # Scalar loss
        return loss

