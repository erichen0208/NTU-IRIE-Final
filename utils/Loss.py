import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, query_embedding, provision_embedding, rel):
        distances = torch.norm(query_embedding - provision_embedding, dim=1)  # Euclidean distance
        loss = 0.5 * rel * distances**2 + 0.5 * (1 - rel) * torch.clamp(self.margin - distances, min=0)**2
        return loss.mean()
    
class ContrastiveLoss_old(nn.Module):
    def __init__(self, margin=0.5, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, query_embedding, positive_embeddings, negative_embeddings):
        """
        Contrastive loss with multiple positives and negatives per query.
        
        Args:
            query_embedding: Tensor of shape [batch_size, hidden_dim]
            positive_embeddings: Tensor of shape [batch_size, num_positives, hidden_dim]
            negative_embeddings: Tensor of shape [batch_size, num_negatives, hidden_dim]
            
        Returns:
            loss: Scalar tensor
            similarities: Dict containing positive and negative similarities for monitoring
        """

        # Normalize embeddings (if not already normalized)
        # query_embedding = F.normalize(query_embedding, p=2, dim=1)
        # positive_embeddings = F.normalize(positive_embeddings, p=2, dim=2)
        # negative_embeddings = F.normalize(negative_embeddings, p=2, dim=2)
        
        # Compute similarities
        positive_similarities = torch.bmm(
            positive_embeddings, 
            query_embedding.unsqueeze(2)
        ).squeeze(2)  # [batch_size, num_positives]
        
        negative_similarities = torch.bmm(
            negative_embeddings, 
            query_embedding.unsqueeze(2)
        ).squeeze(2)  # [batch_size, num_negatives]

        # Compute loss for each positive-negative pair
        positive_similarities = positive_similarities.unsqueeze(2)  # [batch_size, num_positives, 1]
        negative_similarities = negative_similarities.unsqueeze(1)  # [batch_size, 1, num_negatives]
        
        # Broadcasting to compare each positive with each negative
        loss_matrix = torch.relu(
            negative_similarities - positive_similarities + self.margin
        )  # [batch_size, num_positives, num_negatives]

        # Reduce loss based on specified reduction method
        if self.reduction == 'mean':
            loss = loss_matrix.mean()
        elif self.reduction == 'sum':
            loss = loss_matrix.sum()
        elif self.reduction == 'none':
            loss = loss_matrix
        
        # Return loss and similarities for monitoring
        similarities = {
            'positive_similarities': positive_similarities.mean().item(),
            'negative_similarities': negative_similarities.mean().item(),
            'loss': loss.item()
        }
        
        return loss, similarities

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query_embedding, positive_embeddings, negative_embeddings):
        """
        InfoNCE (NT-Xent) loss implementation.
        
        Args:
            query_embedding: Tensor of shape [batch_size, hidden_dim]
            positive_embeddings: Tensor of shape [batch_size, num_positives, hidden_dim]
            negative_embeddings: Tensor of shape [batch_size, num_negatives, hidden_dim]
            
        Returns:
            loss: Scalar tensor
            similarities: Dict containing metrics for monitoring
        """
        # Normalize embeddings
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=2)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=2)

        # Compute positive similarities
        positive_similarities = torch.bmm(
            positive_embeddings,
            query_embedding.unsqueeze(2)
        ).squeeze(2) / self.temperature  # [batch_size, num_positives]

        # Compute negative similarities
        negative_similarities = torch.bmm(
            negative_embeddings,
            query_embedding.unsqueeze(2)
        ).squeeze(2) / self.temperature  # [batch_size, num_negatives]

        # Combine positive and negative similarities
        logits = torch.cat([positive_similarities, negative_similarities], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Calculate cross entropy loss
        loss = F.cross_entropy(logits, labels)

        similarities = {
            'positive_similarities': positive_similarities.mean().item(),
            'negative_similarities': negative_similarities.mean().item(),
            'loss': loss.item()
        }
        
        return loss, similarities

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, predictions, targets):
        return self.loss_fn(predictions, targets)
    