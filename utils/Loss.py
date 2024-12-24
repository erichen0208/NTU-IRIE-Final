import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, query_embedding, provision_embedding, rel):
        distances = torch.norm(query_embedding - provision_embedding, dim=1)  # Euclidean distance
        loss = 0.5 * rel * distances**2 + 0.5 * (1 - rel) * torch.clamp(self.margin - distances, min=0)**2
        return loss.mean()

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, predictions, targets):
        return self.loss_fn(predictions, targets)
    