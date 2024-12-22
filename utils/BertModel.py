import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
)

class BertModel(nn.Module):
    def __init__(self, config, device, mode='inference'):
        super().__init__()
        self.device = device
        
        bert_config = AutoConfig.from_pretrained(config['model_path'])
        self.model = AutoModel.from_pretrained(config['model_path'], config=bert_config)
        self.model.to(device)
        self.pool = self._init_pool().to(device)

        # if mode != 'finetune':
        #     self.load_model(config['pth_save_path'])

    def _init_pool(self):
        hidden_size = self.model.config.hidden_size
        return nn.Sequential(
            nn.Linear(hidden_size, 512),   
            nn.GELU(),                        
            nn.Dropout(0.15),                    
            nn.Linear(512, 256),                
            nn.GELU(),                        
            nn.Dropout(0.15),
            nn.Linear(256, 1),                 
            nn.Sigmoid() 
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): Input token ids of shape [batch_size, seq_length]
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Normalized embeddings of shape [batch_size, 1]
        """

        # Ensure inputs are tensors
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
            
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Apply max pooling over the sequence dimension
        # Mask out padding tokens before pooling
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(outputs)  # [batch_size, seq_length, hidden_size]
        masked_outputs = outputs.masked_fill(expanded_mask == 0, float('-inf'))  # Replace padded positions with -inf
        pooled_output, _ = torch.max(masked_outputs, dim=1)  # Max pooling over seq_length, shape: [batch_size, hidden_size]
        
        # Pass through the fully connected layers
        score = self.pool(pooled_output)

        # score = self.pool(outputs)

        return score
    
    def save_model(self, model_save_path, epoch, optimizer, loss):
        """
        Save everything in one file
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': {
                'model': self.model.state_dict(),  # Save only BERT weights
                'pool': self.pool.state_dict(),  # Save pooling weights separately
            },
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, model_save_path)

    def load_model(self, pth_save_path):
        """
        Load a binary model checkpoint (.pth file) and restore model weights.
        """
        device = self.device
        checkpoint = torch.load(pth_save_path, map_location=device) 
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")

        if 'model_state_dict' in checkpoint:
            # Load BERT weights
            bert_weights = checkpoint['model_state_dict']['model']
            missing_keys, unexpected_keys = self.model.load_state_dict(bert_weights, strict=False)
            print(f"BERT missing keys: {missing_keys}")
            print(f"BERT unexpected keys: {unexpected_keys}")
            
            # Load pool weights
            pool_weights = checkpoint['model_state_dict']['pool']
            missing_keys, unexpected_keys = self.pool.load_state_dict(pool_weights, strict=False)
            print(f"pool missing keys: {missing_keys}")
            print(f"pool unexpected keys: {unexpected_keys}")
        else:
            # Load entire model
            missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")

class DenseModel(nn.Module):
    def __init__(self, config, device, mode='inference'):
        super().__init__()
        self.device = device
        
        bert_config = AutoConfig.from_pretrained(config['model_path'])
        self.model = AutoModel.from_pretrained(config['model_path'], config=bert_config)
        self.model.to(device)
        self.pool = self._init_pool().to(device)

        # if mode != 'finetune':
        self.load_model(config['pth_save_path'])

    def _init_pool(self):
        hidden_size = self.model.config.hidden_size
        return nn.Sequential(
            nn.Linear(hidden_size, 512),   
            nn.GELU(),                        
            nn.Dropout(0.10),                    
            nn.Linear(512, 256),                
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): Input token ids of shape [batch_size, seq_length]
            attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Normalized embeddings of shape [batch_size, 1]
        """

        # Ensure inputs are tensors
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
                
        # Get hidden states from BERT
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]

        # Max pooling across sequence length
        # Masking padded positions with a very small value
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()  # Shape: [batch_size, seq_length, hidden_size]
        hidden_states[mask == 0] = -1e9  # Replace masked positions with a large negative value for max pooling
        max_pooled = torch.max(hidden_states, dim=1)[0]  # Shape: [batch_size, hidden_size]

        # Pass through fully connected layers (if any)
        embedding = self.pool(max_pooled)  # Shape: [batch_size, embedding_dim]

        # Normalize embeddings
        normalized_embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalization

        return normalized_embedding
    
    def save_model(self, model_save_path, epoch, optimizer, loss):
        """
        Save everything in one file
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': {
                'model': self.model.state_dict(),  # Save only BERT weights
                'pool': self.pool.state_dict(),  # Save pooling weights separately
            },
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, model_save_path)

    def load_model(self, pth_save_path):
        """
        Load a binary model checkpoint (.pth file) and restore model weights.
        """
        device = self.device
        checkpoint = torch.load(pth_save_path, map_location=device) 
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")

        if 'model_state_dict' in checkpoint:
            # Load BERT weights
            bert_weights = checkpoint['model_state_dict']['model']
            missing_keys, unexpected_keys = self.model.load_state_dict(bert_weights, strict=False)
            print(f"BERT missing keys: {missing_keys}")
            print(f"BERT unexpected keys: {unexpected_keys}")
            
            # Load pool weights
            pool_weights = checkpoint['model_state_dict']['pool']
            missing_keys, unexpected_keys = self.pool.load_state_dict(pool_weights, strict=False)
            print(f"pool missing keys: {missing_keys}")
            print(f"pool unexpected keys: {unexpected_keys}")
        else:
            # Load entire model
            missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")