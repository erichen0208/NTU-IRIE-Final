import torch
import torch.nn as nn

class DualBERTEncoder(nn.Module):
    def __init__(self, model):
        super(DualBERTEncoder, self).__init__()
        self.bert = model
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
    
    def forward(self, query_input_ids, provision_input_ids_list):
        """
        Forward method for the dual encoder.

        Args:
        - query_input_ids (torch.Tensor): Tensor of query input IDs (batch_size, seq_len)
        - provision_input_ids_list (torch.Tensor): Tensor of provision input IDs 
                                                   with shape (batch_size, num_provisions, seq_len)
        
        Returns:
        - query_embedding (torch.Tensor): Embedding for query with shape (batch_size, hidden_size)
        - provision_embeddings (torch.Tensor): Embeddings for all provisions with shape 
                                               (batch_size, num_provisions, hidden_size)
        """
        
        # Pass the query and provision through BERT
        query_outputs = self.bert(query_input_ids, attention_mask=(query_input_ids > 0))
        query_embedding = query_outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        
        batch_size, num_provisions, seq_len = provision_input_ids_list.shape
        flat_provision_input_ids = provision_input_ids_list.view(batch_size * num_provisions, seq_len)
        
        # Attention mask for provision inputs
        attention_mask = (flat_provision_input_ids > 0).long()
        
        # Forward pass through BERT
        provision_outputs = self.bert(flat_provision_input_ids, attention_mask=attention_mask)
        
        # Extract the [CLS] embeddings for all provision inputs
        provision_embeddings = provision_outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size * num_provisions, hidden_size)
        
        # Reshape it back to (batch_size, num_provisions, hidden_size)
        hidden_size = provision_embeddings.shape[1]
        provision_embeddings = provision_embeddings.view(batch_size, num_provisions, hidden_size)
        
        return query_embedding, provision_embeddings
    
    def encode(self, input_ids):
        outputs = self.bert(input_ids, attention_mask=(input_ids > 0))
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding