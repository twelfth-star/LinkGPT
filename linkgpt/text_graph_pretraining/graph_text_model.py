import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import BertTokenizer, BertModel, BertConfig

from .Graphformer import GraphFormersForLinkPredict

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_name: str,
        text_encoder_is_pretrained: bool,
        text_encoder_num_layers_to_train: int
    ):
        super().__init__()
        if text_encoder_is_pretrained:
            self.model = BertModel.from_pretrained(text_encoder_name)
        else:
            self.model = BertModel(config=BertConfig())
        for param in self.model.parameters():
            param.requires_grad = False
        if text_encoder_num_layers_to_train > 0:
            for param in self.model.encoder.layer[-text_encoder_num_layers_to_train:].parameters():
                param.requires_grad = True
        self.cls_token_idx = 0
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.cls_token_idx, :]

class GraphEncoder(nn.Module):
    def __init__(
        self,
        graph_encoder_name: str,
        text_encoder_name: str,
    ):
        super().__init__()
        self.graph_encoder_name = graph_encoder_name
        self.text_encoder_name = text_encoder_name
        
        if self.graph_encoder_name == 'graphformer':
            config = BertConfig.from_pretrained(self.text_encoder_name)
            self.model = GraphFormersForLinkPredict(config)
        else:
            raise ValueError(f"Invalid graph encoder name: {self.graph_encoder_name}")
        
        self.cls_token_idx = 0
        
    def forward(self, center_input, neighbor_input, neighbor_mask):
        output = self.model(center_input, neighbor_input, neighbor_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.cls_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int,
        dropout: float
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CGTPModel(nn.Module):
    '''
    Contrastive Graph Text Pretraining Model
    '''
    def __init__(
        self,
        text_encoder_name: str='bert-base-uncased',
        text_encoder_is_pretrained: bool=True,
        text_encoder_num_layers_to_train: int=3,
        text_embed_dim: int=768,
        graph_encoder_name: str='graphformer',
        graph_embed_dim: int=768,
        embed_dim: int=256,
        dropout: float=0.1,
        temperature: float=1.0
    ):
        super().__init__()
        self.text_encoder_name = text_encoder_name
        self.graph_encoder_name = graph_encoder_name
        
        self.graph_encoder = GraphEncoder(graph_encoder_name, text_encoder_name)
        self.text_encoder = TextEncoder(text_encoder_name, text_encoder_is_pretrained, text_encoder_num_layers_to_train)
        
        self.graph_proj = ProjectionHead(graph_embed_dim, embed_dim, dropout)
        self.text_proj = ProjectionHead(text_embed_dim * 2, embed_dim, dropout)
        
        self.temperature = temperature
        
    def forward(self, center_input, neighbor_input, neighbor_mask, all_input, all_mask):
        # get graph and text features
        graph_features = self.graph_encoder(center_input, neighbor_input, neighbor_mask)
        
        # concatenate the text embedding of the center node and the average of the text embeddings of the neighbor nodes to get the text features
        center_text_features = self.text_encoder(center_input['input_ids'], center_input['attention_mask'])
        batch_size, num_neighbor, length = all_input['input_ids'].shape
        
        neighbor_text_features = self.text_encoder(
            all_input['input_ids'].reshape(-1, length),
            all_input['attention_mask'].reshape(-1, length)
        )
        # neighbor_text_features: [batch_size * num_neighbor, text_embed_dim]
        
        neighbor_text_features = neighbor_text_features.reshape(batch_size, num_neighbor, neighbor_text_features.shape[-1])
        # neighbor_text_features: [batch_size, num_neighbor, text_embed_dim]
        
        A = neighbor_text_features
        expanded_mask = all_mask.unsqueeze(-1).expand_as(A)
        weighted_A = A * expanded_mask
        summed_A = weighted_A.sum(dim=1)
        mask_count = expanded_mask.sum(dim=1)
        averaged_A = summed_A / (mask_count + 1e-9)
        neighbor_text_features = averaged_A
        # neighbor_text_features: [batch_size, text_embed_dim]
        
        text_features = torch.hstack([center_text_features, neighbor_text_features])
        # text_features: [batch_size, text_embed_dim * 2]

        # get graph and text embeddings (with same dimension)
        #graph_embeddings = self.graph_proj(graph_features)
        #text_embeddings = self.text_proj(text_features)
        graph_embeddings = F.normalize(self.graph_proj(graph_features), p=2, dim=1)
        text_embeddings = F.normalize(self.text_proj(text_features), p=2, dim=1)
        
        # calculate the loss
        logits = (text_embeddings @ graph_embeddings.T) / self.temperature
        graphs_similarity = graph_embeddings @ graph_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (graphs_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        graphs_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (graphs_loss + texts_loss) / 2.0 # shape: (batch_size)
        
        return loss.mean()        