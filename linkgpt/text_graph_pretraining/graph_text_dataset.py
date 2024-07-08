import random
from typing import Callable

import numpy as np
import torch
from torch.utils import data
from transformers import BertTokenizer, BertModel, BertConfig

from ..dataset.tag_dataset_for_lm import TAGDatasetForLM

class CGTPDataset(data.Dataset):
    """
    Dataset for Contrastive Graph Text Pretraining (CGTP)
    """
    def __init__(
        self,
        get_text: Callable,
        num_neighbor: int,
        tag_dataset_for_lm: TAGDatasetForLM,
        text_encoder_name: str='bert-base-uncased',
        max_length: int=64
    ):
        """
        Args:
            get_text: function, input: tag_dataset_for_lm.data_list[i], output: corresponding text 
            num_neighbor (int): number of neighbors to sample
            tag_dataset_for_lm (TAGDatasetForLM)
            text_encoder_name (str): 
            max_length (int): max number of tokens
        """
        self.num_neighbor = num_neighbor
        self.tokenizer = BertTokenizer.from_pretrained(text_encoder_name)
        self.tag_dataset_for_lm = tag_dataset_for_lm
        self.get_text = get_text
        self.max_length = max_length
    
    def __getitem__(self, idx: int):
        '''
        Output:
            center_input: [B, L]
            neighbor_input: [B, N, L]
            neighbor_mask: [B, N]
            All have key "input_ids" and "attention_mask", with shapes indicated above
            
            B: batch size
            N: number of neighbors
            L: hidden dimensions
        '''
        gnid = idx
        center_text = self.get_text(self.tag_dataset_for_lm.data_list[gnid])
        
        neighbor_gnids = self.tag_dataset_for_lm.get_neighbors_in_training_set(gnid)
        full_neighbor_texts = [self.get_text(self.tag_dataset_for_lm.data_list[gnid]) for gnid in neighbor_gnids]
        
        center_input, neighbor_input, neighbor_mask, all_input, all_mask = text_to_cgtp_input(center_text, full_neighbor_texts, self.tokenizer, self.max_length, self.num_neighbor)
        return center_input, neighbor_input, neighbor_mask, all_input, all_mask
        
    def __len__(self):
        return len(self.tag_dataset_for_lm.data_list)


def text_to_cgtp_input(
    center_text: str,
    full_neighbor_texts: list,
    tokenizer,
    max_length: int,
    num_neighbor: int
):
    neighbor_texts = random.sample(full_neighbor_texts, min(len(full_neighbor_texts), num_neighbor)) + [''] * (num_neighbor - len(full_neighbor_texts))
    neighbor_mask = torch.ones(num_neighbor)
    for i in range(num_neighbor):
        if neighbor_texts[i] == '':
            neighbor_mask[i] = 0
    
    all_input = tokenizer([center_text] + neighbor_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    center_input = {
        'input_ids': all_input['input_ids'][0],
        'attention_mask': all_input['attention_mask'][0],
    }
    neighbor_input = {
        'input_ids': all_input['input_ids'][1:],
        'attention_mask': all_input['attention_mask'][1:]
    }
    
    all_mask = torch.tensor([1] + neighbor_mask.numpy().tolist())
    return center_input, neighbor_input, neighbor_mask, all_input, all_mask