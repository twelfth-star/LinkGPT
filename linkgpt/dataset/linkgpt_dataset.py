from typing import List, Tuple, Dict, Callable, Set
from dataclasses import dataclass
import random
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm, trange
import torch_sparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..utils import basics
from .utils import NODE_START_TOKEN, NODE_TOKEN, PAIRWISE_START_TOKEN, PAIRWISE_TOKEN, LINKGPT_SPECIAL_TOKENS, IGNORE_INDEX
from .np_dataset import NPDataset, NPDatasetConfig, NPData
from .yn_dataset import YNTargetData, YNData, YNDatasetConfig, YNDataset


class LinkGPTDataset(Dataset):
    """
    This dataset class is designed to mix multiple datasets. Every time you call __getitem__, it will randomly choose a dataset and return a sample from it.
    """
    def __init__(self, dataset_list: List[Dataset]):
        """
        dataset_list (List[Dataset]): A list of datasets to mix. Each dataset should be either NPDataset or YNDataset.
        """
        self.dataset_list = dataset_list
        self.dataset_lengths = [len(ds) for ds in dataset_list]
        self.cumulative_lengths = [sum(self.dataset_lengths[:i+1]) for i in range(len(self.dataset_lengths))]
        
        self.lengths = []
        
        # This was added to group the texts by length, but it's abandoned in the current version. Please ignore it.
        self.column_names = ['length'] 
        for ds in dataset_list:
            self.lengths += ds['length']
                    
    def __len__(self):
        return sum(self.dataset_lengths)
    
    def __getitem__(self, index: int):
        if index == 'length':
            # This was added to group the texts by length, but it's abandoned in the current version. Please ignore it.
            return self.lengths
        
        if index < 0 or index >= self.__len__():
            raise IndexError("Index out of range")
        dataset_idx = 0
        while index >= self.cumulative_lengths[dataset_idx]:
            dataset_idx += 1
        if dataset_idx > 0:
            index -= self.cumulative_lengths[dataset_idx - 1]
        return self.dataset_list[dataset_idx][index]


class LinkGPTDataCollator():
    """
    Data collator for LinkGPTDataset, NPDataset, and YNDataset. It pads the input_ids, attention_mask, and labels to the same length, as well as dealing with the graph data.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        input_ids_list = [i[0]['input_ids'] for i in batch]
        attention_mask_list = [i[0]['attention_mask'] for i in batch]
        labels_list = [i[0]['labels'] for i in batch]
        input_ids_mat = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_mat = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels_mat = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
        graph_data_list = [i[1] for i in batch]
        graph_data = {key: [gd[key] for gd in graph_data_list] for key in graph_data_list[0].keys()}
        
        data = {
            'input_ids': input_ids_mat,
            'attention_mask': attention_mask_mat,
            'labels': labels_mat,
            'graph_data': graph_data
        }
        return data