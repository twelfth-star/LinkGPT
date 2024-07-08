from typing import List, Tuple, Dict, Callable, Set
from dataclasses import dataclass
import random
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, trange
import torch_sparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..utils import basics
from .utils import NODE_START_TOKEN, NODE_TOKEN, PAIRWISE_START_TOKEN, PAIRWISE_TOKEN,\
    LINKGPT_SPECIAL_TOKENS, IGNORE_INDEX, get_text_with_encoding_token

# for Neighbor prediction (NP) task

@dataclass
class NPData:
    src_node: int
    neighbors: List[int]
    
@dataclass
class NPDatasetConfig:
    max_neighbor_num_per_src: int = 2 # the maximum number of neighbors per source node
    max_neighbor_num_per_prompt: float = 2 # the maximum number of neighbors per prompt
    
    # The following four are for the prompt generation. The wording can be modified to fit different tasks.
    task_desc: str = ""
    source_node_intro: str = "Source node:\n"
    question: str = "What neighbors does this node have?\n"
    
    ablate_node_encoding: bool = False
    node_encoding_max_hop: int = 0
    # the maximum hop to encode the node
    # if node_encoding_max_hop == n, then the node encoding will contain n+1 tokens, i.e., 
    # (itself, avg. of its 1-hop neighbors, avg. of its 2-hop neighbors, ..., avg. of its n-hop neighbors)
    # However, in this paper, node_encoding_max_hop is fixed to be 0
    
    learn_src_text: bool = True # whether to calculate the loss for the source node text
    learn_neighbor_text: bool = True # whether to calculate the loss for the neighbor node text
    learn_all: bool = False # whether to calculate the loss for all text (if True, learn_src_text and learn_neighbor_text will be ignored. And the training would be ordinary FT, instead of supervised FT.)
    return_tokenized: bool = True # whether to return tokenized data in __getitem__
    
    generate_at_initialization: bool = True

class NPDataset(Dataset):
    def __init__(self, dgl_graph, gnid2text: Dict[int, str], config: NPDatasetConfig, tokenizer):
        self.dgl_graph = dgl_graph
        self.gnid2text = gnid2text
        self.num_nodes = dgl_graph.num_nodes()
        self.config = config
        self.tokenizer = tokenizer
        self.lengths = None
        self.column_names = ['length'] # This was added to group the texts by length, but it's abandoned in the current version. Please ignore it.
        
        self.np_data_list = None
        if self.config.generate_at_initialization:
            self.generate_np_data_list()
        
    def generate_np_data_list(self, num_src_nodes=None):
        np_data_list = []
        
        config = self.config
        total_num = num_src_nodes if num_src_nodes is not None else self.num_nodes # for debug only
        for src in trange(total_num):
            all_neighbors = self.dgl_graph.successors(src).cpu().tolist()
            selected_neighbors = random.sample(all_neighbors, min(len(all_neighbors), config.max_neighbor_num_per_src))
            for start in range(0, len(selected_neighbors), config.max_neighbor_num_per_prompt):
                end = min(len(selected_neighbors), start + config.max_neighbor_num_per_prompt)
                cur_neighbors = selected_neighbors[start:end]
                np_data_list.append(NPData(src, cur_neighbors))
        self.np_data_list = np_data_list
    
    def __len__(self):
        if self.np_data_list is None:
            raise ValueError("NPData list not generated yet!")
        return len(self.np_data_list)

    def __getitem__(self, index: int):
        if index == 'length':
            # This was added to group the texts by length, but it's abandoned in the current version. Please ignore it.
            return self.lengths
        if self.np_data_list is None:
            raise ValueError("NPData list not generated yet!")
        np_data = self.np_data_list[index]
        config = self.config
        prompt, graph_data = np_data_to_prompt_and_graph_data(self.gnid2text, np_data, config)
        if not config.return_tokenized:
            return prompt, graph_data
        tokenizer = self.tokenizer
        encoding = tokenizer(prompt, return_tensors='pt')
        input_ids = encoding['input_ids'][0]
        attention_mask = encoding['attention_mask'][0]
        labels = generate_np_labels(tokenizer, input_ids, learn_src_text=config.learn_src_text, learn_neighbor_text=config.learn_neighbor_text, learn_all=config.learn_all)
        return ({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }, graph_data)
    
    def set_lengths(self):
        # This was added to group the texts by length, but it's abandoned in the current version. Please ignore it.
        old_config = self.config
        length_list = []
        self.config.return_tokenized = True
        for i in trange(self.__len__()):
            data = self.__getitem__(i)
            length = data[0]['input_ids'].shape[0]
            length_list.append(length)
            
        self.config = old_config
        self.lengths = length_list
        
    
def np_data_to_prompt_and_graph_data(gnid2text: Dict[int, str], np_data: NPData, config: NPDatasetConfig):
    graph_data = {
        'source_node': -1,
        'node_id_ls': [],
        'pairwise_target_id_ls': [],
    }
    src = np_data.src_node
    graph_data['source_node'] = src
    src_text = get_text_with_encoding_token(src, config, gnid2text, do_pairwise=False) + '\n'
    graph_data['node_id_ls'] += [(src, i) for i in range(config.node_encoding_max_hop + 1)]
    
    neighbor_text_ls = ["text: " + gnid2text[neighbor] for neighbor in np_data.neighbors]
    neighbor_text_all = '\n'.join(neighbor_text_ls)
    
    prompt = f"{config.task_desc}{config.source_node_intro}{src_text}{config.question}Answer:\n{neighbor_text_all}\n"
    return prompt, graph_data


def generate_np_labels(tokenizer, input_ids, learn_src_text: bool=True, learn_neighbor_text: bool=True, learn_all: bool=False):
    """
    Generate labels for the NP task.
    Args:
        tokenizer: the tokenizer
        input_ids: the input_ids of the prompt
        learn_src_text: whether to calculate the loss for the source node text
        learn_neighbor_text: whether to calculate the loss for the neighbor node text
        learn_all: whether to calculate the loss for all text (if True, learn_src_text and learn_neighbor_text will be ignored. And the training would be ordinary FT, instead of supervised FT.)
    """
    if learn_all:
        labels = input_ids[1:]
        return labels
    labels = (IGNORE_INDEX * torch.ones(len(input_ids)-1)).long()
    
    if learn_src_text:
        newline_id = tokenizer('x\nx', add_special_tokens=False)['input_ids'][1]
        text_prefix_ids = tokenizer.convert_tokens_to_ids(['text', ':'])
        potential_pos_list = torch.where(input_ids == text_prefix_ids[0])[0].tolist()
        real_pos_list = [pos for pos in potential_pos_list if pos+1 < len(input_ids) and input_ids[pos+1] == text_prefix_ids[1]]
        start_pos_list = [pos + 2 for pos in real_pos_list]
        start_pos = start_pos_list[0]
        end_pos = torch.where(input_ids[start_pos:] == newline_id)[0][0].item() + start_pos + 1
        labels[(start_pos-1):(end_pos-1)] = input_ids[start_pos:end_pos]
    if learn_neighbor_text:
        newline_id = tokenizer('x\nx', add_special_tokens=False)['input_ids'][1]
        text_prefix_ids = tokenizer.convert_tokens_to_ids(['Answer', ':'])
        potential_pos_list = torch.where(input_ids == text_prefix_ids[0])[0].tolist()
        real_pos_list = [pos for pos in potential_pos_list if pos+1 < len(input_ids) and input_ids[pos+1] == text_prefix_ids[1]]
        start_pos_list = [pos + 3 for pos in real_pos_list]
        start_pos = start_pos_list[0]
        end_pos = len(input_ids)
        labels[(start_pos-1):(end_pos-1)] = input_ids[start_pos:end_pos]
        
    return labels