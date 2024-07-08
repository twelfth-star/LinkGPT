"""
Adapted from https://github.com/HarryShomer/LPFormer/blob/master/src/util/read_datasets.py
"""

import os
import argparse
import sys
import random

import torch
import numpy as np
from torch_sparse import SparseTensor
from torch.nn.init import xavier_uniform_
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, degree
import joblib  # Make ogb loads faster...idk
from ogb.linkproppred import PygLinkPropPredDataset
import dgl

from ..dataset.tag_dataset_for_lm import TAGDatasetForLM
from ..utils import basics

def get_lpformer_dataset(dataset_name: str, split_edge, dgl_g, ppr_data, device):
    """
    Returns:
        dataset for LPFormer
    """
    data_obj = {
        "dataset": dataset_name,
    }
    
    data_obj['num_nodes'] = dgl_g.num_nodes()

    source, target = split_edge['train']['source_node'], split_edge['train']['target_node']
    source, target = torch.tensor(source), torch.tensor(target)
    data_obj['train_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)

    source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
    source, target = torch.tensor(source), torch.tensor(target)
    data_obj['valid_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
    pair_list = []
    for src, neg_tgt_list in zip(split_edge['valid']['source_node'], split_edge['valid']['target_node_neg']):
        pair_list += [[src, tgt] for tgt in neg_tgt_list]
    selected_items = random.sample(pair_list, 10000)
    data_obj['valid_neg'] = torch.tensor(selected_items).to(device) 

    source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
    source, target = torch.tensor(source), torch.tensor(target)
    data_obj['test_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
    pair_list = []
    for src, neg_tgt_list in zip(split_edge['test']['source_node'], split_edge['test']['target_node_neg']):
        pair_list += [[src, tgt] for tgt in neg_tgt_list]
    selected_items = random.sample(pair_list, 10000)
    data_obj['test_neg'] = torch.tensor(selected_items).to(device) 
    
    idx = torch.randperm(data_obj['train_pos'].size(0))[:data_obj['valid_pos'].size(0)]
    data_obj['train_pos_val'] = data_obj['train_pos'][idx]
    
    data_obj['x'] = dgl_g.ndata['feat'].to(device).to(torch.float)
    
    src_ls, tgt_ls = dgl_g.edges()
    src_ls, tgt_ls = src_ls.reshape(1, -1), tgt_ls.reshape(1, -1)
    edge_index = torch.concat([src_ls, tgt_ls], dim=0).to(device)
    
    edge_weight = torch.ones(edge_index.size(1)).to(device).float()
    data_obj['adj_t'] = SparseTensor.from_edge_index(
        edge_index, 
        edge_weight.squeeze(-1),
        [data_obj['num_nodes'], data_obj['num_nodes']]
    ).to(device)
    data_obj['adj_t'] = data_obj['adj_t'].to_symmetric().coalesce().to(device)
    data_obj['adj_mask'] = data_obj['adj_t'].to_symmetric().to_torch_sparse_coo_tensor()
    data_obj['adj_mask'] = data_obj['adj_mask'].coalesce().bool().int()
    data_obj['full_adj_t'] = data_obj['adj_t']
    data_obj['full_adj_mask'] = data_obj['adj_mask']
    data_obj['degree'] = degree(edge_index[0], num_nodes=data_obj['num_nodes']).to(device)
    
    data_obj['ppr'] = ppr_data.to(device)
    data_obj['ppr'] = data_obj['ppr'].to_torch_sparse_coo_tensor()
    data_obj['ppr_test'] = data_obj['ppr']
    
    train_pos = data_obj['train_pos'].cpu().numpy().tolist()
    pair_to_edge_idx = {tuple(pair): idx for idx, pair in enumerate(train_pos)}
    data_obj['pair_to_edge_idx'] = pair_to_edge_idx

    return data_obj