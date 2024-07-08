import random
import json
import os
import pickle
import numpy as np
import torch
import tqdm
import torch.nn as nn
from torch_geometric.utils import degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
import dgl
import networkx as nx
import copy
import torch_geometric

def set_seeds(seed=42):
    '''
    Set seed for Python ramdom, torch and numpy.
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return seed

def save_json(data, file_name):
    """
    Save data to file_name with json format
    """
    with open(file_name, "w") as f:
        json.dump(data, f)
        
def load_json(file_name):
    """
    Load data from file_name with json format
    """
    with open(file_name, "r") as f:
        data = json.load(f)
    return data

def load_json_line(file_name):
    """
    Load data from file_name with json line (i.e., regard each line as a json file) format
    """
    data = []
    with open(file_name, "r") as f:
        for line in f:
            try:
                data.append(eval(line))
            except:
                print('ERROR: broken line!')
                continue
    return data

def save_pickle(data, file_name):
    """
    Save data from file_name with pickle format
    """
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def load_pickle(file_name):
    """
    Load data from file_name with pickle format
    """
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data

def load_dgl_graph(graph_path: str):
    """
    Load a dgl graph from graph_path
    """
    graphs, _ = dgl.load_graphs(graph_path)
    return graphs[0]

def save_dgl_graph(g, graph_path: str):
    dgl.save_graphs(graph_path, [g])
        
def get_device():
    '''
    Return 'cuda' if GPU is available, otherwise return 'cpu'.
    '''
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_token_num(text: str, tokenizer):
    """
    Return the number of tokens of a given text
    """
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)-1
    return seq_len

def get_rank(pred, n=0, descending=True):
    """
    Calculate the rank of the nth element in pred
    descending=True means large values ranks higher,
    descending=False means small values ranks higher.
    """
    arg = torch.argsort(torch.tensor(pred), descending=descending)
    rank = torch.where(arg==n)[0]+1
    return rank.tolist()[0]

def calculate_mrr(rank_ls):
    """
    Return the MRR (Mean Reciprocal Rank) of a list of ranks.
    """
    if type(rank_ls) is list:
        rk = np.array(rank_ls)
    else:
        rk = rank_ls
    return (1 / rk).mean()

def calculate_hit(rank_ls, n):
    """
    Return the Hit@n of a list of ranks.
    """
    if type(rank_ls) is list:
        rk = np.array(rank_ls)
    else:
        rk = rank_ls
    rk = np.array(rank_ls)
    return rk[rk <= n].shape[0] / rk.shape[0]

def dgl_graph_to_pyg_graph(dgl_graph):
    """
    Convert a dgl graph to a pytorch geometric graph
    """
    if dgl_graph.ndata:
        x = dgl_graph.ndata.get('feat', None)
    else:
        x = None
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    if dgl_graph.edata:
        edge_attr = dgl_graph.edata.get('feat', None)
    else:
        edge_attr = None
    pyg_graph = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyg_graph

def dgl_graph_to_nx_graph(dgl_g):
    """
    Convert a dgl graph to a networkx graph
    """
    temp_g = dgl.to_networkx(dgl_g)
    nx_g = nx.Graph()
    for u, v in temp_g.edges():
        if not nx_g.has_edge(u, v):
            nx_g.add_edge(u, v)
    for node in dgl_g.nodes():
        nx_g.add_node(node.item())
    return nx_g

# borrowed from LLaGA code
class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def partition_propagate(self, data_edge_index, x, norm, select_idx=None, chunk_size=800, cuda=False):
        if select_idx is None:
            n = x.shape[0]
            select_idx = torch.arange(n)
        else:
            n = select_idx.shape[0]
        os=[]
        for i in trange(0, n, chunk_size):
            key=select_idx[i:i+chunk_size]
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(key, 1, data_edge_index, relabel_nodes=True)
            if cuda:
                o =  self.propagate(edge_index.cuda(), x=x[subset].cuda(), norm=norm[edge_mask].cuda())
            else:
                o = self.propagate(edge_index, x=x[subset], norm=norm[edge_mask])
            os.append(o[mapping])
        return torch.cat(os, dim=0)
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

# borrowed from LLaGA code
def generate_multi_hop_x(edge_index, dataset_path: str, emb_name: str, device: str='cpu', max_hop: int=4):
    x = torch.load(os.path.join(dataset_path, f"text_emb_{emb_name}.pt"), map_location=device)
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    mp = MP()
    for i in range(max_hop):
        x = mp.propagate(edge_index, x=x, norm=norm)
        torch.save(x, os.path.join(dataset_path, f"text_emb_{emb_name}_{i+1}hop.pt"))
        

def get_pair_list(nid_ls, get_neighbors):
    nid_set = set(nid_ls)
    pair_ls = []
    for nid in nid_ls:
        neighbors = get_neighbors(nid)
        pair_ls += [(nid, neighbor) for neighbor in neighbors if neighbor in nid_set]
    return pair_ls

def to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.cpu().numpy()
    elif isinstance(a, np.ndarray):
        return a
    else:
        return np.array(a)