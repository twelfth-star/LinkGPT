"""
Precompute the PPR scores for a dataset.

Scores are used to select nodes for attention

Adapted from https://github.com/HarryShomer/LPFormer
"""

import os
import sys
import argparse
from time import time

import torch
import numpy as np
import pandas as pd
from torch_sparse import SparseTensor 
from torch_geometric.utils import coalesce, to_undirected
import torch_geometric.transforms as T
import joblib  # Make ogb loads faster
from ogb.linkproppred import PygLinkPropPredDataset
import dgl

project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")) # path to LinkGPT
if project_path not in sys.path:
    sys.path.insert(0, project_path)
from linkgpt.utils import basics
from linkgpt.dataset.tag_dataset_for_lm import tag_dataset_for_lm_to_dgl_graph

def get_ppr_matrix(edge_index, num_nodes, alpha=0.15, eps=5e-5):
    """
    Calc PPR data

    Returns scores and the corresponding nodes

    Adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/transforms/gdc.py
    
    Args:
        edge_index (torch.Tensor): edge index, shape is [2, m], where m is the number of edges.
            edge_index must be symmetric, i.e., if (a, b) in edge_index, then (b, a) must be in edge_index
        num_nodes (int): number of nodes
        alpha (float): Teleportation probability
        eps (float): Stopping criterion threshold
    """
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index_np = edge_index.cpu().numpy()

    # Assumes sorted and coalesced edge indices (NOTE: coalesce also sorts edges)
    indptr = torch._convert_indices_from_coo_to_csr(edge_index[0], num_nodes).cpu().numpy()
    
    out_degree = indptr[1:] - indptr[:-1]
    
    start = time()
    print("Calculating PPR...", flush=True)
    neighbors, neighbor_weights = get_calc_ppr()(indptr, edge_index_np[1], out_degree, alpha, eps)
    print(f"Time: {time()-start:.2f} seconds")

    print("\n# Nodes with 0 PPR scores:", sum([len(x) == 1 for x in neighbors]))  # 1 bec. itself
    print(f"Mean # of scores per Node: {np.mean([len(x) for x in neighbors]):.1f}")

    return neighbors, neighbor_weights

def get_calc_ppr():
    """
    Courtesy of https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/transforms/gdc.py
    """
    import numba

    @numba.jit(nopython=True, parallel=True)
    def calc_ppr(
        indptr: np.ndarray,
        indices: np.ndarray,
        out_degree: np.ndarray,
        alpha: float,
        eps: float,
    ):
        r"""Calculate the personalized PageRank vector for all nodes
        using a variant of the Andersen algorithm
        (see Andersen et al. :Local Graph Partitioning using PageRank Vectors.)

        Args:
            indptr (np.ndarray): Index pointer for the sparse matrix
                (CSR-format).
            indices (np.ndarray): Indices of the sparse matrix entries
                (CSR-format).
            out_degree (np.ndarray): Out-degree of each node.
            alpha (float): Alpha of the PageRank to calculate.
            eps (float): Threshold for PPR calculation stopping criterion
                (:obj:`edge_weight >= eps * out_degree`).

        :rtype: (:class:`List[List[int]]`, :class:`List[List[float]]`)
        """
        alpha_eps = alpha * eps
        js = [[0]] * len(out_degree)
        vals = [[0.]] * len(out_degree)
        for inode_uint in numba.prange(len(out_degree)):
            inode = numba.int64(inode_uint)
            p = {inode: 0.0}
            r = {}
            r[inode] = alpha
            q = [inode]
            while len(q) > 0:
                unode = q.pop()

                res = r[unode] if unode in r else 0
                if unode in p:
                    p[unode] += res
                else:
                    p[unode] = res
                r[unode] = 0
                for vnode in indices[indptr[unode]:indptr[unode + 1]]:
                    _val = (1 - alpha) * res / out_degree[unode]
                    if vnode in r:
                        r[vnode] += _val
                    else:
                        r[vnode] = _val

                    res_vnode = r[vnode] if vnode in r else 0
                    if res_vnode >= alpha_eps * out_degree[vnode]:
                        if vnode not in q:
                            q.append(vnode)
            js[inode] = list(p.keys())
            vals[inode] = list(p.values())

        return js, vals

    return calc_ppr

def create_sparse_ppr_matrix(neighbors, neighbor_weights):
    """
    For all calculated pairs, we can arrange in a NxN sparse weighted Adj matrix 
    """
    ppr_scores = []
    source_edge_ix, target_edge_ix = [], []
    for source_ix, (source_neighbors, source_weights) in enumerate(zip(neighbors, neighbor_weights)):
        source_edge_ix.extend([source_ix] * len(source_neighbors))
        target_edge_ix.extend(source_neighbors)
        ppr_scores.extend(source_weights)

    source_edge_ix = torch.Tensor(source_edge_ix).unsqueeze(0)
    target_edge_ix = torch.Tensor(target_edge_ix).unsqueeze(0)

    ppr_scores = torch.Tensor(ppr_scores)
    edge_ix = torch.cat((source_edge_ix, target_edge_ix), dim=0).long()

    num_nodes = len(neighbors)
    sparse_adj = SparseTensor.from_edge_index(edge_ix, ppr_scores, [num_nodes, num_nodes])

    return sparse_adj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--dataset_for_lm_path', required=True, type=str)
    parser.add_argument('--ppr_data_save_path', required=True, type=str)

    parser.add_argument("--eps", help="Stopping criterion threshold", type=float, default=5e-5)
    parser.add_argument("--alpha", help="Teleportation probability", type=float, default=0.15)

    args = parser.parse_args()
    
    dataset_for_lm = basics.load_pickle(args.dataset_for_lm_path)
    g = tag_dataset_for_lm_to_dgl_graph(dataset_for_lm)
    src_node_ls, tgt_node_ls = g.edges()
    src_node_ls, tgt_node_ls = src_node_ls.reshape(1, -1), tgt_node_ls.reshape(1, -1)
    edge_index = torch.concat([src_node_ls, tgt_node_ls], dim=0)
    num_nodes = g.num_nodes()

    neighbors, neighbor_weights = get_ppr_matrix(edge_index, num_nodes, args.alpha, args.eps)
    sparse_adj = create_sparse_ppr_matrix(neighbors, neighbor_weights)
    torch.save(sparse_adj, args.ppr_data_save_path)

if __name__ == "__main__":
    main()

