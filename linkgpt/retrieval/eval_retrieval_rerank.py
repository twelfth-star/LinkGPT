from typing import List, Optional, Tuple, Union, Dict
import pickle
import json
import os
import random
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
import dgl
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import DataCollatorForLanguageModeling, Trainer, HfArgumentParser
from transformers import LlamaForCausalLM, LlamaTokenizer
import wandb
from peft import PeftConfig, PeftModel
from tqdm import tqdm, trange
from rank_bm25 import BM25Okapi
import networkx as nx
import llmtuner
from llmtuner.model.patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
from llmtuner.hparams.parser import get_train_args
import llmtuner.hparams.parser as llm_tuner_parser
from llmtuner.extras.misc import count_parameters
from llmtuner.model.loader import init_adapter

project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")) # path to LinkGPT
if project_path not in sys.path:
    sys.path.insert(0, project_path)
from linkgpt.dataset.tag_dataset_for_lm import TAGDatasetForLM, tag_dataset_for_lm_to_dgl_graph
from linkgpt.pairwise_encoding.lpformer_dataset import get_lpformer_dataset
from linkgpt.pairwise_encoding.models.link_transformer import LinkTransformer
from linkgpt.pairwise_encoding.lpformer_model_api import get_lpformer_model
from linkgpt.model.linkgpt_model import LinkGPTForCausalLM, LinkGPTConfig, \
    unfreeze_graph_related_modules, unfreeze_lora_adapter, freeze_all_parameters, \
        save_lora_model, get_model_and_tokenizer, load_model_and_tokenizer, get_tokenizer
from linkgpt.dataset.np_dataset import NPData, NPDatasetConfig, np_data_to_prompt_and_graph_data
from linkgpt.dataset.utils import NODE_START_TOKEN, NODE_TOKEN, PAIRWISE_START_TOKEN, PAIRWISE_TOKEN, LINKGPT_SPECIAL_TOKENS
from linkgpt.utils import basics

def print_top_n(scores_ls, documents, n: int=10):
    """
    Print the top n documents based on the scores
    """
    top_n = scores_ls.argsort()[::-1][:n]
    for rank, idx in enumerate(top_n):
        print(f"rank: {rank+1}\tscore: {scores_ls[idx]:3.3f}{' (√)' if idx == 0 else ''}\ttext: {documents[idx]}")

def do_retrieval(
    dataset_for_lm: TAGDatasetForLM,
    question_data: Dict[str, Union[List[int], List[List[int]]]],
    prediction_list: List[List[str]],
    start: Optional[int]=None,
    end: Optional[int]=None,
    num_neg_tgt: int=1800,
    num_to_retrieve: int=30,
    apply_dist_based_grouping: bool=True,
    max_dist: int=2,
    max_num_in_neighborhood: int=10,
):
    """
    Perform retrieval based on BM25
    """
    if apply_dist_based_grouping:
        print(f"Applying distance-based grouping with max_dist={max_dist} and max_num_in_neighborhood={max_num_in_neighborhood}")
    else:
        print("Not applying distance-based grouping. The performance may be worse.")
    
    start = 0 if start is None else start
    end = len(prediction_list) if end is None else end
    
    final_set_list = []
    is_retrieved_list = []
    rank_list = []
    gnid2text = {i: dataset_for_lm[i][dataset_for_lm.text_field] for i in range(len(dataset_for_lm))}
    dgl_graph = tag_dataset_for_lm_to_dgl_graph(dataset_for_lm, include_valid=True)
    nx_graph = basics.dgl_graph_to_nx_graph(dgl_graph.cpu())
    for idx in trange(start, end):
        src = question_data['source_node'][idx]
        tgt = question_data['target_node'][idx]
        neg_tgt_list = question_data['target_node_neg'][idx][:num_neg_tgt]

        tgt_text = gnid2text[tgt]
        neg_tgt_text_ls = [gnid2text[gnid] for gnid in neg_tgt_list]

        neighbors_in_train = dataset_for_lm.get_neighbors_in_training_set(src)

        queries = prediction_list[idx]
        if len(queries) == 0:
            # If no queries are found, use an empty query
            queries = ['']
            
        documents = [tgt_text] + neg_tgt_text_ls
        tokenized_documents = [doc.split(" ") for doc in documents]
        bm25 = BM25Okapi(tokenized_documents)
        scores_ls = [bm25.get_scores(query.split(" ")) for query in queries]
        scores_ls = np.sum(scores_ls, axis=0)
        rank = np.where(scores_ls.argsort()[::-1] == 0)[0][0]+1 # rank of the positive target candidate node
        
        cand_tgt_list = [tgt] + neg_tgt_list
        combined_list = list(zip(cand_tgt_list, scores_ls))
        combined_list.sort(key=lambda x: x[1], reverse=True)
    
        if not apply_dist_based_grouping:
            # do not apply distance-based grouping, simply retrieve the top num_to_retrieve candidates
            final_set = {t[0] for t in combined_list[:num_to_retrieve]}
        else:
            # apply distance-based grouping
            # retrieve the top max_num_in_neighborhood candidates in the neighborhood
            # and then retrieve the rest from the remaining candidates
            neighbor_set = set(nx.single_source_shortest_path_length(nx_graph, source=src, cutoff=max_dist).keys())
            in_neighbor = [item for item in combined_list if item[0] in neighbor_set]
            not_in_neighbor = [item for item in combined_list if item[0] not in neighbor_set]
            top_in_neighbor = in_neighbor[:max_num_in_neighborhood]
            top_not_in_neighbor = not_in_neighbor[:num_to_retrieve-len(top_in_neighbor)]
            final_set = {t[0] for t in top_in_neighbor}.union({t[0] for t in top_not_in_neighbor})
        
        is_retrieved = 1 if tgt in final_set else 0
        final_set_list.append(final_set)
        rank_list.append(rank)
        is_retrieved_list.append(is_retrieved)
    
    is_retrieved_list = np.array(is_retrieved_list)
    rank_list = np.array(rank_list)
    
    return is_retrieved_list, rank_list, final_set_list

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_list_path', required=True)
    parser.add_argument('--dataset_for_lm_path', required=True)
    parser.add_argument('--eval_dataset_path', required=True)
    parser.add_argument('--eval_output_path', required=True)
    parser.add_argument('--result_saving_path', required=True)
    parser.add_argument('--dataset_name', required=True)

    parser.add_argument('--num_neg_tgt', default=1800, type=int)
    parser.add_argument('--num_to_retrieve', default=30, type=int)

    parser.add_argument('--apply_dist_based_grouping', action='store_true')
    parser.add_argument('--max_dist', default=2, type=int)
    parser.add_argument('--beta', default=0.65, type=float, help="the factor used in distance-based grouping, refer to our paper for more details")

    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    prediction_list = basics.load_json(args.prediction_list_path)
    dataset_for_lm = basics.load_pickle(args.dataset_for_lm_path)
    eval_dataset = basics.load_pickle(args.eval_dataset_path)
    eval_output = basics.load_json(args.eval_output_path)
    num_neg_tgt = args.num_neg_tgt
    num_to_retrieve = args.num_to_retrieve
    apply_dist_based_grouping = args.apply_dist_based_grouping
    max_dist = args.max_dist
    beta = args.beta
    
    orig_rank_list = [basics.get_rank(i) for i in eval_output]

    
    stats_text = f'MRR={basics.calculate_mrr(orig_rank_list)}, Hit@1={basics.calculate_hit(orig_rank_list, 1)}, Hit@10={basics.calculate_hit(orig_rank_list, 10)}'
    
    f_out = open(args.result_saving_path, 'w')
    f_out.write(f"LinkGPT w/o retrieval:\n{stats_text}\n\n")
    print(f"LinkGPT w/o retrieval:\n{stats_text}\n")
    
    is_retrieved_list, rank_list, final_set_list = do_retrieval(
        dataset_for_lm=dataset_for_lm,
        question_data=eval_dataset.question_data,
        prediction_list=prediction_list,
        start=None,
        end=None,
        num_neg_tgt=num_neg_tgt,
        num_to_retrieve=num_to_retrieve,
        apply_dist_based_grouping=apply_dist_based_grouping,
        max_dist=max_dist,
        max_num_in_neighborhood=int(num_to_retrieve * beta),
    )
    print(f'Retrieval rate: {is_retrieved_list.mean()}')

    rank_list = []
    for idx in trange(len(final_set_list)):
        src = eval_dataset.question_data['source_node'][idx]
        tgt = eval_dataset.question_data['target_node'][idx]
        neg_tgts = eval_dataset.question_data['target_node_neg'][idx]
        if tgt not in final_set_list[idx]:
            # If the positive target is not in the final set, assign a rank of infinity
            rank_list.append(float('inf'))
            continue
        for i in final_set_list[idx]:
            if i not in [tgt] + neg_tgts:
                print('ERROR: Invalid node in final set.')
        tgt2val = {tgt_node:val for tgt_node, val in zip([tgt] + neg_tgts, eval_output[idx])}
        val_list = [tgt2val[tgt]] + \
            [tgt2val[i] for i in final_set_list[idx] if i != tgt]
        assert len(val_list) == len(final_set_list[idx])
        rank = basics.get_rank(val_list)
        rank_list.append(rank)

    
    stats_text = f'MRR={basics.calculate_mrr(rank_list)}, Hit@1={basics.calculate_hit(rank_list, 1)}, Hit@10={basics.calculate_hit(rank_list, 10)}'
    f_out.write(f"LinkGPT:\n{stats_text}\n")
    f_out.close()
    print(f"LinkGPT:\n{stats_text}")
    
    
if __name__ == '__main__':
    main()