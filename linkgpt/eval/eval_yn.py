from typing import List, Optional, Tuple, Union, Dict
import pickle
import json
import os
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
from linkgpt.dataset.utils import NODE_START_TOKEN, NODE_TOKEN, PAIRWISE_START_TOKEN, PAIRWISE_TOKEN, LINKGPT_SPECIAL_TOKENS
from linkgpt.utils import basics

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--text_embedding_method', required=True)
    parser.add_argument('--text_embedding_folder_path', required=True)
    parser.add_argument('--dataset_for_lm_path', required=True)
    parser.add_argument('--eval_dataset_path', required=True)
    parser.add_argument('--ppr_data_path', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--ft_model_path', default=None)
    parser.add_argument('--stage', default=2, type=int)
    parser.add_argument('--max_hop', default=0, type=int)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output_path', required=True) # Path to save the Yes/No Difference Score Data
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--ablate_pairwise_encoding', action='store_true')
    parser.add_argument('--ablate_node_encoding', action='store_true')

    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--report_to', default=None)
    parser.add_argument('--wandb_key', default=None, type=str)
    parser.add_argument('--wandb_project_name', default='untitled', type=str)
    parser.add_argument('--wandb_run_name', default='untitled_run', type=str)
    parser.add_argument('--logging_steps', default=1, type=int)
    parser.add_argument('--saving_steps', default=500, type=int)
    parser.add_argument('--max_steps', default=2000, type=int)
    
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=2000, type=int)

    args = parser.parse_args()

    if args.wandb_key:
        if args.wandb_key == 'None':
            args.report_to = None
        else:
            wandb.login(key=args.wandb_key)
            os.environ["WANDB_PROJECT"] = args.wandb_project_name
        
    if args.report_to == 'wandb':
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
        )
    
    if args.output_path is not None:
        output_directory = os.path.dirname(args.output_path)
        os.makedirs(output_directory, exist_ok=True)

    device = args.device
    
    # Load dataset
    text_emb_list = []
    for i in range(args.max_hop + 1):
        if i == 0:
            text_emb_path = os.path.join(args.text_embedding_folder_path, f'text_emb_{args.text_embedding_method}.pt')
        else:
            text_emb_path = os.path.join(args.text_embedding_folder_path, f'text_emb_{args.text_embedding_method}_{i}hop.pt')
            
        print(text_emb_path, flush=True)
        text_emb = torch.load(text_emb_path, map_location=device)
        text_emb_list.append(text_emb)

    dataset_for_lm = basics.load_pickle(args.dataset_for_lm_path)
    ppr_data = torch.load(args.ppr_data_path).to(device)

    eval_dataset = basics.load_pickle(args.eval_dataset_path)
    eval_dataset.config.ablate_node_encoding = args.ablate_node_encoding
    eval_dataset.config.ablate_pairwise_encoding = args.ablate_pairwise_encoding
    eval_dataset.config.node_encoding_max_hop = args.max_hop

    dgl_graph = tag_dataset_for_lm_to_dgl_graph(dataset_for_lm, include_valid=True).to(device)
    dgl_graph.ndata['feat'] = text_emb_list[0]
    
    # Load pairwise encoder
    lpformer_dataset = get_lpformer_dataset(args.dataset_name, dataset_for_lm.edge_split, dgl_graph, ppr_data, device)
    lpformer_model = get_lpformer_model(lpformer_dataset, device).to(device)
    
    if args.ft_model_path is not None:
        model, tokenizer = load_model_and_tokenizer(
            args.model_name_or_path,
            device,
            text_emb_list,
            lpformer_model,
            args.ft_model_path,
            args.stage,
            torch.float16 if args.fp16 else torch.float32
        )
    else:
        # args.ft_model_path is None, load the vanilla model
        tokenizer = get_tokenizer(args.model_name_or_path)
        model = LinkGPTForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        model.model.set_node_encoder(text_emb_list)
        model.model.set_pairwise_encoder(lpformer_model)
        special_token_id_ls = [tokenizer.vocab[token] for token in LINKGPT_SPECIAL_TOKENS]
        model.model.set_linkgpt_special_token_emb(special_token_id_ls, LINKGPT_SPECIAL_TOKENS)
    
    model.eval()

    yn_diff_data = []
    rank_list = []

    total_num = len(eval_dataset)
    for idx in trange(total_num):
        if idx < args.start_index:
            continue
        candidate_list = eval_dataset[idx]
        yn_diff_list = eval_one(candidate_list, model, tokenizer, device, batch_size=args.batch_size, verbose=False)
        yn_diff_data.append(yn_diff_list)
        rank = basics.get_rank(yn_diff_list)
        rank_list.append(rank)
        if args.report_to == 'wandb':
            if idx % args.logging_steps == 0:
                mrr = basics.calculate_mrr(rank_ls=rank_list)
                hit_1 = basics.calculate_hit(rank_ls=rank_list, n=1)
                hit_10 = basics.calculate_hit(rank_ls=rank_list, n=10)
                wandb.log({'rank': rank, 'MRR': mrr, 'Hit@1': hit_1, 'Hit@10': hit_10,}, step=idx)
            else:
                wandb.log({'rank': rank,}, step=idx)
        else:
            if idx % args.logging_steps == 0:
                mrr = basics.calculate_mrr(rank_ls=rank_list)
                hit_1 = basics.calculate_hit(rank_ls=rank_list, n=1)
                hit_10 = basics.calculate_hit(rank_ls=rank_list, n=10)
                print(f"idx={idx}, rank={rank}, MRR={mrr}, Hit@1={hit_1}, Hit@10={hit_10}")
            else:
                print(f"idx={idx}, rank={rank}")
        
        if idx % args.saving_steps == 0 and args.output_path is not None:
            basics.save_json(yn_diff_data, args.output_path)
        if idx == args.end_index:
            break
    if args.output_path is not None:
        basics.save_json(yn_diff_data, args.output_path)
        
    mrr = basics.calculate_mrr(rank_ls=rank_list)
    hit_1 = basics.calculate_hit(rank_ls=rank_list, n=1)
    hit_10 = basics.calculate_hit(rank_ls=rank_list, n=10)
    print(f"Final result: MRR={mrr}, Hit@1={hit_1}, Hit@10={hit_10}")

def get_real_shared_prefix_length(sample_prompt_list: List[str], tokenizer):
    """
    Get the length of the shared prefix of the sample_prompt
    Doesn't work directly since special tokens (e.g., <node>) for different nodes are regarded as the same in this function
    """
    input_ids = tokenizer(sample_prompt_list, return_tensors="pt", padding=True)['input_ids']
    bool_tensor = (input_ids == input_ids[0]).sum(dim=0) == input_ids.shape[0]
    false_positions = torch.nonzero(bool_tensor == False).squeeze()
    return false_positions[0]

def get_shared_prefix_length(sample_prompt_list: List[str], tokenizer):
    """
    Get the length of the shared prefix of the sample_prompt_list
    """
    prompt = sample_prompt_list[0]
    input_ids = tokenizer([prompt], return_tensors="pt")['input_ids'][0]
    
    pairwise_start_token_pos = torch.where(input_ids == tokenizer.convert_tokens_to_ids(PAIRWISE_START_TOKEN))[0]
    last_pairwise_start_token_pos = pairwise_start_token_pos[-1] if len(pairwise_start_token_pos) > 0 else np.Inf

    node_start_token_pos = torch.where(input_ids == tokenizer.convert_tokens_to_ids(NODE_START_TOKEN))[0]
    last_node_start_token_pos = node_start_token_pos[-1] if len(node_start_token_pos) > 0 else np.Inf

    real_shared_prefix_length = get_real_shared_prefix_length(sample_prompt_list, tokenizer)

    return min(last_pairwise_start_token_pos, last_node_start_token_pos, real_shared_prefix_length)

def get_batch_shared_past_key_values(shared_past_key_values: torch.Tensor, batch_size: int=4):
    """
    Repeat the shared_past_key_values for batch_size times
    """
    batch_shared_past_key_values = []
    for key, value in shared_past_key_values:
        batch_key = key.repeat(batch_size, 1, 1, 1)
        batch_value = value.repeat(batch_size, 1, 1, 1)
        batch_shared_past_key_values.append((batch_key, batch_value))
    return tuple(batch_shared_past_key_values)

def get_last_token_pos(input_ids: torch.Tensor, padding_id: int):
    """
    Get the position of the last non-padding token in each sample
    """
    not_padding = input_ids != padding_id
    not_padding_flipped = not_padding.flip(dims=[1])
    last_not_padding_indices = not_padding_flipped.int().argmax(dim=1)
    corrected_indices = input_ids.shape[1] - 1 - last_not_padding_indices
    return corrected_indices

def eval_one(candidate_list: List[Tuple[str, Dict]], model, tokenizer, device, batch_size: int=4, verbose: bool=False) -> List[float]:
    """
    Evaluate the Yes/No difference score for a single sample. 
    Note that logit_yes - logit_no can represent the Yes/No Index proposed in the original paper.
    This is because p("Yes" | context) = exp(logit_yes) / (sum of all exp(logit)).
    """
    prompt_list = [i[0] for i in candidate_list]
    graph_data_list = [i[1] for i in candidate_list]


    # calculate the shared prefix length
    num_sample = min(30, len(prompt_list))
    shared_prefix_length = get_shared_prefix_length(prompt_list[:num_sample], tokenizer)
    shared_prefix_ids = tokenizer(prompt_list[0], return_tensors="pt")['input_ids'][0][:shared_prefix_length].reshape(1, -1).to(device)
    
    num_node_tokens = (shared_prefix_ids == tokenizer.convert_tokens_to_ids(NODE_TOKEN)).sum().item()
    num_pairwise_tokens = (shared_prefix_ids == tokenizer.convert_tokens_to_ids(PAIRWISE_TOKEN)).sum().item()
    shared_graph_data = {
        'source_node': [graph_data_list[0]['source_node']],
        'node_id_ls': [graph_data_list[0]['node_id_ls'][:num_node_tokens]],
        'pairwise_target_id_ls': [graph_data_list[0]['pairwise_target_id_ls'][:num_pairwise_tokens]],
    }
    with torch.no_grad():
        output = model(input_ids=shared_prefix_ids, graph_data=shared_graph_data)
        shared_past_key_values = output.past_key_values
        del output
        
    yes_id = tokenizer(": Yes")['input_ids'][-1]
    no_id = tokenizer(": No")['input_ids'][-1]
    yn_diff_list = []
    for start in trange(0, len(prompt_list), batch_size, disable=not verbose):
        end = min(start + batch_size, len(prompt_list))
        batch_prompts = prompt_list[start:end]
        batch_graph_data = graph_data_list[start:end]

        encoded_input = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        suffix_encoded_input = {
            'input_ids': encoded_input['input_ids'][:, shared_prefix_length:].to(device),
            'attention_mask': encoded_input['attention_mask'][:, :].to(device),
        }
        batch_shared_past_key_values = get_batch_shared_past_key_values(shared_past_key_values, len(batch_prompts))

        batch_suffix_graph_data = {
            'source_node': [gd['source_node'] for gd in batch_graph_data],
            'node_id_ls': [gd['node_id_ls'][num_node_tokens:] for gd in batch_graph_data],
            'pairwise_target_id_ls': [gd['pairwise_target_id_ls'][num_pairwise_tokens:] for gd in batch_graph_data],
        }
        
        with torch.no_grad():
            batch_output = model(**suffix_encoded_input, graph_data=batch_suffix_graph_data, past_key_values=batch_shared_past_key_values)
            del batch_shared_past_key_values
        
        last_token_pos_list = get_last_token_pos(encoded_input['input_ids'], tokenizer.pad_token_id) # last token is either "__Yes" or "__No"
        for bid, last_token_pos in enumerate(last_token_pos_list):
            assert encoded_input['input_ids'][bid][last_token_pos] == yes_id or encoded_input['input_ids'][bid][last_token_pos] == no_id
            
        batch_yn_diff = []
        for j, last_token_pos in enumerate(last_token_pos_list):
            logits = batch_output.logits[j][last_token_pos - shared_prefix_length - 1]
            yes_logit, no_logit = logits[yes_id].item(), logits[no_id].item()
            yn_diff = yes_logit - no_logit
            batch_yn_diff.append(yn_diff)
        yn_diff_list += batch_yn_diff
    
    return yn_diff_list

if __name__ == '__main__':
    main()