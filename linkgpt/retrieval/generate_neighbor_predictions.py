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
from linkgpt.utils.prompts import get_prompts

def get_answer(model, tokenizer, prompt: str, graph_data, device: str, max_new_tokens: int=100) -> str:
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        generate_ids = model.generate(input_ids=inputs["input_ids"], max_new_tokens=max_new_tokens, graph_data=graph_data)
        answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    return answer


def get_diverse_answers(
    model,
    tokenizer,
    device,
    prompt,
    graph_data,
    max_new_tokens: int=50,
    num_beam_groups: int=5,
    num_beam_per_group: int=3,
    diversity_penalty: float=0.9,
    top_p: float=0.9,
    do_sample: bool=False,
):
    """
    Generate diverse answers (i.e., neighbor predictions) using diverse beam search.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    beam_outputs = model.generate(
        input_ids=inputs["input_ids"], 
        max_new_tokens=max_new_tokens, 
        num_beam_groups=num_beam_groups,
        num_beams=num_beam_groups * num_beam_per_group,
        num_return_sequences=num_beam_groups * num_beam_per_group,
        early_stopping=True,
        eos_token_id=tokenizer.convert_tokens_to_ids(tokenizer.tokenize('x\nx')[1]), # '\n'
        top_p=top_p,
        diversity_penalty=diversity_penalty,
        do_sample=do_sample,
    )
    answers = [tokenizer.decode(beam_output, skip_special_tokens=True) for beam_output in beam_outputs]
    answers = [ans.split('text: ')[-1].strip() for ans in answers]
    return answers

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--text_embedding_method', required=True)
    parser.add_argument('--text_embedding_folder_path', required=True)
    parser.add_argument('--dataset_for_lm_path', required=True)
    parser.add_argument('--ppr_data_path', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--ft_model_path', default=None)
    parser.add_argument('--stage', default=2, type=int)
    parser.add_argument('--max_hop', required=True, type=int)
    parser.add_argument('--device', default=None)
    parser.add_argument('--fp16', action='store_true')

    parser.add_argument('--max_context_neighbors', default=3, type=int)
    parser.add_argument('--max_new_tokens', default=150, type=int)
    parser.add_argument('--max_num', default=200, type=int)
    parser.add_argument('--np_ablate_node_encoding', action='store_true')
    parser.add_argument('--apply_get_diverse_answers', action='store_true')
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--diversity_penalty', default=0.9, type=float)
    parser.add_argument('--num_beam_groups', default=5, type=int)
    parser.add_argument('--num_beam_per_group', default=3, type=int)
    

    args = parser.parse_args()
    device = args.device if args.device is not None else basics.get_device()
    
    text_emb_list = []
    for i in range(args.max_hop + 1):
        if i == 0:
            text_emb_path = os.path.join(args.text_embedding_folder_path, f'text_emb_{args.text_embedding_method}.pt')
        else:
            text_emb_path = os.path.join(args.text_embedding_folder_path, f'text_emb_{args.text_embedding_method}_{i}hop.pt')
        text_emb = torch.load(text_emb_path, map_location=device)
        text_emb_list.append(text_emb)
    
    dataset_for_lm = basics.load_pickle(args.dataset_for_lm_path)
    ppr_data = torch.load(args.ppr_data_path).to(device)
    dgl_graph = tag_dataset_for_lm_to_dgl_graph(dataset_for_lm, include_valid=True).to(device)
    dgl_graph.ndata['feat'] = text_emb_list[0]
    nx_graph = basics.dgl_graph_to_nx_graph(dgl_graph.cpu())
    gnid2text = {i:dataset_for_lm[i][dataset_for_lm.text_field] for i in range(len(dataset_for_lm))}
    
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
    
    dataset_name = args.dataset_name
    prompts = get_prompts(dataset_name, 'np', allow_general_prompts=False)
    task_desc = prompts['task_desc']
    source_node_intro = prompts['source_node_intro']
    question = prompts['question']

    np_config = NPDatasetConfig(
        task_desc=task_desc,
        source_node_intro=source_node_intro,
        question=question,
        ablate_node_encoding=args.np_ablate_node_encoding,
        node_encoding_max_hop=args.max_hop,
        return_tokenized=False
    )

    src_list = dataset_for_lm.edge_split['test']['source_node']
    tgt_list = dataset_for_lm.edge_split['test']['target_node']
    
    max_num = args.max_num
    max_new_tokens = args.max_new_tokens
    max_context_neighbors = args.max_context_neighbors
    top_p = args.top_p
    diversity_penalty = args.diversity_penalty
    num_beam_groups = args.num_beam_groups
    num_beam_per_group = args.num_beam_per_group

    prediction_list = []
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for idx in trange(min(len(src_list), max_num)):
        src, tgt = src_list[idx], tgt_list[idx]
        all_neighbors = dgl_graph.successors(src).cpu().tolist()
        all_neighbor_text = [gnid2text[i] for i in all_neighbors]
        selected_neighbors = random.sample(all_neighbors, min(len(all_neighbors), max_context_neighbors))

        np_data = NPData(src_node=src, neighbors=selected_neighbors + [tgt])
        prompt, graph_data = np_data_to_prompt_and_graph_data(gnid2text, np_data, np_config)
        graph_data = {key: [graph_data[key]] for key in graph_data.keys()}
        prompt = 'text:'.join(prompt.split('text:')[:-1]) + 'text:'
        
        if not args.apply_get_diverse_answers:
            answer = get_answer(model, tokenizer, prompt, graph_data, device, max_new_tokens=max_new_tokens)
            text_list = [text[len('text: '):] for text in answer.split('Answer:\n')[-1].split('\n')]
            text_list = text_list[len(selected_neighbors):]
        else:
            text_list = get_diverse_answers(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                graph_data=graph_data,
                max_new_tokens=max_new_tokens,
                num_beam_groups=num_beam_groups,
                num_beam_per_group=num_beam_per_group,
                diversity_penalty=diversity_penalty,
                top_p=top_p,
                do_sample=False,
            )
        
        prediction_list.append(text_list)
        
        if idx % 100 == 0:
            basics.save_json(prediction_list, f'{output_dir}')

    
    basics.save_json(prediction_list, f'{output_dir}')
    
if __name__ == '__main__':
    main()