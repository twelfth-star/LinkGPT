import argparse
import os
import sys
import pickle

import torch
from torch.utils import data
from tqdm import tqdm
import dgl

project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")) # path to LinkGPT
if project_path not in sys.path:
    sys.path.insert(0, project_path)
from linkgpt.dataset.tag_dataset_for_lm import TAGDatasetForLM, tag_dataset_for_lm_to_dgl_graph
from linkgpt.text_graph_pretraining.graph_text_dataset import CGTPDataset
from linkgpt.text_graph_pretraining.graph_text_model import CGTPModel
from linkgpt.utils import basics


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_for_lm_path', required=True, type=str, help='path of dataset_for_lm (input)')
    parser.add_argument('--embedding_save_path', required=True, type=str, help='path of the embedding (output)')
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--ckpt_save_path', required=True, type=str, help='path of the checkpoint of cgtp model')
    
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--max_text_length', default=64, type=int)
    parser.add_argument('--num_neighbor', default=5, type=int)
    parser.add_argument('-t', '--text_encoder_name', default='bert-base-uncased')
    parser.add_argument('--device', default=None)
    
    args = parser.parse_args()
        
    print(f'dataset_for_lm_path: {args.dataset_for_lm_path}')
    print(f'embedding_save_path: {args.embedding_save_path}')
    print(f'dataset_name: {args.dataset_name}')
    print(f'ckpt_save_path: {args.ckpt_save_path}')
    
    
    get_text = lambda x: x[dataset_for_lm.text_field] if dataset_for_lm.text_field in x.keys() else ""
    dataset_for_lm = basics.load_pickle(args.dataset_for_lm_path)
    cgtp_dataset = CGTPDataset(get_text, args.num_neighbor, dataset_for_lm, args.text_encoder_name, args.max_text_length)
    if args.device:
        device = args.device
    else:
        device = basics.get_device()
        
    cgtp_model = CGTPModel().to(device)
    state_dict = torch.load(args.ckpt_save_path, map_location=device)
    cgtp_model.load_state_dict(state_dict, strict=False)
    print('CGTP model loaded')
    
    cgtp_dataloader = data.DataLoader(cgtp_dataset, batch_size=args.batch_size, shuffle=False)
    
    cgtp_model.eval()
    all_node_emb = None
    for batch in tqdm(cgtp_dataloader):
        center_input, neighbor_input, neighbor_mask, _, _ = batch
        for part in [center_input, neighbor_input]:
            for k in part.keys():
                part[k] = part[k].to(device)
        neighbor_mask = neighbor_mask.to(device)
        with torch.no_grad():
            batch_node_emb = cgtp_model.graph_encoder(center_input, neighbor_input, neighbor_mask)
        if all_node_emb is None:
            all_node_emb = batch_node_emb
        else:
            all_node_emb = torch.concat([all_node_emb, batch_node_emb], dim=0)
            
    all_node_emb = all_node_emb.to('cpu')
    torch.save(all_node_emb, args.embedding_save_path)


if __name__ == '__main__':
    main()

