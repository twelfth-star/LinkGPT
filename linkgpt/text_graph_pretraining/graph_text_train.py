import sys
import itertools
import os
import argparse

import transformers
from transformers import AutoConfig, BertTokenizer, BertModel
import torch
import pickle
from torch.utils import data
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)
from linkgpt.text_graph_pretraining.graph_text_dataset import CGTPDataset
from linkgpt.text_graph_pretraining.graph_text_model import CGTPModel
from linkgpt.utils import basics

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        center_input, neighbor_input, neighbor_mask, all_input, all_mask = batch
        for part in [center_input, neighbor_input, all_input]:
            for k in part.keys():
                part[k] = part[k].to(device)
        neighbor_mask = neighbor_mask.to(device)
        all_mask = all_mask.to(device)
        
        loss = model(center_input, neighbor_input, neighbor_mask, all_input, all_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_for_lm_path', required=True, type=str)
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--ckpt_save_path', required=True, type=str)
    
    parser.add_argument('-t', '--text_encoder_name', default='bert-base-uncased')
    parser.add_argument('-g', '--graph_encoder_name', default='graphformer')
    parser.add_argument('--num_neighbor', default=5, type=int)
    parser.add_argument('--max_text_length', default=64, type=int)
    parser.add_argument('--text_encoder_num_layers_to_train', default=2, type=int, help='only to train the last n layers of the text encoder (e.g., BERT)')
    parser.add_argument('--text_emb_dim', default=768, type=int, help='number of text embedding dimension ')
    parser.add_argument('--graph_embed_dim', default=768, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--head_lr', default=1e-3, type=float)
    parser.add_argument('--graph_encoder_lr', default=1e-4, type=float)
    parser.add_argument('--text_encoder_lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--lr_scheduler_patience', default=3, type=int)
    parser.add_argument('--lr_scheduler_factor', default=0.8, type=float)
    parser.add_argument('--num_epochs', default=5, type=int)
    
    args = parser.parse_args()

    with open(args.dataset_for_lm_path, 'rb') as f:
        dataset_for_lm = pickle.load(f)
    
    print(f'dataset_for_lm_path: {args.dataset_for_lm_path}')
    print(f'dataset_name: {args.dataset_name}')
    print(f'ckpt_save_path: {args.ckpt_save_path}')
    
    get_text = lambda x: x[dataset_for_lm.text_field] if dataset_for_lm.text_field in x.keys() else ""
    train_dataset = CGTPDataset(get_text, args.num_neighbor, dataset_for_lm, args.text_encoder_name, args.max_text_length)

    device = basics.get_device()
    cgtp_model = CGTPModel().to(device)
    total_params = sum(p.numel() for p in cgtp_model.parameters())
    trainable_params = sum(p.numel() for p in cgtp_model.parameters() if p.requires_grad)
    print(f'# total parameters: {total_params:,}\n# trainable parameters: {trainable_params:,}')
    
    params = [
        {"params": cgtp_model.graph_encoder.parameters(), "lr": args.graph_encoder_lr},
        {"params": cgtp_model.text_encoder.parameters(), "lr": args.text_encoder_lr},
        {"params": itertools.chain(
            cgtp_model.graph_proj.parameters(), cgtp_model.text_proj.parameters()
        ), "lr": args.head_lr, "weight_decay": args.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_scheduler_patience, factor=args.lr_scheduler_factor
    )
    
    for epoch in range(args.num_epochs):
        print(f"epoch: {epoch + 1}")
        cgtp_model.train()
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        step = 'epoch'
        train_epoch(cgtp_model, train_loader, optimizer, lr_scheduler, step, device)
    
    os.makedirs(os.path.dirname(args.ckpt_save_path), exist_ok=True) # make the directory if it does not exist
    torch.save(cgtp_model.state_dict(), args.ckpt_save_path)
    print(f"the model saved in {args.ckpt_save_path}")

if __name__ == '__main__':
    main()
