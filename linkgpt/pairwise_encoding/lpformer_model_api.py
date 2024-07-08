import os
import sys

import torch
from torch_sparse import SparseTensor

from .models.link_transformer import LinkTransformer

def get_lpformer_model(
    data,
    device,
    dim=256,
    num_heads=1,
    gnn_layers=2,
    pred_layers=2,
    dropout=0.2,
    gnn_drop=0.2,
    att_drop=0.1,
    pred_drop=0,
    feat_drop=0,
    residual=False,
    no_layer_norm=False,
    no_relu=False,
    count_ra=False,
    thresh_cn=1e-2,
    thresh_1hop=1e-2,
    thresh_non1hop=1,
    filter_cn=False,
    filter_1hop=False,
    trans_layers=1,
    ablate_counts=False,
    ablate_ppr_type=False,
    ablate_ppr=False,
    gcn_cache=True,
    layer_norm=False,
    relu=True,
    ablate_att=False,
    ablate_pe=False,
    ablate_feats=False,
):
    model_args = {
        'dim': dim,
        'num_heads': num_heads,
        'gnn_layers': gnn_layers,
        'pred_layers': pred_layers,
        'dropout': dropout,
        'gnn_drop': gnn_drop,
        'att_drop': att_drop,
        'pred_drop': pred_drop,
        'feat_drop': feat_drop,
        'residual': residual,
        'no_layer_norm': no_layer_norm,
        'no_relu': no_relu,
        'count_ra': count_ra,
        'thresh_cn': thresh_cn,
        'thresh_1hop': thresh_1hop,
        'thresh_non1hop': thresh_non1hop,
        'filter_cn': filter_cn,
        'filter_1hop': filter_1hop,
        'trans_layers': trans_layers,
        'ablate_counts': ablate_counts,
        'ablate_ppr_type': ablate_ppr_type,
        'ablate_ppr': ablate_ppr,
        'gcn_cache': gcn_cache,
        'layer_norm': layer_norm,
        'relu': relu,
        'ablate_att': ablate_att,
        'ablate_pe': ablate_pe,
        'ablate_feats': ablate_feats,
    }

    model = LinkTransformer(model_args, data, device=device).to(device)
    
    return model
