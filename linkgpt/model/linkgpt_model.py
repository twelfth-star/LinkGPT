from typing import List, Optional, Tuple, Union
import pickle
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import dgl
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.integrations import is_deepspeed_zero3_enabled
from peft import PeftConfig, PeftModel

import llmtuner
from llmtuner.model.patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
from llmtuner.model.loader import init_adapter

from ..dataset.linkgpt_dataset import NODE_START_TOKEN, NODE_TOKEN, PAIRWISE_START_TOKEN, \
    PAIRWISE_TOKEN, LINKGPT_SPECIAL_TOKENS, IGNORE_INDEX
from ..dataset.tag_dataset_for_lm import TAGDatasetForLM
from ..pairwise_encoding.models.link_transformer import LinkTransformer
from ..pairwise_encoding.lpformer_model_api import get_lpformer_model
from ..utils import basics

class LinkGPTConfig(LlamaConfig):
    model_type = "LinkGPT"

# adapted from GraphGPT
class LinkGPTModel(LlamaModel):
    config_class = LinkGPTConfig
    
    def __init__(self, config: LlamaConfig):
        super(LinkGPTModel, self).__init__(config)
        self.node_encoding_max_hop = None
        self.node_encoding_dim = None
        self.node_encoding_list = None
        self.node_alignment_proj = None
        
        self.pairwise_encoder = None
        self.pairwise_encoding_dim = None
        self.pairwise_alignment_proj = None
        
        self.linkgpt_special_token_id_ls = None
        self.linkgpt_special_token_id_to_idx = None
        self.linkgpt_special_token_emb = None
        
    def set_node_encoder(self, node_encoding_list: List[torch.Tensor], num_layers=1):
        self.node_encoding_max_hop = len(node_encoding_list) - 1
        self.node_encoding_dim = node_encoding_list[0].shape[1]
        self.node_encoding_list = node_encoding_list
        if num_layers == 1:
            self.node_alignment_proj = nn.Linear(self.node_encoding_dim, self.config.hidden_size).to(self.device)
        elif num_layers == 2:
            print('Using 2-layer MLP as node projector')
            self.node_alignment_proj = nn.Sequential(
                nn.Linear(self.node_encoding_dim, 768),
                nn.ReLU(),
                nn.Linear(768, self.config.hidden_size)
            ).to(self.device)
        elif num_layers == 3:
            print('Using 3-layer MLP as node projector')
            self.node_alignment_proj = nn.Sequential(
                nn.Linear(self.node_encoding_dim, 640),
                nn.ReLU(),
                nn.Linear(640, 768),
                nn.ReLU(),
                nn.Linear(768, self.config.hidden_size)
            ).to(self.device)
        else:
            raise ValueError('num_layers must be 1, 2, or 3')
    
    def set_pairwise_encoder(self, pairwise_encoder: LinkTransformer):
        self.pairwise_encoder = pairwise_encoder.to(self.device)
        self.pairwise_encoding_dim = pairwise_encoder.out_dim - pairwise_encoder.dim 
        # the encoding of node i and j produced by LPFormer is actually the concatenation of (h_i * h_j) and pairwise encoding
        # so output dim = node encoding dim + pairwise encoding dim
        self.pairwise_alignment_proj = nn.Linear(self.pairwise_encoding_dim, self.config.hidden_size).to(self.device)
    
    def set_linkgpt_special_token_emb(self, linkgpt_special_token_id_ls: List[int], linkgpt_special_token_ls: List[str]):
        self.linkgpt_special_token_id_ls = linkgpt_special_token_id_ls
        self.linkgpt_special_token_to_id = {token: idx for token, idx in zip(linkgpt_special_token_ls, linkgpt_special_token_id_ls)}
    
    def is_valid(self):
        """
        Ensure set_node_encoder, set_pairwise_encoder, and set_pairwise_encoder are called before calling `forward()`
        """
        return self.node_encoding_list is not None and \
            self.pairwise_encoder is not None and \
                self.linkgpt_special_token_id_ls is not None
    
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        past_key_values: Optional[List[torch.FloatTensor]]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        use_cache: Optional[bool]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool] = None,
        graph_data: dict=None,
    ):
        """
        suppose the batch_size is 2, an example of graph_data is
        graph_data = {
            'source_node': [0, 100],
            'node_id_ls': [
                [(0, 0), (4, 0), (5, 0), (7, 0)], 
                [(100, 0), (156, 0), (172, 0), (180, 0), (190, 0)]
                # Each pair represents (node_id, hop_num). In this paper the maximum hop_num is 0, so all hop_num is 0.
            ], 
            'pairwise_target_id_ls': [
                [4, 5, 7],
                [156, 172, 180, 190]
            ],
        }
        node_id_ls (and pairwise_target_id_ls) appear in the order of the corresponding spectial tokens in the prompt (input_ids).
        Refer to the prompt templates for details.
        """
        assert self.is_valid(), "LinkGPTModel is not valid now. Call set_node_encoder, set_pairwise_encoder, and set_linkgpt_special_token_emb before calling forward()."
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        if graph_data is not None:
            # need to change the embeddings of the special tokens related to graph
            new_input_embeds = []
            for batch_idx, cur_input_ids, cur_input_embeds in zip(range(inputs_embeds.shape[0]), input_ids, inputs_embeds):
                if len(graph_data['source_node']) == inputs_embeds.shape[0]:
                    # single GPU
                    graph_batch_index = batch_idx
                elif len(graph_data['source_node']) % inputs_embeds.shape[0] == 0:
                    # multi-GPU
                    # rank = input_ids.device.index
                    # graph_batch_index = rank * inputs_embeds.shape[0] + batch_idx
                    raise NotImplementedError("Multi-GPU training is not supported yet")
                else:
                    raise ValueError("Length of graph data must be a multiple of the batch size of input_ids")
                
                source_node = graph_data['source_node'][graph_batch_index]
                node_id_ls = graph_data['node_id_ls'][graph_batch_index]
                pairwise_target_id_ls = graph_data['pairwise_target_id_ls'][graph_batch_index]
                
                if len(node_id_ls) > 0:
                    node_encoding_list = []
                    for node_id, hop_num in node_id_ls:
                        node_encoding = self.node_encoding_list[hop_num][node_id]
                        node_encoding_list.append(node_encoding)
                    node_encodings = torch.stack(node_encoding_list, dim=0).to(self.device)
                    node_encodings = self.node_alignment_proj(node_encodings)
                else:
                    # no <node> token
                    node_encodings = None
                
                if len(pairwise_target_id_ls) > 0:
                    if self.pairwise_encoder.device != self.device:
                        self.pairwise_encoder.to(self.device)
                    pairwise_encodings = self.pairwise_encoder.get_pairwise_encoding(
                        torch.tensor([[source_node, tgt] for tgt in pairwise_target_id_ls]).t()
                    ).to(self.device)
                    pairwise_encodings = self.pairwise_alignment_proj(pairwise_encodings)
                else:
                    # no <pairwise> token
                    pairwise_encodings = None

                cur_new_input_embeds = cur_input_embeds.clone()
                
                for special_token in [NODE_TOKEN, PAIRWISE_TOKEN]:
                    if special_token == NODE_TOKEN:
                        encodings = node_encodings
                    elif special_token == PAIRWISE_TOKEN:
                        encodings = pairwise_encodings
                    if encodings is None:
                        # no such special token in the input_ids
                        continue
                    special_token_id = self.linkgpt_special_token_to_id[special_token]
                    special_token_pos_ls = torch.where(cur_input_ids == special_token_id)[0]
                    for token_pos, encodings_item in zip(special_token_pos_ls, encodings):
                        cur_new_input_embeds[token_pos] = encodings_item
                new_input_embeds.append(cur_new_input_embeds)
                
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
                
        return super(LinkGPTModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

# adapted from GraphGPT
class LinkGPTForCausalLM(LlamaForCausalLM):
    config_class = LinkGPTConfig
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LinkGPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        graph_data: Optional[dict]=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            graph_data = graph_data,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is None:
            shift_labels = input_ids[..., 1:].contiguous()
        else:
            shift_labels = labels.contiguous() # `labels` passed in is already shifted
        
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_data": kwargs.get("graph_data", None),
            }
        )
        return model_inputs
    
def unfreeze_graph_related_modules(model_with_lora):
    """
    Unfreeze graph related modules in the model_with_lora, including node_alignment_proj, pairwise_alignment_proj, and pairwise_encoder
    """
    graph_related_modules = [
        model_with_lora.base_model.model.model.node_alignment_proj,
        model_with_lora.base_model.model.model.pairwise_alignment_proj,
        model_with_lora.base_model.model.model.pairwise_encoder
    ]
    for modules in graph_related_modules:
        for param in modules.parameters():
            param.requires_grad = True

    linkgpt_special_token_id_ls = model_with_lora.base_model.model.model.linkgpt_special_token_id_ls
    model_with_lora.base_model.model.model.embed_tokens.weight[linkgpt_special_token_id_ls].requires_grad = True

def unfreeze_lora_adapter(model_with_lora):
    """
    Unfreeze the LoRA adapter in the model_with_lora
    """
    for name, param in model_with_lora.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

def freeze_all_parameters(model_with_lora):
    """
    Freeze all parameters in the model_with_lora
    """
    for param in model_with_lora.parameters():
        param.requires_grad = False
        
def save_lora_model(model_with_lora, output_dir):
    """
    Save the model_with_lora to the output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    lora_path = os.path.join(output_dir, 'lora_model')
    os.makedirs(lora_path, exist_ok=True)
    model_with_lora.save_pretrained(lora_path, save_embedding_layers=False)
    
    node_alignment_proj_path = os.path.join(output_dir, 'node_alignment_proj.pt')
    torch.save(model_with_lora.base_model.model.model.node_alignment_proj.state_dict(), node_alignment_proj_path)
    
    pairwise_encoder_path = os.path.join(output_dir, 'pairwise_encoder.pt')
    torch.save(model_with_lora.base_model.model.model.pairwise_encoder.state_dict(), pairwise_encoder_path)
    
    pairwise_alignment_proj_path = os.path.join(output_dir, 'pairwise_alignment_proj.pt')
    torch.save(model_with_lora.base_model.model.model.pairwise_alignment_proj.state_dict(), pairwise_alignment_proj_path)
    
    linkgpt_special_token_id_ls = model_with_lora.base_model.model.model.linkgpt_special_token_id_ls
    linkgpt_special_token_emb_path = os.path.join(output_dir, 'linkgpt_special_token_emb.pt')
    torch.save(model_with_lora.base_model.model.model.embed_tokens.weight[linkgpt_special_token_id_ls], linkgpt_special_token_emb_path)

def get_tokenizer(model_name_or_path: str='meta-llama/Llama-2-7b-hf'):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    num_added_token = tokenizer.add_special_tokens({
        'additional_special_tokens': LINKGPT_SPECIAL_TOKENS
    })
    return tokenizer

def get_model_and_tokenizer(
    model_args,
    finetuning_args,
    node_encoding_list: List[torch.Tensor],
    lpformer_model,
    is_trainable=False,
    add_valuehead=False,
    device='cpu',
    apply_lora=True,
    node_proj_num_layers=1
):
    """
    Create pre-trained model and tokenizer for fine-tuning
    """
    tokenizer = get_tokenizer(model_args.model_name_or_path)
    
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    special_token_id_ls = [tokenizer.vocab[token] for token in LINKGPT_SPECIAL_TOKENS]
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    patch_config(config, tokenizer, model_args, config_kwargs, is_trainable, add_valuehead=False)
    
    config_kwargs.pop("config", None)
    config_kwargs.pop("torch_dtype", None)
    config_kwargs.pop("low_cpu_mem_usage", None)
    config_kwargs.pop("device_map", None)
    if str(device).startswith('cuda'):
        model = LinkGPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=model_args.compute_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map='auto',
            **config_kwargs,
        )
    else:
        model = LinkGPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            torch_dtype=model_args.compute_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs,
        )
    patch_model(model, tokenizer, model_args, is_trainable, add_valuehead=False)
    
    model.model.set_node_encoder(node_encoding_list, num_layers=node_proj_num_layers)
    model.model.set_pairwise_encoder(lpformer_model)
    model.model.set_linkgpt_special_token_emb(special_token_id_ls, LINKGPT_SPECIAL_TOKENS)
    
    if apply_lora:
        model = init_adapter(config, model, model_args, finetuning_args, is_trainable)
    
    return model, tokenizer


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str,
    node_encoding_list: List[torch.Tensor],
    lpformer_model,
    ft_model_path: int,
    stage: int,
    compute_dtype=torch.float16,
):
    tokenizer = get_tokenizer(model_name_or_path)
    special_token_id_ls = [tokenizer.vocab[token] for token in LINKGPT_SPECIAL_TOKENS]
    
    if device.startswith('cuda'):
        model = LinkGPTForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map='auto',
        )
    else:
        model = LinkGPTForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            device_map='cpu',
        )
    
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    model.model.set_node_encoder(node_encoding_list)
    model.model.set_pairwise_encoder(lpformer_model)
    model.model.set_linkgpt_special_token_emb(special_token_id_ls, LINKGPT_SPECIAL_TOKENS)
    
    lora_model_path = os.path.join(ft_model_path, f'stage{stage}/lora_model')
    
    if device.startswith('cuda'):
        ft_model = PeftModel.from_pretrained(model, lora_model_path, device_map='auto', is_trainable=True)
    else:
        ft_model = PeftModel.from_pretrained(model, lora_model_path, device_map='cpu', is_trainable=True)
    
    # load node alignment projector
    node_alignment_proj_path = os.path.join(ft_model_path, f'stage{stage}/node_alignment_proj.pt')
    node_alignment_proj_dict = torch.load(node_alignment_proj_path, map_location=device)
    linear_layer_count = sum(1 for key in node_alignment_proj_dict.keys() if 'weight' in key and len(node_alignment_proj_dict[key].shape) == 2)
    model.model.set_node_encoder(node_encoding_list, num_layers=linear_layer_count)
    ft_model.base_model.model.model.node_alignment_proj.load_state_dict(node_alignment_proj_dict)
    
    # load pairwise alignment projector
    pairwise_alignment_proj_path = os.path.join(ft_model_path, f'stage{stage}/pairwise_alignment_proj.pt')
    pairwise_alignment_proj_dict = torch.load(pairwise_alignment_proj_path, map_location=device)
    ft_model.base_model.model.model.pairwise_alignment_proj.load_state_dict(pairwise_alignment_proj_dict)
    
    # load pairwise encoder
    pairwise_encoder_path = os.path.join(ft_model_path, f'stage{stage}/pairwise_encoder.pt')
    pairwise_encoder_dict = torch.load(pairwise_encoder_path, map_location=device)
    ft_model.base_model.model.model.pairwise_encoder.load_state_dict(pairwise_encoder_dict)
    
    # load LinkGPT special token embbeddings
    linkgpt_special_token_emb_path = os.path.join(ft_model_path, f'stage{stage}/linkgpt_special_token_emb.pt')
    linkgpt_special_token_emb = torch.load(linkgpt_special_token_emb_path, map_location=device).to(device)
    linkgpt_special_token_id_ls = ft_model.base_model.model.model.linkgpt_special_token_id_ls
    ft_model.base_model.model.model.embed_tokens.weight[linkgpt_special_token_id_ls] = linkgpt_special_token_emb
    
    return ft_model, tokenizer