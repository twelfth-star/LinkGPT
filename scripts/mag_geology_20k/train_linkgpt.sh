#!/bin/bash

DATASET_NAME=mag_geology_20k
LINKGPT_DATA_PATH=../../data # you can change this to any other path you like to store the data
PROJECT_PATH=../..
WANDB_KEY=None # you can set this to your own wandb key
LINKGPT_MODEL_NAME=linkgpt-llama2-7b-cgtp

python ${PROJECT_PATH}/linkgpt/train/train.py \
	--model_name_or_path meta-llama/Llama-2-7b-hf \
	--text_embedding_method cgtp \
	--text_embedding_folder_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME} \
	--dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
	--ppr_data_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ppr_data.pt \
	--dataset_name ${DATASET_NAME} \
	--lp_dataset_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ft_yn_dataset.pkl \
	--np_dataset_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ft_np_dataset.pkl \
	--device_setting cuda:0 \
	--output_dir ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}/${LINKGPT_MODEL_NAME} \
	--max_hop 0 \
	--finetuning_type lora \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 4 \
	--lr_scheduler_type cosine \
	--logging_steps 1 \
	--learning_rate 3e-4 \
	--num_train_epochs_stage1 1 \
	--num_train_epochs_stage2 1 \
	--dataloader_num_workers 4 \
	--dataloader_prefetch_factor 8 \
	--fp16 \
	--lora_target q_proj,v_proj \
	--lora_alpha 32 \
	--lora_rank 16 \
	--lora_dropout 0.0 \
	--report_to wandb \
	--wandb_key ${WANDB_KEY} \
	--project_name ${DATASET_NAME}_ft \
	--run_name ${LINKGPT_MODEL_NAME} \
	--freeze_graph_related_modules_in_stage2 \
	--stage1_task np,lp \
	--stage2_task np,lp \
	--node_proj_num_layers 1 \
