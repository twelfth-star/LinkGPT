#!/bin/bash

DATASET_NAME=amazon_sports_20k
LINKGPT_DATA_PATH=../../data # you can change this to any other path you like to store the data
PROJECT_PATH=../..
WANDB_KEY=None # you can set this to your own wandb key
NUM_OF_EXAMPLES=4 # you can change this to 0, 2, 4
LINKGPT_MODEL_NAME=linkgpt-llama2-7b-cgtp

python ${PROJECT_PATH}/linkgpt/eval/eval_yn.py \
	--model_name_or_path meta-llama/Llama-2-7b-hf \
	--text_embedding_method cgtp \
	--text_embedding_folder_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ \
	--dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
	--eval_dataset_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/eval_yn_dataset_${NUM_OF_EXAMPLES}_examples.pkl \
	--ppr_data_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ppr_data.pt \
	--dataset_name ${DATASET_NAME} \
	--ft_model_path ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}/${LINKGPT_MODEL_NAME} \
	--stage 2 \
	--device cuda \
	--output_path ${LINKGPT_DATA_PATH}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/eval_yn_dataset_${NUM_OF_EXAMPLES}_examples_output.json \
	--max_hop 0 \
	--fp16 \
	--report_to wandb \
	--wandb_key ${WANDB_KEY} \
	--wandb_project_name ${DATASET_NAME}_eval \
	--wandb_run_name ${LINKGPT_MODEL_NAME}-${NUM_OF_EXAMPLES}-examples \
