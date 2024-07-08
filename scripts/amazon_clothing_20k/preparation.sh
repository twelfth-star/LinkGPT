#!/bin/bash

DATASET_NAME=amazon_clothing_20k
LINKGPT_DATA_PATH=../../data # you can change this to any other path you like to store the data
PROJECT_PATH=../..

# Calculate the PPR scores
python ${PROJECT_PATH}/linkgpt/pairwise_encoding/calc_ppr_scores.py \
    --dataset_name ${DATASET_NAME} \
    --dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
    --ppr_data_save_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ppr_data.pt

# Train the CGTP model
python ${PROJECT_PATH}/linkgpt/text_graph_pretraining/graph_text_train.py \
	--dataset_name ${DATASET_NAME} \
	--dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
	--ckpt_save_path ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}/cgtp_model.pt \
	--num_epochs 3

# Create embeddings using the trained CGTP model
python ${PROJECT_PATH}/linkgpt/text_graph_pretraining/create_embedding.py \
	--dataset_name ${DATASET_NAME} \
	--dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
	--ckpt_save_path ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}/cgtp_model.pt \
	--embedding_save_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/text_emb_cgtp.pt \
