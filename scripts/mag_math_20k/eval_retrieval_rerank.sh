#!/bin/bash

DATASET_NAME=mag_math_20k
LINKGPT_DATA_PATH=../../data # you can change this to any other path you like to store the data
PROJECT_PATH=../..
LINKGPT_MODEL_NAME=linkgpt-llama2-7b-cgtp
WANDB_KEY=None # you can set this to your own wandb key

# evaluate the pipeline without retrieval
python ${PROJECT_PATH}/linkgpt/eval/eval_yn.py \
	--model_name_or_path meta-llama/Llama-2-7b-hf \
	--text_embedding_method cgtp \
	--text_embedding_folder_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ \
	--dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
	--eval_dataset_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/eval_yn_dataset_large_candidate_set.pkl \
	--ppr_data_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ppr_data.pt \
	--dataset_name ${DATASET_NAME} \
	--ft_model_path ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}/${LINKGPT_MODEL_NAME} \
	--stage 2 \
	--device cuda \
	--output_path ${LINKGPT_DATA_PATH}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/eval_yn_dataset_large_candidate_set_output.json \
	--max_hop 0 \
	--fp16 \
	--report_to wandb \
	--wandb_key ${WANDB_KEY} \
	--wandb_project_name ${DATASET_NAME}_eval \
	--wandb_run_name ${LINKGPT_MODEL_NAME}-large-candidate-set \

# generate neighbor predictions
python ${PROJECT_PATH}/linkgpt/retrieval/generate_neighbor_predictions.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --text_embedding_method cgtp \
    --text_embedding_folder_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME} \
    --dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
    --ppr_data_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ppr_data.pt \
    --dataset_name ${DATASET_NAME} \
    --output_dir ${DATASET_NAME}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/predicted_neighbors.json \
    --ft_model_path ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}/${LINKGPT_MODEL_NAME} \
    --stage 2 \
    --max_hop 0 \
    --device cuda \
    --fp16 \
    --max_context_neighbors 0 \
    --max_new_tokens 50 \
    --max_num 200 \
    --apply_get_diverse_answers \
    --top_p 0.9 \
    --diversity_penalty 0.9 \
    --num_beam_groups 5 \
    --num_beam_per_group 3

# evaluate the retrieval rerank pipeline
python ${PROJECT_PATH}/linkgpt/retrieval/eval_retrieval_rerank.py \
    --prediction_list_path ${DATASET_NAME}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/predicted_neighbors.json \
    --dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
    --eval_dataset_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/eval_yn_dataset_large_candidate_set.pkl \
    --eval_output_path ${DATASET_NAME}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/eval_yn_dataset_large_candidate_set_output.json \
    --result_saving_path ${DATASET_NAME}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/retrieval_rerank_results.txt \
    --dataset_name ${DATASET_NAME} \
    --num_neg_tgt 1800 \
    --num_to_retrieve 60 \
    --apply_dist_based_grouping \
    --max_dist 2 \
    --beta 0.65