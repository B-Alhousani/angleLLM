#!/bin/bash

model_path=$1
output_path=$2

# 1. we use ';' to separate the image_folder, dataset_name and sample_weights
# 2. please follow the optimal hyper-paramters from the paper
# 3. max_steps removed to train based on epochs

# OPTIMIZED FOR 8x 10GB GPUs & 80 CPU CORES
torchrun --nproc_per_node=8 --master_port=12433 train_angle_final.py \
--model_name_or_path $model_path \
--image_folder "/mnt/dgx_lab/datasets/depthLMoriginal/sun/output_image_folder;/mnt/dgx_lab/datasets/depthLMoriginal/nu/processed_images_mini;/mnt/dgx_lab/datasets/depthLMoriginal/nyu/output_image_folder" \
--dataset_name "/mnt/dgx_lab/datasets/depthLMoriginal/sun/output_jsonl/sunRGBD.jsonl;/mnt/dgx_lab/datasets/depthLMoriginal/nu/nuscenes_mini.jsonl;/mnt/dgx_lab/datasets/depthLMoriginal/nyu/output_jsonl/NYUv2.jsonl" \
--sample_weights "1;1;1" \
--max_seq_length 4096 \
--learning_rate 1e-5 \
--lr_scheduler_type cosine \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 3 \
--dataloader_num_workers 8 \
--warmup_ratio 0.1 \
--max_grad_norm 0.1 \
--logging_steps 1 \
--report_to tensorboard \
--gradient_checkpointing false \
--attn_implementation "flash_attention_2" \
--log_level info \
--logging_strategy steps \
--output_dir $output_path \
--save_strategy "epoch" \
--eval_strategy no \
--torch_dtype bfloat16 \
--seed 42 \
--dataset_class dataset_train \
--remove_unused_columns false \
--num_train_epochs 1 \
--fsdp "full_shard auto_wrap" \
--fsdp_config '{"activation_checkpointing": true, "transformer_layer_cls_to_wrap": "Qwen2_5_VLDecoderLayer", "mixed_precision": {"param_dtype": "bfloat16", "reduce_dtype": "bfloat16", "buffer_dtype": "bfloat16"}}'