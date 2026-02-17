import os
import sys

# --- CRITICAL FIX: FORCE CACHE TO BIG DISK ---
# We set these BEFORE importing torch/transformers to prevent home directory crashes
os.environ["HF_HOME"] = "/mnt/dgx_lab/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

import json
import glob
import torch
import datasets 
from PIL import Image
from datasets import Dataset as HFDataset, Features, Value
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration, 
    set_seed
)
from trl import SFTConfig, SFTTrainer
from transformers.trainer_utils import get_last_checkpoint  # Added for Resume
# --- PATHS ---
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH = "/mnt/dgx_lab/datasets/DepthLMApproach"
OUTPUT_PATH = "/mnt/dgx_lab/models/DepthLm_Qwen3B_final_final"
DS_CONFIG_PATH = "/mnt/dgx_lab/training/ds_zero3_bf16.json"

# --- HARDWARE CONFIG ---
NUM_EPOCHS = 3
LR = 2.0e-5
PER_DEVICE_BATCH = 8       # Safe limit for 80GB A100 with High-Res
GRAD_ACCUM = 2              
NUM_WORKERS = 24  
PREFETCH_FACTOR = 2                
MAX_SEQ_LENGTH = 4096
ZERO2_CONFIG = {
    "fp16": {
        "enabled": False
    },
    "bf16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "gradient_accumulation_steps": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_clipping": 1.0,
}

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def load_hf_dataset():
    """Robust dataset loading with Schema Enforcement."""
    if get_rank() == 0:
        print("[*] Loading dataset samples...")
    
    subdirs = ["nu", "nyu", "sun"]
    samples = []
    
    for subdir in subdirs:
        files = glob.glob(os.path.join(DATA_PATH, subdir, "*.jsonl"))
        for fp in files:
            with open(fp, "r") as f:
                for line in f:
                    try:
                        s = json.loads(line)
                        if s.get("image"):
                            samples.append({
                                "image": str(s["image"]),
                                "prompt": str(s["prompt"]),
                                "answer": str(s["answer"]),
                            })
                    except:
                        continue
    
    # Explicit schema definition
    features = Features({
        "image": Value("string"),
        "prompt": Value("string"),
        "answer": Value("string"),
    })
    
    return HFDataset.from_list(samples, features=features)

def collate_fn_qwen(batch, processor):
    # 1. Image Opening 
    try:
        images = [Image.open(item["image"]).convert("RGB") for item in batch]
    except KeyError:
        print(f"[Rank {get_rank()}] CRITICAL: Batch item missing keys! Keys found: {batch[0].keys()}")
        raise
        
    # 2. Text Preparation
    texts = []
    for item in batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text", "text": item["prompt"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # 3. Processing
    batch_enc = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # 4. Label Masking
    labels = batch_enc["input_ids"].clone()
    
    # Mask padding
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
    # Search only for "<|im_start|>assistant"
    sep_text = "<|im_start|>assistant"
    sep_ids = processor.tokenizer(sep_text, add_special_tokens=False)["input_ids"]
    sep_len = len(sep_ids)

    input_ids = batch_enc["input_ids"]
    for i in range(len(input_ids)):
        input_row = input_ids[i].tolist()
        start_idx = -1
        
        for j in range(len(input_row) - sep_len):
            if input_row[j : j + sep_len] == sep_ids:
                start_idx = j + sep_len
                break
        
        if start_idx != -1:
            labels[i, :start_idx] = -100 
    
    batch_enc["labels"] = labels
    return batch_enc

def main():
    set_seed(42)
    
    # 1. Load Model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 2. Processor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 3. Load & Split Data
    full_dataset = load_hf_dataset()
    split_dataset = full_dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    if get_rank() == 0:
        print(f"[*] Data Loaded. Training: {len(train_dataset)} | Validation: {len(eval_dataset)}")

    # 4. Config
    sft_config = SFTConfig(
        output_dir=OUTPUT_PATH,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        max_length=MAX_SEQ_LENGTH, 
        
        # Validation & Saving
        eval_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        
        # Hardware
        dataloader_num_workers=NUM_WORKERS,
        dataloader_prefetch_factor=PREFETCH_FACTOR,
        dataloader_pin_memory=True,
        
        remove_unused_columns=False, 
        dataset_text_field="unused",
        
        # PASSING DICT DIRECTLY HERE
        deepspeed=ZERO2_CONFIG,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda b: collate_fn_qwen(b, processor),
    )

    if get_rank() == 0:
        print(f"[*] Starting 3B Training on 8x A100 (ZeRO-2).")
        print(f"[*] Batch Size: {PER_DEVICE_BATCH} (Accum {GRAD_ACCUM}) | Workers: {NUM_WORKERS}")

    # --- RESUME LOGIC ---
    last_checkpoint = None
    if os.path.isdir(OUTPUT_PATH):
        from transformers.trainer_utils import get_last_checkpoint  # Added for Resume
        last_checkpoint = get_last_checkpoint(OUTPUT_PATH)
    
    if last_checkpoint is not None:
        if get_rank() == 0:
            print(f"[*] Found checkpoint: {last_checkpoint}. Resuming training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        if get_rank() == 0:
            print("[*] No checkpoint found. Starting from scratch.")
        trainer.train()
        
    trainer.save_model(OUTPUT_PATH)
    processor.save_pretrained(OUTPUT_PATH)

if __name__ == "__main__":
    main()