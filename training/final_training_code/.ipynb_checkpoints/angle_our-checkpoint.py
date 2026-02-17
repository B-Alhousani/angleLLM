import os
import sys
import json
import glob
import torch
import numpy
import _codecs
# --- 1. THE CRITICAL FIX (Insert this at the very top) ---
# This monkey-patches torch.load to disable the "Weights Only" security check.
# Without this, you cannot RESUME from a checkpoint later.
_original_torch_load = torch.load
def _unsafe_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _unsafe_torch_load
# Force communication over PCIe (slower, but bypasses the broken NVLink)
os.environ["NCCL_P2P_DISABLE"] = "1"
# ---------------------------------------------------------
from PIL import Image

# Environment Setup
os.environ["HF_HOME"] = "/mnt/dgx_lab/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

from datasets import Dataset as HFDataset, Features, Value
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration, 
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

# --- CONFIGURATION ---
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH = "/mnt/dgx_lab/datasets/AngleLmDiscriptive"
OUTPUT_PATH = "/mnt/dgx_lab/models/AndgeLM_Qwen3B_Discriptive_final"

NUM_EPOCHS = 3
LR = 2.0e-5
PER_DEVICE_BATCH = 8       
GRAD_ACCUM = 2              
NUM_WORKERS = 24  
PREFETCH_FACTOR = 2                
MAX_SEQ_LENGTH = 4096

# DeepSpeed ZeRO-2 Configuration
ZERO2_CONFIG = {
    "fp16": {"enabled": False},
    "bf16": {"enabled": True},
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
        # Fixed path joining for safety
        search_path = os.path.join(DATA_PATH, subdir, "*.jsonl")
        files = glob.glob(search_path)
        for fp in files:
            with open(fp, "r") as f:
                for line in f:
                    try:
                        s = json.loads(line)
                        # Ensure ALL keys exist before adding to samples
                        if all(k in s for k in ["image", "prompt", "label"]):
                            samples.append({
                                "image": str(s["image"]),
                                "prompt": str(s["prompt"]),
                                "answer": str(s["label"]),
                            })
                    except Exception:
                        continue
    
    # 1. Check if samples is empty to avoid the "target {}" error
    if not samples:
        raise ValueError(f"No valid samples found in {DATA_PATH}. Check your JSONL files and paths.")

    features = Features({
        "image": Value("string"),
        "prompt": Value("string"),
        "answer": Value("string"),
    })
    
    # 2. Convert to dict first then to HF Dataset (this is often more stable)
    # This bypasses the reorder_fields_as mismatch in some versions
    data_dict = {
        "image": [x["image"] for x in samples],
        "prompt": [x["prompt"] for x in samples],
        "answer": [x["answer"] for x in samples],
    }
    
    return HFDataset.from_dict(data_dict, features=features)
def collate_fn_qwen(batch, processor):
    """Custom collator for Qwen2.5-VL to handle image-text pairs and label masking."""
    try:
        images = [Image.open(item["image"]).convert("RGB") for item in batch]
    except KeyError:
        print(f"[Rank {get_rank()}] CRITICAL: Batch item missing keys!")
        raise
        
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

    batch_enc = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # Label Masking: Only calculate loss on the Assistant's response
    labels = batch_enc["input_ids"].clone()
    
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
    sep_text = "<|im_start|>assistant"
    sep_ids = processor.tokenizer(sep_text, add_special_tokens=False)["input_ids"]
    sep_len = len(sep_ids)

    input_ids = batch_enc["input_ids"]
    for i in range(len(input_ids)):
        input_row = input_ids[i].tolist()
        start_idx = -1
        
        # Find where the assistant response starts
        for j in range(len(input_row) - sep_len):
            if input_row[j : j + sep_len] == sep_ids:
                start_idx = j + sep_len
                break
        
        if start_idx != -1:
            # Mask everything before the assistant starts
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

    # 4. SFT Config
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
        
        # Hardware & Efficiency
        dataloader_num_workers=NUM_WORKERS,
        dataloader_prefetch_factor=PREFETCH_FACTOR,
        dataloader_pin_memory=True,
        remove_unused_columns=False, 
        dataset_text_field="unused",
        
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
        print(f"[*] Starting 3B Training on Multi-GPU (ZeRO-2).")
        print(f"[*] Batch Size: {PER_DEVICE_BATCH} (Accum {GRAD_ACCUM}) | Workers: {NUM_WORKERS}")

    # --- RESUME LOGIC ---
    last_checkpoint = None
    if os.path.isdir(OUTPUT_PATH):
        print("LOADING CHECKPOINT..")
        from transformers.trainer_utils import get_last_checkpoint
        last_checkpoint = get_last_checkpoint(OUTPUT_PATH)
    
    if last_checkpoint is not None:
        if get_rank() == 0:
            print(f"[*] Found checkpoint: {last_checkpoint}. Resuming training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        if get_rank() == 0:
            print("[*] No checkpoint found. Starting from scratch.")
        trainer.train()
        
    # Save final results
    trainer.save_model(OUTPUT_PATH)
    processor.save_pretrained(OUTPUT_PATH)

if __name__ == "__main__":
    main()