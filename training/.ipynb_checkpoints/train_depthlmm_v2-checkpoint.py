import os
import json
import glob
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer, set_seed

MODEL_PATH = "/mnt/dgx_lab/models/qwen2.5-vl-3b"
DATA_PATH = "/mnt/dgx_lab/datasets/DepthLMApproach"
OUTPUT_PATH = "/mnt/dgx_lab/models/DepthLm_final_model"
DS_CONFIG_PATH = "/mnt/dgx_lab/training/ds_zero3_bf16.json"

SUBDIRS = ["nu", "nyu", "sun"]

SEED = 42
NUM_EPOCHS = 3
LR = 5.2e-5

WARMUP_RATIO = 0.1
MAX_LENGTH = 2048

PER_DEVICE_BATCH = 2
GRAD_ACCUM = 6

SAVE_STEPS = 2000
LOGGING_STEPS = 50
DATALOADER_NUM_WORKERS = 32

SIZE_CACHE_PATH = os.path.join(OUTPUT_PATH, "image_sizes_cache.json")

def get_rank_world() -> Tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))

def list_jsonl_files(root: str) -> List[str]:
    files = []
    for sd in SUBDIRS:
        p = os.path.join(root, sd)
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "*.jsonl")))
    files.sort()
    return files

def load_samples(jsonl_files: List[str]) -> List[Tuple[str, str, str]]:
    out = []
    for fp in jsonl_files:
        with open(fp, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                img = s.get("image")
                prm = s.get("prompt")
                ans = s.get("answer")
                if not img or not prm or ans is None:
                    continue
                out.append((img, prm, ans))
    return out

def load_image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return w, h

def build_or_load_size_cache(samples: List[Tuple[str, str, str]]) -> Dict[str, Tuple[int, int]]:
    """Build or load image size cache with proper distributed synchronization."""
    import torch.distributed as dist
    
    rank, world = get_rank_world()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Temp file for atomic writes
    temp_cache_path = SIZE_CACHE_PATH + ".tmp"
    
    if rank == 0:
        # Only rank 0 builds the cache
        if not os.path.exists(SIZE_CACHE_PATH):
            print(f"[Rank 0] Building image size cache...", flush=True)
            uniq = sorted(list({s[0] for s in samples}))
            sizes = {}
            bad = 0
            for i, p in enumerate(uniq):
                if i % 10000 == 0:
                    print(f"[Rank 0] Processed {i}/{len(uniq)} images...", flush=True)
                try:
                    sizes[p] = load_image_size(p)
                except Exception as e:
                    bad += 1
            
            # Write to temp file first, then atomic rename
            with open(temp_cache_path, "w") as f:
                json.dump({k: [v[0], v[1]] for k, v in sizes.items()}, f)
            os.replace(temp_cache_path, SIZE_CACHE_PATH)
            
            if bad > 0:
                print(f"[Rank 0] bad_images_in_size_cache={bad}", flush=True)
            print(f"[Rank 0] Size cache complete: {len(sizes)} images", flush=True)
        else:
            print(f"[Rank 0] Size cache already exists", flush=True)
    
    # Synchronize all processes - wait for rank 0 to finish writing
    if world > 1 and dist.is_initialized():
        dist.barrier()
    
    # Now ALL ranks can safely read the complete file
    print(f"[Rank {rank}] Loading size cache...", flush=True)
    with open(SIZE_CACHE_PATH, "r") as f:
        raw = json.load(f)
    return {k: (int(v[0]), int(v[1])) for k, v in raw.items()}

class DepthLMDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, str]], processor: AutoProcessor, size_map: Dict[str, Tuple[int, int]]):
        self.samples = samples
        self.processor = processor
        self.size_map = size_map

    def __len__(self):
        return len(self.samples)

    def get_bucket(self, idx: int) -> Tuple[int, int]:
        img_path = self.samples[idx][0]
        w, h = self.size_map.get(img_path, (0, 0))
        return (h, w)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, prompt, answer = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        convo_full = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        convo_prompt = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]

        text_full = self.processor.apply_chat_template(convo_full, add_generation_prompt=False, tokenize=False)
        text_prompt = self.processor.apply_chat_template(convo_prompt, add_generation_prompt=True, tokenize=False)

        enc_full = self.processor(
        text=[text_full],
        images=[image],
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
        enc_prompt = self.processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=False,
        truncation=False,
)


        input_ids = enc_full["input_ids"][0]
        attention_mask = enc_full["attention_mask"][0]
        pixel_values = enc_full["pixel_values"][0]

        prompt_len = enc_prompt["input_ids"][0].shape[0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        h = int(pixel_values.shape[-2])
        w = int(pixel_values.shape[-1])

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pv_hw": torch.tensor([h, w], dtype=torch.int32)
        }

@dataclass
class DataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        hws = [tuple(f["pv_hw"].tolist()) for f in features]
        if len(set(hws)) != 1:
            raise RuntimeError(f"Batch contains mixed pixel_values sizes: {set(hws)}")

        pixel_values = torch.stack([f["pixel_values"] for f in features])

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        max_len = max(x.shape[0] for x in input_ids)

        def pad_1d(x: torch.Tensor, fill: int):
            if x.shape[0] == max_len:
                return x
            pad_len = max_len - x.shape[0]
            return torch.cat([x, torch.full((pad_len,), fill, dtype=x.dtype)])

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack([pad_1d(x, self.pad_token_id) for x in input_ids]),
            "attention_mask": torch.stack([pad_1d(x, 0) for x in attention_mask]),
            "labels": torch.stack([pad_1d(x, -100) for x in labels]),
        }

class DistributedExactSizeBucketBatchSampler:
    def __init__(self, dataset: DepthLMDataset, batch_size: int, seed: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.rank, self.world = get_rank_world()

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        buckets: Dict[Tuple[int, int], List[int]] = {}
        for i in range(len(self.dataset)):
            b = self.dataset.get_bucket(i)
            if b == (0, 0):
                continue
            buckets.setdefault(b, []).append(i)

        all_batches: List[List[int]] = []
        for _, idxs in buckets.items():
            g.shuffle(idxs)
            for k in range(0, len(idxs), self.batch_size):
                all_batches.append(idxs[k:k + self.batch_size])

        g.shuffle(all_batches)
        
        # Truncate to ensure equal batches per rank (prevents DDP hang)
        all_batches = all_batches[: len(all_batches) - (len(all_batches) % self.world)]
        
        my_batches = [all_batches[i] for i in range(len(all_batches)) if (i % self.world) == self.rank]

        self.epoch += 1
        for b in my_batches:
            yield b

    def __len__(self):
        buckets: Dict[Tuple[int, int], int] = {}
        for i in range(len(self.dataset)):
            b = self.dataset.get_bucket(i)
            if b == (0, 0):
                continue
            buckets[b] = buckets.get(b, 0) + 1

        total_batches = 0
        for _, n in buckets.items():
            total_batches += (n + self.batch_size - 1) // self.batch_size
        
        # Truncate to match __iter__ logic
        total_batches = total_batches - (total_batches % self.world)
        return total_batches // self.world

class BucketTrainer(Trainer):
    def __init__(self, *args, bucket_batch_size: int, bucket_seed: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_batch_size = bucket_batch_size
        self.bucket_seed = bucket_seed

    def get_train_dataloader(self):
        sampler = DistributedExactSizeBucketBatchSampler(self.train_dataset, self.bucket_batch_size, self.bucket_seed)
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=self.args.dataloader_num_workers > 0,
            collate_fn=self.data_collator
        )

def main():
    # Initialize distributed training FIRST (required for barrier synchronization)
    import torch.distributed as dist
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    jsonl_files = list_jsonl_files(DATA_PATH)
    if len(jsonl_files) == 0:
        raise RuntimeError("No JSONL files found")

    samples = load_samples(jsonl_files)
    if len(samples) == 0:
        raise RuntimeError("Loaded 0 samples")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    # Disable internal image resizing to preserve focal length
    # This is critical for testing the hypothesis that visual angle cues
    # can replace focal length normalization
    rank, _ = get_rank_world()
    if hasattr(processor, "image_processor"):
        ip = processor.image_processor
        if hasattr(ip, "do_resize"):
            ip.do_resize = False
            if rank == 0:
                print("✅ Disabled processor resize - focal length preserved", flush=True)
        if hasattr(ip, "do_center_crop"):
            ip.do_center_crop = False

    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(processor.tokenizer))
    model.config.use_cache = False

    size_map = build_or_load_size_cache(samples)
    dataset = DepthLMDataset(samples, processor, size_map)
    collator = DataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        ddp_find_unused_parameters=False,
        report_to="none",
        deepspeed=DS_CONFIG_PATH
    )

    trainer = BucketTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        bucket_batch_size=PER_DEVICE_BATCH,
        bucket_seed=SEED
    )

    trainer.train()
    trainer.save_model(OUTPUT_PATH)
    processor.save_pretrained(OUTPUT_PATH)


if __name__ == "__main__":
    main()
