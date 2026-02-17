"""
Evaluation script for DepthLM Qwen2.5-VL-3B model with CSV Reporting.
Features:
 - Batch Processing (Significantly faster)
 - Resume Capability (Skips already processed images)
 - Robust Error Handling (Saves progress immediately)
 - Left-Padding Fix (Critical for batch accuracy) 
"""

import argparse
import json
import re
import csv
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ============================================================================
# METRIC FUNCTIONS
# ============================================================================
def extract_depth_from_output(text: str) -> float | None:
    """Parse model output to extract depth value from <answer> tags."""
    # Priority 1: <answer>X</answer>
    m = re.search(r"<answer>\s*([\d.]+)\s*</answer>", text, re.IGNORECASE)
    if m:
        try: return float(m.group(1))
        except: pass
    
    # Priority 2: "X meters"
    m = re.search(r"([\d.]+)\s*meters?", text, re.IGNORECASE)
    if m:
        try: return float(m.group(1))
        except: pass

    # Fallback: First number found
    m = re.search(r"(\d+\.?\d*)", text)
    if m:
        try: return float(m.group(1))
        except: pass
    
    return None

def compute_delta1(pred: float, gt: float) -> float:
    """δ₁: 1 if max(pred/gt, gt/pred) < 1.25, else 0."""
    if pred <= 0 or gt <= 0:
        return 0.0
    ratio = max(pred / gt, gt / pred)
    return 1.0 if ratio < 1.25 else 0.0

# ============================================================================
# DATA UTILS
# ============================================================================
def load_processed_images(csv_path):
    """Returns a set of image paths already in the CSV to allow resuming."""
    if not os.path.exists(csv_path):
        return set()
    processed = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("image"):
                    processed.add(row["image"])
    except Exception:
        pass
    return processed

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def calculate_global_accuracy(csv_path):
    """Reads the entire CSV to calculate total accuracy across all runs."""
    if not os.path.exists(csv_path):
        return 0.0, 0
    
    total_valid = 0
    total_correct = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Ensure we are reading a valid row with a score
                if row.get("delta1_score") and row["delta1_score"] != "":
                    try:
                        score = float(row["delta1_score"])
                        total_valid += 1
                        if score == 1.0:
                            total_correct += 1
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Could not calculate global accuracy: {e}")
        return 0.0, 0
        
    acc = (total_correct / total_valid * 100) if total_valid > 0 else 0.0
    return acc, total_valid

# ============================================================================
# MAIN EVALUATION
# ============================================================================
def main(args):
    print("="*60)
    print(f"DepthLM Evaluation | Batch Size: {args.batch_size}")
    print(f"Model: {args.model_path}")
    print("="*60)
    
    # 1. Load Processor & Fix Padding
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # --- CRITICAL FIX: Set padding to left for Decoder-only batching ---
    processor.tokenizer.padding_side = "left"
    # -------------------------------------------------------------------

    # 2. Load Model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    # 3. Load Dataset
    samples = []
    try:
        with open(args.eval_jsonl, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))
    except Exception as e:
        print(f"Error loading JSONL: {e}")
        return

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    # 4. Check for Resume
    processed_images = load_processed_images(args.output_csv)
    if processed_images:
        print(f"Found {len(processed_images)} already processed samples. Skipping them.")
    
    samples_to_run = [s for s in samples if s["image"] not in processed_images]
    print(f"Starting evaluation on {len(samples_to_run)} samples...")

    # 5. Prepare CSV
    fieldnames = ["image", "ground_truth", "prediction", "is_correct", "delta1_score", "raw_output"]
    file_exists = os.path.isfile(args.output_csv)

    # Open CSV in append mode
    with open(args.output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        total_correct = 0
        total_valid = 0
        
        # 6. Batch Evaluation Loop
        pbar = tqdm(total=len(samples_to_run), unit="img")
        
        for batch in chunks(samples_to_run, args.batch_size):
            
            # --- Prepare Batch ---
            images = []
            texts = []
            valid_indices = [] # Map local batch index to original sample

            for i, sample in enumerate(batch):
                try:
                    img_path = sample["image"]
                    prompt = sample.get("prompt", "Estimate the depth.")
                    
                    # Ensure image loads
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt},
                        ],
                    }]
                    
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    texts.append(text)
                    valid_indices.append(i)
                    
                except Exception as e:
                    print(f"\n[Load Error] {sample.get('image')}: {e}")
                    # Log error to CSV
                    writer.writerow({
                        "image": sample.get("image"),
                        "ground_truth": sample.get("ground_truth_depth"),
                        "prediction": "ERROR",
                        "is_correct": "False",
                        "delta1_score": 0.0,
                        "raw_output": f"Load Error: {str(e)}"
                    })

            if not images:
                pbar.update(len(batch))
                continue

            # --- Inference ---
            try:
                # Padding handled by processor + padding_side="left"
                inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                
                # --- Decoding ---
                generated_ids = outputs[:, inputs.input_ids.shape[1]:]
                decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                # --- Processing Results ---
                for idx, out_text in enumerate(decoded_texts):
                    original_sample = batch[valid_indices[idx]]
                    
                    gt = original_sample.get("ground_truth_depth", 0.0)
                    pred = extract_depth_from_output(out_text)
                    
                    score = 0.0
                    is_correct = "False"
                    pred_val = "N/A"

                    if pred is not None:
                        score = compute_delta1(pred, gt)
                        is_correct = "True" if score == 1.0 else "False"
                        pred_val = pred
                        total_valid += 1
                        if score == 1.0: total_correct += 1

                    # Clean text (remove newlines for CSV safety)
                    clean_output = out_text.replace('\n', ' ').replace('\r', '').strip()
                    
                    writer.writerow({
                        "image": original_sample["image"],
                        "ground_truth": gt,
                        "prediction": pred_val,
                        "is_correct": is_correct,
                        "delta1_score": score,
                        "raw_output": clean_output
                    })
                
                # Flush to disk immediately
                f.flush()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[OOM] Batch failed. Try reducing --batch_size.")
                    torch.cuda.empty_cache()
                else:
                    print(f"\n[Runtime Error] {e}")
            except Exception as e:
                print(f"\n[Unknown Error] {e}")

            # Update Progress
            current_acc = (total_correct / total_valid * 100) if total_valid > 0 else 0.0
            pbar.set_description(f"Session Acc: {current_acc:.2f}%")
            pbar.update(len(batch))

    # 7. Final Global Report
    print("\nCalculating global statistics...")
    global_acc, global_count = calculate_global_accuracy(args.output_csv)
    
    print("\n" + "="*40)
    print("EVALUATION COMPLETE")
    print(f"Session Processed:  {total_valid}")
    print(f"Total CSV Samples:  {global_count}")
    print("-" * 20)
    print(f"GLOBAL ACCURACY:    {global_acc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Updated defaults based on your inputs
    p.add_argument("--model_path", default="/mnt/dgx_lab/models/DepthLm_Qwen3B_final_final")
    p.add_argument("--eval_jsonl", default="/mnt/dgx_lab/datasets/IbimsOG/31859_dataset_metadata.jsonl")
    p.add_argument("--output_csv", default="evaluation_results.csv")
    p.add_argument("--batch_size", type=int, default=4, help="Images to process in parallel")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=128)
    
    args = p.parse_args()
    main(args)