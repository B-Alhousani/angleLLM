"""
Evaluation script for DepthLM Qwen2.5-VL-3B model with CSV Reporting.
Matches the original eval.py with focal length normalization.

INPUT:
  - Model path: Your fine-tuned Qwen2.5-VL model
  - JSONL file: Each line has {image, ground_truth_depth, prompt, intrinsics (optional)}

OUTPUT:
  - CSV file with per-image results and final accuracy.
  - Console output for progress.

Usage:
    python eval_depthlm_qwen.py --output_csv my_results.csv --verbose
"""

import argparse
import json
import re
import cv2
import csv
import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================
def extract_depth_from_output(text: str) -> float | None:
    """Parse model output to extract depth value."""
    # Try 1: <answer>X</answer>
    m = re.search(r"<answer>\s*([\d.]+)\s*</answer>", text, re.IGNORECASE)
    if m:
        try: return float(m.group(1))
        except: pass
    
    # Try 2: "around X meters"
    m = re.search(r"around\s+([\d.]+)\s*meters?", text, re.IGNORECASE)
    if m:
        try: return float(m.group(1))
        except: pass
    
    # Try 3: "X meters"
    m = re.search(r"([\d.]+)\s*meters?\b", text, re.IGNORECASE)
    if m:
        try: return float(m.group(1))
        except: pass
    
    # Try 4: Any number (fallback)
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
# DATA LOADING
# ============================================================================
def load_eval_data(jsonl_path: str) -> list[dict]:
    """Load JSONL. Each line needs: image, ground_truth_depth."""
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                s = json.loads(line.strip())
                if "image" in s and "ground_truth_depth" in s:
                    samples.append(s)
            except:
                pass
    return samples


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def main(args):
    print("=" * 60)
    print("DepthLM Evaluation - Qwen2.5-VL-3B")
    print("=" * 60)
    print(f"Model:       {args.model_path}")
    print(f"Data:        {args.eval_jsonl}")
    print(f"Normalize:   {args.normalize_focal} (FL={NORMALIZED_FOCAL_LENGTH})")
    print(f"Output CSV:  {args.output_csv}")
    print("=" * 60)
    
    # 1. Load model
    print("\n[1/4] Loading model...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()
    
    # 2. Load data
    print("[2/4] Loading data...")
    samples = load_eval_data(args.eval_jsonl)
    print(f"      {len(samples)} samples loaded")
    
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    
    # 3. Evaluate
    print(f"[3/4] Evaluating {len(samples)} samples...")
    
    results_data = []  # To store CSV rows
    scores = []
    failed = 0
    
    with torch.no_grad():
        for sample in tqdm(samples):
            img_path = sample["image"]
            gt = sample["ground_truth_depth"]
            
            # Attempt to load image
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                # Log load failures
                results_data.append({
                    "image": img_path,
                    "ground_truth": gt,
                    "prediction": "ERROR",
                    "status": "Image Load Error",
                    "delta1_score": 0.0,
                    "raw_output": str(e)
                })
                failed += 1
                continue
            
            # Focal Normalization
            if args.normalize_focal and "intrinsics" in sample:
                intrinsics = sample["intrinsics"][:4]
                if intrinsics[0] == 0: intrinsics[0] = intrinsics[1]
                if intrinsics[1] == 0: intrinsics[1] = intrinsics[0]
                
                img, intrinsics = undistort_image(intrinsics, img)
                img, intrinsics = normalize_focal_length(img, intrinsics, NORMALIZED_FOCAL_LENGTH)
            
            # Prepare Prompt
            prompt = sample.get("prompt", "How far is this point from the camera?")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to("cuda")
            
            # Generate
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            out_text = processor.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            # Parse & Score
            pred = extract_depth_from_output(out_text)
            
            status = "Parse Fail"
            score = 0.0
            pred_val = "N/A"
            
            if pred is not None:
                score = compute_delta1(pred, gt)
                scores.append(score)
                pred_val = pred
                
                if score == 1.0:
                    status = "Correct"
                else:
                    status = "Incorrect"
                    
                if args.verbose:
                    icon = '✓' if score == 1.0 else '✗'
                    print(f"\n{icon} GT:{gt:.2f}m Pred:{pred:.2f}m | {status}")
            else:
                failed += 1
                if args.verbose:
                    print(f"\n✗ Parse failed: {out_text[:60]}...")

            # Store Result Row
            results_data.append({
                "image": img_path,
                "ground_truth": gt,
                "prediction": pred_val,
                "status": status,
                "delta1_score": score,
                "raw_output": out_text.replace('\n', ' ').strip()
            })
    
    # 4. Save Results to CSV
    print(f"\n[4/4] SAVING RESULTS")
    
    # Calculate final stats
    total_valid = len(scores)
    final_acc = (sum(scores) / total_valid * 100) if total_valid > 0 else 0.0
    
    # Define CSV Headers
    fieldnames = ["image", "ground_truth", "prediction", "status", "delta1_score", "raw_output"]
    
    try:
        with open(args.output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write all sample rows
            writer.writerows(results_data)
            
            # Write Summary Section at the bottom
            writer.writerow({})
            writer.writerow({})
            writer.writerow({"image": "--- SUMMARY ---"})
            writer.writerow({"image": "Total Samples", "ground_truth": len(samples)})
            writer.writerow({"image": "Valid Predictions", "ground_truth": total_valid})
            writer.writerow({"image": "Failed Parsing/Loading", "ground_truth": failed})
            writer.writerow({"image": "FINAL ACCURACY (delta1)", "ground_truth": f"{final_acc:.2f}%"})
            
        print("=" * 60)
        print(f"Results saved to: {os.path.abspath(args.output_csv)}")
        print(f"Final Accuracy:   {final_acc:.2f}%")
        print(f"Correct Samples:  {int(sum(scores))}/{total_valid}")
        print("=" * 60)
        
    except IOError as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/mnt/dgx_lab/models/DepthLm_Qwen3B_final_final")
    p.add_argument("--eval_jsonl", default="/mnt/dgx_lab/datasets/IbimsOG/65220_dataset_metadata.jsonl")
    p.add_argument("--output_csv", default="evaluation_results.csv", help="Path to save the CSV report")
    
    p.add_argument("--normalize_focal", action="store_true", default=True,
                   help="Apply focal length normalization (default: True)")
    p.add_argument("--no_normalize", action="store_false", dest="normalize_focal",
                   help="Disable focal normalization")
    
    p.add_argument("--max_samples", type=int, default=-1, help="Limit number of samples (-1 for all)")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--verbose", action="store_true", help="Print details for every sample")
    
    main(p.parse_args())