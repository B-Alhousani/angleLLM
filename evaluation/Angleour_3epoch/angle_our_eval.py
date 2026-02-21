"""
Evaluation script for AngleLM (no focal length normalization).
Images already preprocessed with arrows drawn.

INPUT:
  - Model path: Your fine-tuned model
  - JSONL file: {image, ground_truth_depth, prompt}

OUTPUT:
  - δ₁ score: Percentage within 25% of ground truth

Usage:
    python eval_anglelm.py --verbose --max_samples 50
"""

import argparse
import json
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


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
    
    # Try 4: Any number
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


def main(args):
    print("=" * 60)
    print("AngleLM Evaluation (No Focal Normalization)")
    print("=" * 60)
    print(f"Model:  {args.model_path}")
    print(f"Data:   {args.eval_jsonl}")
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
    scores = []
    failed = 0
    
    with torch.no_grad():
        for sample in tqdm(samples):
            # Load image directly (no normalization)
            try:
                img = Image.open(sample["image"]).convert("RGB")
            except Exception as e:
                failed += 1
                continue
            
            # Use prompt from JSONL
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
            
            # Compute metric
            gt = sample["ground_truth_depth"]
            pred = extract_depth_from_output(out_text)
            
            if pred is not None:
                score = compute_delta1(pred, gt)
                scores.append(score)
                if args.verbose:
                    print(f"\n{'✓' if score else '✗'} GT:{gt:.2f}m Pred:{pred:.2f}m | {out_text[:80]}...")
            else:
                failed += 1
                if args.verbose:
                    print(f"\n✗ Parse failed: {out_text[:60]}...")
    
    # 4. Results
    print(f"\n[4/4] RESULTS")
    print("=" * 60)
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  Samples:   {len(scores)}")
        print(f"  Failed:    {failed}")
        print(f"  δ₁ Score:  {avg:.4f} ({avg*100:.2f}%)")
        print(f"  Correct:   {int(sum(scores))}/{len(scores)}")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # ============ PLACEHOLDER PATHS - UPDATE THESE ============
    p.add_argument("--model_path", default="/mnt/dgx_lab/models/AndgeLM_Qwen3B_final")
    p.add_argument("--eval_jsonl", default="/mnt/dgx_lab/datasets/IbimsOurs/YOUR_METADATA.jsonl")
    # ===========================================================
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--verbose", action="store_true")
    main(p.parse_args())
