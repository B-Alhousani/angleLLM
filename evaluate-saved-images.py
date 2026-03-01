import argparse
import csv
import os
import re
import torch
import json
import math
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

# Import OFFICIAL loader to get the correct visual cues
from utils.angle_dataset import DatasetEval

# --- HELPER: Extract Float from <answer> tags ---
def extract_value(text):
    """
    Extracts the first number found inside <answer> tags.
    """
    if not text:
        return None
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        number_match = re.search(r"[-+]?\d*\.\d+|\d+", content)
        if number_match:
            try:
                return float(number_match.group())
            except ValueError:
                return None
    return None

# --- CONVERSION HELPERS ---
def convert_example_pixtral(example, image_before_text=None):
    messages = []
    problem = example.get("problem")
    if "images" in example:
        images = example.get("images")
        content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "content": problem}] if image_before_text else \
                  [{"type": "text", "content": problem}] + [{"type": "image", "image": img} for img in images]
    else:
        # UPDATED: Fallback to "vlm_image"
        image = example.get("image") or example.get("vlm_image")
        content = [{"type": "image", "image": image}, {"type": "text", "content": problem}] if image_before_text else \
                  [{"type": "text", "content": problem}, {"type": "image", "image": image}]
    
    messages.append({"role": "user", "content": content})
    example["messages"] = messages
    return example

def convert_example_qwen(example, image_before_text=None):
    messages = []
    problem = example.get("problem")
    messages.append({"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]})
    
    if "images" in example:
        images = example.get("images")
        content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": problem}]
    else:
        # UPDATED: Fallback to "vlm_image"
        image = example.get("image") or example.get("vlm_image")
        content = [{"type": "image", "image": image}, {"type": "text", "text": problem}]
        
    messages.append({"role": "user", "content": content})
    example["messages"] = messages
    return example

def main(args):
    model_path = args.model_path
    img_folder = args.image_folder
    json_path = args.json_path
    
    # --- SETUP OUTPUT ---
    output_base_dir = "/mnt/dgx_lab/evaluation-with-red/angle-prompt-2/"
    output_img_dir = os.path.join(output_base_dir, "images")
    output_csv_path = os.path.join(output_base_dir, "results.csv")
    
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    
    # --- 1. LOAD PARENT FILENAMES ---
    # We read the JSONL to get the list of 100 original names
    print(f"Loading parent filenames from {json_path}...")
    parent_filenames = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Try keys in order of likelihood
                    fname = data.get("image", data.get("file_name", data.get("filename", data.get("id"))))
                    if not fname and "images" in data and len(data["images"]) > 0:
                        fname = data["images"][0] # Fallback for multi-image lists
                    
                    if fname:
                        parent_filenames.append(str(fname))
                except:
                    continue

    num_parents = len(parent_filenames)
    print(f"Found {num_parents} original images in JSONL.")

    # --- LOAD MODEL ---
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    is_llava_arch = any(x in model_path.lower() for x in ["pixtral", "depthlm", "13b", "llava"])
    if is_llava_arch:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, 
            attn_implementation={"text_config": "flash_attention_2", "vision_config": "eager"}, 
            device_map="auto"
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2", 
            device_map="auto"
        )
    model.eval()

    # --- LOAD DATASET ---
    dataset = DatasetEval(json_path, img_folder)
    print(f"Dataset expanded size: {len(dataset)} samples.")

    # --- EVALUATION LOOP ---
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Index", "Parent_File", "Ground_Truth", "Prediction", "Delta1_Met", "Raw_Output"])

        samples_to_eval = min(args.samples_to_eval, len(dataset))
        indices = list(range(samples_to_eval))

        with torch.no_grad():
            for i in tqdm(range(0, len(indices), args.bsz)):
                batch_indices = indices[i : i + args.bsz]
                batch_messages = [dataset[j] for j in batch_indices]

                # --- MODEL INPUT PREP (Standard) ---
                if is_llava_arch:
                    chat = [convert_example_pixtral(msg, True)["messages"] for msg in batch_messages]
                    inputs = processor.apply_chat_template(
                        chat, add_generation_prompt=True, tokenize=True, return_dict=True,
                        padding=True, return_tensors="pt"
                    ).to("cuda", dtype=torch.bfloat16)
                else:
                    text_inputs = [
                        processor.apply_chat_template(convert_example_qwen(msg)["messages"], tokenize=False, add_generation_prompt=True)
                        for msg in batch_messages
                    ]
                    # UPDATED: Fallback to "vlm_image" in list comprehension
                    image_inputs = [x["images"] if "images" in x else (x.get("image") or x.get("vlm_image")) for x in batch_messages]
                    inputs = processor(text=text_inputs, images=image_inputs, padding=True, return_tensors="pt").to("cuda")
                if "image_sizes" in inputs: inputs.pop("image_sizes")

                # --- GENERATE ---
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

                # --- SAVE LOGIC (CORRECTED) ---
                for idx, (msg_data, output_text) in enumerate(zip(batch_messages, batch_output_text)):
                    current_idx = batch_indices[idx]
                    
                    # --- FIX: USE MODULO FOR INTERLEAVED DATA ---
                    # If current_idx is 0, 1, 2... and parents are A, B, C...
                    # We map 0->A, 1->B ... 100->A
                    if num_parents > 0:
                        parent_idx = current_idx % num_parents
                        original_fname = parent_filenames[parent_idx]
                    else:
                        original_fname = f"sample_{current_idx}"

                    # Clean filename
                    clean_name = os.path.splitext(os.path.basename(original_fname))[0]
                    
                    # Format: my_image_idx505.png
                    img_save_name = f"{clean_name}_idx{current_idx}.png"
                    img_save_path = os.path.join(output_img_dir, img_save_name)
                    
                    try:
                        if "image" in msg_data:
                            msg_data["image"].save(img_save_path)
                        # UPDATED: Added fallback clause for saving "vlm_image"
                        elif "vlm_image" in msg_data:
                            msg_data["vlm_image"].save(img_save_path)
                        elif "images" in msg_data and len(msg_data["images"]) > 0:
                            msg_data["images"][0].save(img_save_path)
                    except Exception as e:
                        print(f"Error saving {img_save_name}: {e}")

                    # Metrics
                    pred_val = extract_value(output_text)
                    gt_text = msg_data.get("solution", "")
                    gt_val = extract_value(gt_text)
                    
                    delta1_status = 0
                    if pred_val and gt_val and gt_val > 0 and pred_val > 0:
                        ratio = max(gt_val / pred_val, pred_val / gt_val)
                        if ratio < 1.25: delta1_status = 1

                    csv_writer.writerow([
                        current_idx,
                        original_fname, 
                        gt_val if gt_val else "ERR",
                        pred_val if pred_val else "ERR",
                        delta1_status,
                        output_text.replace("\n", " ")
                    ])
                
                csv_file.flush()
    print("Evaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, default="./examples/ibims1/")
    parser.add_argument("--json_path", type=str, default="./examples/ibims1/ibims1_val.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--samples_to_eval", type=int, default=11000)
    args = parser.parse_args()
    main(args)