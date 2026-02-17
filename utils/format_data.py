import cv2
import numpy as np
import json
import os
from tqdm import tqdm

def generate_depthlm_sample(
    image_path,
    base_output_name,
    fx,
    fy,
    cx,
    cy,
    target_u,
    target_v,
    ground_truth_meters,
    multipliers=[1.0], # Defaulted to 1.0 as per your TODO note
):
    """
    Generates training samples (Wide, True, Tele) to decouple pixel size from angular depth.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return []

    h, w = img.shape[:2]
    samples_metadata = []

    COLOR_GRID = (0, 255, 255)  # Yellow
    COLOR_TARGET = (0, 0, 255)  # Red
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for mult in multipliers:
        curr_fx = fx * mult
        curr_fy = fy * mult

        label = "true"
        if mult < 1.0: label = "wide"
        if mult > 1.0: label = "tele"

        # Create unique filename for this augmentation
        filename = f"{base_output_name}_{label}.jpg"
        output_path = os.path.join(os.path.dirname(base_output_name), filename)
        
        canvas = img.copy()

        # 1. Draw Angular Grid
        standard_angles = [5, 10, 15, 20, 30, 45, 60]
        for deg in standard_angles:
            theta_rad = np.deg2rad(deg)
            rx = int(curr_fx * np.tan(theta_rad))
            ry = int(curr_fy * np.tan(theta_rad))

            if rx < w * 2:
                cv2.ellipse(canvas, (int(cx), int(cy)), (rx, ry), 0, 0, 360, COLOR_GRID, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{deg}d", (int(cx) + 5, int(cy) - ry + 12), FONT, 0.4, COLOR_GRID, 1, cv2.LINE_AA)

        # 2. Draw Horizon and Center
        cv2.line(canvas, (0, int(cy)), (w, int(cy)), COLOR_GRID, 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, (int(cx), int(cy)), COLOR_GRID, cv2.MARKER_CROSS, 20, 1)

        # 3. Draw Target Arrow
        dx, dy = target_u - cx, target_v - cy
        mag = np.sqrt(dx**2 + dy**2) or 1
        ux, uy = dx / mag, dy / mag

        head_len, head_w = 40, 20
        shaft_len, shaft_w = 50, 8

        pts = np.array([[0, 0], [-head_len, -head_w], [-head_len, -shaft_w], 
                        [-head_len - shaft_len, -shaft_w], [-head_len - shaft_len, shaft_w], 
                        [-head_len, shaft_w], [-head_len, head_w]], dtype=np.float32)

        angle = np.arctan2(uy, ux)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        pts = (pts @ rot.T) + [target_u, target_v]

        cv2.fillPoly(canvas, [pts.astype(np.int32)], COLOR_TARGET, cv2.LINE_AA)
        cv2.imwrite(output_path, canvas)

        samples_metadata.append({
            "image": filename,
            "type": label,
            "ground_truth_depth": round(ground_truth_meters, 3),
            "prompt": "How far is this point from the camera in meters? The yellow angular grid acts as a ruler, showing angles from the camera center. Use it as a reference to interpret the perspective and the scale of objects in the scene. Output the thinking process in <think></think> and final answer in <answer></answer> tags (only the number without anything else)",
            "label": f"<think>The target is at the Y degree ring...</think><answer>{round(ground_truth_meters, 2)}</answer>",
            "answer": f"The point is around {round(ground_truth_meters, 2)} meters away from the camera.",
        })

    return samples_metadata

def process_dataset(jsonl_path, images_root_path, output_dir):
    """
    Crawls nested folders in images_root_path to match JSONL entries 
    and processes them using generate_depthlm_sample.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Build a lookup map for nested images
    # Maps filename -> absolute_path
    print("Indexing images in nested folders...")
    image_lookup = {}
    for root, dirs, files in os.walk(images_root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Store full path, keyed by the relative path or filename 
                # depending on how your JSONL references them.
                # Here we use the relative path from images_root_path
                rel_path = os.path.relpath(os.path.join(root, file), images_root_path)
                image_lookup[rel_path] = os.path.join(root, file)

    results_metadata = []

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} JSON entries...")

    for line in tqdm(lines):
        data = json.loads(line)
        rel_img_path = data['image'] # e.g., "y/a/img01.jpg"
        
        # Check if the image exists in our crawled lookup
        full_img_path = image_lookup.get(rel_img_path)
        
        if full_img_path is None or not os.path.exists(full_img_path):
            print(f"Warning: Image not found in tree: {rel_img_path}")
            continue

        fx, fy, cx, cy, _, _ = data['intrinsics']
        coords = data['pixel_coords']
        depths = data['depth']
        
        img_id = os.path.splitext(os.path.basename(rel_img_path))[0]
        
        for i, (u, v) in enumerate(coords):
            # Base name for the generated images
            base_name = os.path.join(output_dir, f"{img_id}_p{i:03d}")
            
            # Generate the samples (Wide, True, Tele)
            samples = generate_depthlm_sample(
                full_img_path, 
                base_name, 
                fx, fy, cx, cy, 
                u, v, 
                depths[i]
            )
            
            if samples:
                results_metadata.extend(samples)

    import random
    random_number = random.randint(10000, 99999)
    output_meta_path = os.path.join(output_dir, f"{random_number}_dataset_metadata.jsonl")
    with open(output_meta_path, 'w') as f:
        for entry in results_metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone! Generated {len(results_metadata)} samples in {output_dir}")