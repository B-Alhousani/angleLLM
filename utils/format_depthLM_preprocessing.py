import cv2
import numpy as np
import json
import os
import random
from tqdm import tqdm

def generate_hybrid_sample(
    image_path, base_output_name, fx, fy, cx, cy, 
    target_u, target_v, ground_truth_meters, multipliers=[0.5, 1.0, 1.5]
):
    """
    Combines Format 1's augmentation with Format 2's advanced visual markers.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    samples_metadata = []

    COLOR_GRID = (0, 255, 255)   # Yellow (Ruler)
    COLOR_TARGET = (0, 0, 255)   # Red (Question)
    COLOR_REF = (0, 255, 0)      # Green (Focal Bar)
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for mult in multipliers:
        curr_fx, curr_fy = fx * mult, fy * mult
        
        label = "true"
        if mult < 1.0: label = "wide"
        if mult > 1.0: label = "tele"

        filename = f"{base_output_name}_{label}.jpg"
        output_path = os.path.join(os.path.dirname(base_output_name), filename)
        canvas = img.copy()

        # 1. DRAW YELLOW ANGULAR GRID (The Ruler)
        standard_angles = [5, 10, 15, 20, 30, 45]
        for deg in standard_angles:
            theta_rad = np.deg2rad(deg)
            rx = int(curr_fx * np.tan(theta_rad))
            ry = int(curr_fy * np.tan(theta_rad))
            if rx < w * 1.5:
                cv2.ellipse(canvas, (int(cx), int(cy)), (rx, ry), 0, 0, 360, COLOR_GRID, 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{deg}d", (int(cx) + 5, int(cy) - ry + 12), FONT, 0.4, COLOR_GRID, 1, cv2.LINE_AA)

        # 2. DRAW GREEN FOCAL BAR (Physical Focal Length Reference)
        # Length = f * tan(45deg) = f. This helps the model see the zoom level.
        bar_len = min(int(curr_fx), w - 40)
        start_x, start_y = 20, h - 30
        cv2.line(canvas, (start_x, start_y), (start_x + bar_len, start_y), COLOR_REF, 3)
        cv2.putText(canvas, "Ref: 45deg FoV", (start_x, start_y - 10), FONT, 0.5, COLOR_REF, 1, cv2.LINE_AA)

        # 3. DRAW TARGET GEOMETRY (The Question)
        dx, dy = target_u - cx, target_v - cy
        # Calculate target's specific angle for the red ring
        target_theta_rad = np.arctan(np.sqrt((dx/curr_fx)**2 + (dy/curr_fy)**2))
        target_deg = np.degrees(target_theta_rad)
        
        # Red Target Ring
        t_rx = int(curr_fx * np.tan(target_theta_rad))
        t_ry = int(curr_fy * np.tan(target_theta_rad))
        cv2.ellipse(canvas, (int(cx), int(cy)), (t_rx, t_ry), 0, 0, 360, COLOR_TARGET, 2, cv2.LINE_AA)

        # Target Arrow
        mag = np.sqrt(dx**2 + dy**2) or 1
        ux, uy = dx / mag, dy / mag
        arrow_pts = np.array([[0, 0], [-40, -20], [-40, -8], [-90, -8], [-90, 8], [-40, 8], [-40, 20]], dtype=np.float32)
        angle = np.arctan2(uy, ux)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        arrow_pts = (arrow_pts @ rot.T) + [target_u, target_v]
        cv2.fillPoly(canvas, [arrow_pts.astype(np.int32)], COLOR_TARGET, cv2.LINE_AA)

        cv2.imwrite(output_path, canvas)

        samples_metadata.append({
            "image": filename,
            "type": label,
            "ground_truth_depth": round(ground_truth_meters, 3),
            "prompt": "How far is the point indicated by the red arrow? The yellow rings show standard angular intervals, and the green bar represents the scale of a 45-degree angle. Use the red ring to trace the point's angle back to the ruler.",
            "label": f"<think>The green reference bar indicates a focal length of {int(curr_fx)} pixels. The target's red ring falls near the {round(target_deg, 1)} degree marker. Based on perspective geometry...</think><answer>{round(ground_truth_meters, 2)}</answer>"
        })

    return samples_metadata

def process_dataset(jsonl_path, images_root_path, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # Indexing (Format 1 logic)
    image_lookup = {}
    for root, _, files in os.walk(images_root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.relpath(os.path.join(root, file), images_root_path)
                image_lookup[rel_path] = os.path.join(root, file)

    results_metadata = []
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        data = json.loads(line)
        full_img_path = image_lookup.get(data['image'])
        if not full_img_path: continue

        fx, fy, cx, cy, _, _ = data['intrinsics']
        img_id = os.path.splitext(os.path.basename(data['image']))[0]
        
        for i, (u, v) in enumerate(data['pixel_coords']):
            base_name = os.path.join(output_dir, f"{img_id}_p{i:03d}")
            samples = generate_hybrid_sample(full_img_path, base_name, fx, fy, cx, cy, u, v, data['depth'][i])
            results_metadata.extend(samples)

    # Save Metadata
    out_meta = os.path.join(output_dir, f"dataset_metadata_{random.randint(10,99)}.jsonl")
    with open(out_meta, 'w') as f:
        for entry in results_metadata: f.write(json.dumps(entry) + "\n")