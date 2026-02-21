import cv2
import numpy as np
import json
import os
from tqdm import tqdm
import random

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
    multipliers=[1.0], # Can be used for augmentation if needed
):
    """
    Generates training samples with "Visual Layer 2.0" (Sniper Scope + Focal Bar).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return []

    h, w = img.shape[:2]
    samples_metadata = []

    # Visual Constants
    COLOR_RULER = (0, 255, 255)  # Yellow for Axis/Ticks
    COLOR_TARGET = (0, 0, 255)   # Red for Target
    COLOR_BAR = (0, 255, 0)      # Green for Focal Bar
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for mult in multipliers:
        # Scale intrinsics if using multipliers (optional augmentation)
        curr_fx = fx * mult
        curr_fy = fy * mult
        
        # Label logic
        label = "true"
        if mult < 1.0: label = "wide"
        if mult > 1.0: label = "tele"

        filename = f"{base_output_name}_{label}.jpg"
        output_path = os.path.join(os.path.dirname(base_output_name), filename)
        
        # Work on a copy
        canvas = img.copy()

        # --- 1. Draw "Sniper Scope" Crosshair & Ticks (The Ruler) ---
        # Draw Center Crosshair
        cv2.line(canvas, (int(cx)-20, int(cy)), (int(cx)+20, int(cy)), COLOR_RULER, 1, cv2.LINE_AA)
        cv2.line(canvas, (int(cx), int(cy)-20), (int(cx), int(cy)+20), COLOR_RULER, 1, cv2.LINE_AA)

        # Draw Angle Ticks along the X and Y axes
        standard_angles = [5, 10, 15, 20, 30, 45, 60]
        
        for deg in standard_angles:
            # Calculate pixel distance for this angle
            # distance = f * tan(theta)
            theta_rad = np.deg2rad(deg)
            px_dist_x = int(curr_fx * np.tan(theta_rad))
            px_dist_y = int(curr_fy * np.tan(theta_rad))

            # Draw X-Axis Ticks (Left and Right)
            for sign in [-1, 1]:
                x_pos = int(cx + (px_dist_x * sign))
                if 0 <= x_pos < w:
                    # Draw a small vertical tick mark
                    cv2.line(canvas, (x_pos, int(cy)-5), (x_pos, int(cy)+5), COLOR_RULER, 2, cv2.LINE_AA)
                    
                    # Add Text Label for major angles only (to reduce clutter)
                    if deg in [10, 20, 30, 45]:
                        text = f"{deg}"
                        cv2.putText(canvas, text, (x_pos-10, int(cy)+20), FONT, 0.4, COLOR_RULER, 1, cv2.LINE_AA)

            # Draw Y-Axis Ticks (Up and Down) - Optional, good for vertical perspective
            for sign in [-1, 1]:
                y_pos = int(cy + (px_dist_y * sign))
                if 0 <= y_pos < h:
                    # Draw a small horizontal tick mark
                    cv2.line(canvas, (int(cx)-5, y_pos), (int(cx)+5, y_pos), COLOR_RULER, 2, cv2.LINE_AA)

        # --- 2. Draw "Focal Bar" (The Visual f Token) ---
        # Draw a bar in bottom-left representing 45 degrees FOV
        # Length = f * tan(45) = f pixels
        bar_len = int(curr_fx)
        # Clamp bar to image width safely
        bar_len = min(bar_len, w - 50)
        
        bar_x = 20
        bar_y = h - 30
        cv2.line(canvas, (bar_x, bar_y), (bar_x + bar_len, bar_y), COLOR_BAR, 4, cv2.LINE_AA)
        cv2.putText(canvas, "Ref: 45 deg", (bar_x, bar_y - 10), FONT, 0.5, COLOR_BAR, 1, cv2.LINE_AA)

        # --- 3. Draw Target Arrow (Your existing logic) ---
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
        
        # Save to disk
        cv2.imwrite(output_path, canvas)

        # Append Metadata
        samples_metadata.append({
            "image": filename,
            "type": label,
            "ground_truth_depth": round(ground_truth_meters, 3),
            "prompt": "How far is the point indicated by the red arrow from the camera in meters? "
                      "The image contains a yellow crosshair with ticks marking 5, 10, 20, 30, and 45 degree angles from the center. "
                      "There is also a green reference bar in the bottom corner representing a 45-degree field of view. "
                      "Use these visual guides to estimate the perspective and scale.",
            "label": f"<think>The green bar indicates the focal length. The target is near the X degree tick...</think><answer>{round(ground_truth_meters, 2)}</answer>",
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
    print("Indexing images in nested folders...")
    image_lookup = {}
    for root, dirs, files in os.walk(images_root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.relpath(os.path.join(root, file), images_root_path)
                image_lookup[rel_path] = os.path.join(root, file)

    results_metadata = []

    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} JSON entries...")

    for line in tqdm(lines):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        rel_img_path = data.get('image')
        if not rel_img_path: continue
        
        full_img_path = image_lookup.get(rel_img_path)
        
        if full_img_path is None or not os.path.exists(full_img_path):
            # Try direct path if lookup failed
            full_img_path = os.path.join(images_root_path, rel_img_path)
            if not os.path.exists(full_img_path):
                continue

        # Extract intrinsics safely
        if 'intrinsics' in data:
            intrinsics = data['intrinsics']
            fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        else:
            continue # Skip if no intrinsics

        coords = data.get('pixel_coords', [])
        depths = data.get('depth', [])
        
        img_id = os.path.splitext(os.path.basename(rel_img_path))[0]
        
        for i, (u, v) in enumerate(coords):
            if i >= len(depths): break
            
            base_name = os.path.join(output_dir, f"{img_id}_p{i:03d}")
            
            samples = generate_depthlm_sample(
                full_img_path, 
                base_name, 
                fx, fy, cx, cy, 
                u, v, 
                depths[i]
            )
            
            if samples:
                results_metadata.extend(samples)

    random_number = random.randint(10000, 99999)
    output_meta_path = os.path.join(output_dir, f"{random_number}_dataset_metadata.jsonl")
    with open(output_meta_path, 'w') as f:
        for entry in results_metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone! Generated {len(results_metadata)} samples in {output_dir}")