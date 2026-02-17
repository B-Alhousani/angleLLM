import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from PIL import Image  # Requires: pip install pillow

# --- YOUR HELPER FUNCTION ---
def normalizing_focal_length(normalized_focal_length: float, intrinsics: list, image: Image):
    """
    Resizes image and scales intrinsics to match a target focal length.
    """
    # Calculate the scaling factor for the focal length normalization
    scale_factor = normalized_focal_length / intrinsics[0]
    
    # Resize the image according to the scaling factor
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    # Update the intrinsics with the normalized focal length
    # Input list assumed to be: [fx, fy, cx, cy, w, h]
    new_intrinsics = [
        intrinsics[0] * scale_factor, # fx
        intrinsics[1] * scale_factor, # fy
        intrinsics[2] * scale_factor, # cx
        intrinsics[3] * scale_factor, # cy
        new_width,
        new_height,
    ]

    # Use LANCZOS (high quality) or BILINEAR for resizing
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS), new_intrinsics


def generate_depthlm_sample(
    image_path,
    base_output_name,
    fx, fy, cx, cy,
    target_u, target_v,
    ground_truth_meters,
    target_focal=1000.0  # Default normalized focal length
):
    # 1. Load Image with OpenCV
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Error: Could not load {image_path}")
        return []

    h, w = img_cv.shape[:2]

    # 2. Convert OpenCV (BGR) -> PIL (RGB)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 3. Prepare intrinsics list [fx, fy, cx, cy, w, h]
    intrinsics_in = [fx, fy, cx, cy, w, h]

    # 4. Apply YOUR Normalization Function
    try:
        img_pil_new, intrinsics_out = normalizing_focal_length(
            target_focal, 
            intrinsics_in, 
            img_pil
        )
    except ZeroDivisionError:
        print(f"Skipping {image_path}: fx is 0")
        return []

    # 5. Convert PIL (RGB) -> OpenCV (BGR) for arrow drawing
    img_cv_new = cv2.cvtColor(np.array(img_pil_new), cv2.COLOR_RGB2BGR)

    # 6. Extract New Intrinsics
    new_fx = intrinsics_out[0]
    new_cx = intrinsics_out[2]
    new_cy = intrinsics_out[3]

    # 7. Scale Target (u, v)
    # We must calculate the scale factor used to update the target pixel coordinates
    scale = new_fx / fx 
    new_target_u = target_u * scale
    new_target_v = target_v * scale

    # --- DRAW TARGET ARROW ON NEW IMAGE ---
    canvas = img_cv_new.copy()
    COLOR_TARGET = (0, 0, 255)  # Red

    # Vector from NEW principal point to NEW target
    dx, dy = new_target_u - new_cx, new_target_v - new_cy
    mag = np.sqrt(dx**2 + dy**2) or 1
    ux, uy = dx / mag, dy / mag

    head_len, head_w = 40, 20
    shaft_len, shaft_w = 50, 8

    # Arrow geometry
    pts = np.array([[0, 0], [-head_len, -head_w], [-head_len, -shaft_w], 
                    [-head_len - shaft_len, -shaft_w], [-head_len - shaft_len, shaft_w], 
                    [-head_len, shaft_w], [-head_len, head_w]], dtype=np.float32)

    # Rotate and translate to new target
    angle = np.arctan2(uy, ux)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts = (pts @ rot.T) + [new_target_u, new_target_v]

    cv2.fillPoly(canvas, [pts.astype(np.int32)], COLOR_TARGET, cv2.LINE_AA)

    # Save
    filename = f"{base_output_name}_std.jpg"
    output_path = os.path.join(os.path.dirname(base_output_name), filename)
    cv2.imwrite(output_path, canvas)

    # Metadata
    return [{
        "image": filename,
        "type": "standardized",
        "original_fx": fx,
        "new_fx": new_fx,
        "scale_factor": round(scale, 4),
        "ground_truth_depth": round(ground_truth_meters, 3),
        "prompt": "How far is the point indicated by the red arrow from the camera in meters? Output the thinking process in <think></think> and final answer in <answer></answer> tags (only the number without anything else)",
        "label": f"<think>The red arrow points to an object...</think><answer>{round(ground_truth_meters, 2)}</answer>",
        "answer": f"The point is around {round(ground_truth_meters, 2)} meters away from the camera.",
    }]

# --- DATASET PROCESSING LOOP ---
def process_dataset(jsonl_path, images_root_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Indexing images...")
    image_lookup = {}
    for root, dirs, files in os.walk(images_root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.relpath(os.path.join(root, file), images_root_path)
                image_lookup[rel_path] = os.path.join(root, file)

    results_metadata = []
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} entries...")

    for line in tqdm(lines):
        data = json.loads(line)
        rel_img_path = data['image']
        full_img_path = image_lookup.get(rel_img_path)
        
        if full_img_path is None or not os.path.exists(full_img_path):
            continue

        fx, fy, cx, cy, _, _ = data['intrinsics']
        coords = data['pixel_coords']
        depths = data['depth']
        
        img_id = os.path.splitext(os.path.basename(rel_img_path))[0]
        
        for i, (u, v) in enumerate(coords):
            base_name = os.path.join(output_dir, f"{img_id}_p{i:03d}")
            
            # CALL WITH FOCAL LENGTH 1000
            samples = generate_depthlm_sample(
                full_img_path, 
                base_name, 
                fx, fy, cx, cy, 
                u, v, 
                depths[i],
                target_focal=1000.0
            )
            results_metadata.extend(samples)

    import random
    output_meta_path = os.path.join(output_dir, f"{random.randint(10000,99999)}_dataset_metadata.jsonl")
    with open(output_meta_path, 'w') as f:
        for entry in results_metadata:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\nDone! Generated {len(results_metadata)} samples.")