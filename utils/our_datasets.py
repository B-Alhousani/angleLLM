import json
import os
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# --- 1. PROMPT GENERATION (Angle) ---
def generate_prompt_angle_sft(angle, is_eval=False):
    # We ask for the angle using the visual cues
    problem = "Given this image, what is the angle of the point pointed by the red arrow? Output the thinking process in <think> </think> and final answer (the degree number only, without the unit) in <answer> </answer> tags."

    thinking = (
        f"<think> The point has an angle of around {angle:.2f} degrees. </think>"
    )
    if is_eval:
        solution = f"<answer> {angle} </answer>"
    else:
        solution = f"<answer> {angle:.2f} </answer>"
    return problem, thinking, solution


# --- 2. VISUAL CUE GENERATOR (Radar + Arrow) ---
def apply_radar_and_arrow(pil_image, intrinsics, target_u, target_v):
    """
    Applies the 'Radar' overlay and the 'Blocky Arrow' to the image.
    """
    # Convert PIL to OpenCV (RGB -> BGR)
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- PART A: Radar Grid (Yellow) ---
    color_grid = (0, 255, 255)  # Yellow (BGR: 0, 255, 255)
    standard_angles = [5, 10, 15, 20, 30, 45]

    for deg in standard_angles:
        theta_rad = np.deg2rad(deg)
        ax_x = int(fx * np.tan(theta_rad))
        ax_y = int(fy * np.tan(theta_rad))

        # Only draw if it fits reasonably within the view
        if ax_x < w * 2:
            cv2.ellipse(
                overlay, (int(cx), int(cy)), (ax_x, ax_y),
                0, 0, 360, color_grid, 1, cv2.LINE_AA
            )
            
            # Label the ring (e.g., "10d")
            label_pos_y = int(cy) - ax_y + 10
            if 0 < label_pos_y < h:
                cv2.putText(
                    overlay, f"{deg}d", (int(cx) + 5, label_pos_y),
                    font, 0.4, color_grid, 1, cv2.LINE_AA
                )

    # --- PART B: The Target Ring (Red) ---
    # We draw a ring at the specific angle of the target to help the model "see" the depth
    dx = target_u - cx
    dy = target_v - cy
    dist_from_center = np.sqrt((dx / fx) ** 2 + (dy / fy) ** 2)
    
    # Calculate target ellipse size
    target_ax_x = int(fx * dist_from_center)
    target_ax_y = int(fy * dist_from_center)
    
    color_target = (0, 0, 255)  # Red (BGR: 0, 0, 255)

    cv2.ellipse(
        overlay, (int(cx), int(cy)), (target_ax_x, target_ax_y),
        0, 0, 360, color_target, 1, cv2.LINE_AA
    )

    # --- PART C: The Blocky Arrow (Red) ---
    # Calculate direction vector from Center -> Target
    vec_x, vec_y = dx, dy
    mag = np.sqrt(vec_x**2 + vec_y**2)
    if mag == 0: mag = 1

    # Unit vector pointing to target
    ux, uy = vec_x / mag, vec_y / mag

    # Define Arrow Shape (Tip at 0,0)
    head_len = 50.0
    head_width_half = 25.0
    shaft_len = 60.0
    shaft_width_half = 12.0

    arrow_pts_base = np.array([
        [0.0, 0.0],  # Tip
        [-head_len, -head_width_half], 
        [-head_len, -shaft_width_half], 
        [-head_len - shaft_len, -shaft_width_half], 
        [-head_len - shaft_len, shaft_width_half], 
        [-head_len, shaft_width_half], 
        [-head_len, head_width_half], 
    ], dtype=np.float32)

    # Rotate arrow to point at target
    angle_rad = np.arctan2(uy, ux)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts_rotated = arrow_pts_base @ rot_matrix.T

    # Translate to target pixel
    pts_final = pts_rotated + np.array([target_u, target_v], dtype=np.float32)
    pts_final_int = pts_final.astype(np.int32)

    # Draw Filled Arrow
    cv2.fillPoly(overlay, [pts_final_int], color_target, lineType=cv2.LINE_AA)

    # --- FINAL CONVERSION ---
    # Convert back to RGB for PIL
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay)


# --- 3. DATASET CLASS ---
class dataset_train(Dataset):
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        sample_weights=None, 
        **kwargs 
    ) -> None:
        super().__init__()
        print("Dataset Init: Reading data from ", data_path)
        data_paths = data_path.split(";")
        image_folders = image_folder.split(";")

        self.list_data_dict = []
        
        # Load JSONL files
        for i, dp in enumerate(data_paths):
            print(f"Loading {dp}...")
            try:
                if ".jsonl" in dp:
                    import pandas as pd
                    self.list_data_dict.append(
                        pd.read_json(dp, lines=True).to_dict(orient="records")
                    )
                else:
                    self.list_data_dict.append(json.load(open(dp, "r")))
            except Exception as e:
                print(f"Error loading {dp}: {e}")

        self.image_folders = image_folders
        
        # Sample weights
        self.sample_weights = (
            [int(x) for x in sample_weights.split(";")]
            if sample_weights
            else [1] * len(self.list_data_dict)
        )
        print(f"Finished loading. Found {sum(len(d) for d in self.list_data_dict)} images.")

    def __len__(self) -> int:
        # Estimate total samples (images * points per image)
        length = 0
        for data_dict in self.list_data_dict:
            length += len(data_dict) * 100 
        return length

    def __getitem__(self, index):
        # 1. Select Dataset
        id_dataset = random.choices(
            range(len(self.list_data_dict)), weights=self.sample_weights, k=1
        )[0]
        
        current_dataset = self.list_data_dict[id_dataset]
        current_img_folder = self.image_folders[id_dataset] if id_dataset < len(self.image_folders) else self.image_folders[0]
        
        # 2. Get Image Data
        img_idx = index % len(current_dataset)
        sample_data = current_dataset[img_idx]

        # 3. Load Raw Image
        try:
            # Handle different JSON key names for filename
            fname = sample_data.get("image") or sample_data.get("file_name") or sample_data.get("filename")
            image_path = os.path.join(current_img_folder, fname.lstrip("/"))
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # print(f"Error loading image: {e}")
            return self.__getitem__((index + 1) % self.__len__())

        # 4. Get Data Points
        pixel_coords = sample_data.get("pixel_coords", [])
        intrinsics = sample_data.get("intrinsics", [1000, 1000, image.width/2, image.height/2]) # Default if missing
        
        # Check for Angle or Depth
        if "angle" in sample_data:
            values = sample_data["angle"]
        elif "depth" in sample_data:
            values = sample_data["depth"] # Fallback if you want to test on depth data
        else:
            return self.__getitem__((index + 1) % self.__len__())

        if len(pixel_coords) == 0:
            return self.__getitem__((index + 1) % self.__len__())

        # 5. Pick Random Point
        random_point_idx = random.randint(0, len(pixel_coords) - 1)
        target_coord = pixel_coords[random_point_idx]
        target_val = values[random_point_idx]
        
        target_u, target_v = target_coord[0], target_coord[1]

        # 6. Apply Visuals (Radar + Arrow)
        try:
            # Pass intrinsics (fx, fy, cx, cy)
            # Assuming intrinsics list is [fx, fy, cx, cy, ...]
            image = apply_radar_and_arrow(image, intrinsics[:4], target_u, target_v)
        except Exception as e:
            print(f"Error drawing visuals: {e}")
            return self.__getitem__((index + 1) % self.__len__())

        # 7. Prepare Output
        data_dict = {}
        data_dict["image"] = image
        
        # Prompt
        problem, thinking, solution = generate_prompt_angle_sft(target_val)
        
        data_dict["problem"] = problem
        data_dict["thinking"] = thinking
        data_dict["solution"] = solution
        data_dict["system"] = "You are a helpful assistant."

        # Qwen Format
        data_dict["prompt"] = [
            {
                "content": [
                    {"image": data_dict["image"], "type": "image"},
                    {"text": data_dict["problem"], "type": "text"},
                ],
                "role": "user",
            }
        ]
        
        return data_dict