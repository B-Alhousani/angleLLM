# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import os
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# =====================================================================
# 1. SHARED PROMPT & VISUAL CUE LOGIC
# =====================================================================

import os

def generate_prompt_depth_sft(depth, is_eval=False):
    og_prompt = "Given this image, how far is the point pointed by the red arrow from the camera? Output the thinking process in <think> </think> and final answer (the meter number only, without the unit) in <answer> </answer> tags."
    
    prompt1 = "You are given an image with a circular angular grid and a highlighted point. The rings represent fixed viewing angles of the camera. Estimate the distance from the camera to the highlighted point in meters? Output the thinking process in <think> </think> and final answer (the meter number only, without the unit) in <answer> </answer> tags."
    
    prompt2 = "You are given an image with a circular angular grid and a highlighted point with an arrow. The rings represent the pinhole camera projection angles. Use this angular context to estimate the distance from the camera to the highlighted point in meters? Output the thinking process in <think> </think> and final answer (the meter number only, without the unit) in <answer> </answer> tags."
    
    # Fetch the environment variable, defaulting to 'og' if not set
    prompt_type = os.environ.get("PROMPT_TYPE", "og").lower()
    
    if prompt_type == "prompt1":
        selected_prompt = prompt1
    elif prompt_type == "prompt2":
        selected_prompt = prompt2
    else:
        selected_prompt = og_prompt
    thinking = f"<think> The point is around {depth:.2f} meters away from the camera. </think>"
    
    if is_eval:
        solution = f"<answer> {depth} </answer>"
    else:
        solution = f"<answer> {depth:.2f} </answer>"
        
    return selected_prompt, thinking, solution
# def apply_visual_cues(image: Image, intrinsics: list, target_coords: tuple) -> Image:
#     """
#     Draws the yellow angle grids (radar) and the small 5-pixel red arrow.
#     """
#     # --- PART 1: Draw Radar Grids (CV2) ---
#     overlay = np.array(image)
#     fx, fy, cx, cy = intrinsics[:4]
#     w, h = image.size
    
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     color_grid = (255, 255, 0)  # Yellow

#     standard_angles = [5, 10, 15, 20, 30, 45]
#     for deg in standard_angles:
#         theta_rad = np.deg2rad(deg)
#         ax_x = int(fx * np.tan(theta_rad))
#         ax_y = int(fy * np.tan(theta_rad))

#         if ax_x < w * 2:
#             cv2.ellipse(
#                 overlay, (int(cx), int(cy)), (ax_x, ax_y),
#                 0, 0, 360, color_grid, 1, cv2.LINE_AA
#             )
#             label_pos_y = int(cy) - ax_y + 10
#             if 0 < label_pos_y < h:
#                 cv2.putText(
#                     overlay, f"{deg}d", (int(cx) + 5, label_pos_y),
#                     font, 0.4, color_grid, 1, cv2.LINE_AA
#                 )

#     # --- PART 2: Draw Small Red Arrow (PIL) ---
#     image = Image.fromarray(overlay)
#     scaled_pixel_x, scaled_pixel_y = target_coords
#     cross_size = 5

#     if (cross_size <= scaled_pixel_x < image.width - cross_size and 
#         cross_size <= scaled_pixel_y < image.height - cross_size):
        
#         # Draw arrow shaft
#         for dx in range(1, cross_size + 1):
#             image.putpixel((scaled_pixel_x - dx, scaled_pixel_y), (255, 0, 0))
#         # Draw arrow head
#         for dy in range(1, cross_size // 2 + 1):
#             image.putpixel((scaled_pixel_x - dy - 1, scaled_pixel_y + dy), (255, 0, 0))
#             image.putpixel((scaled_pixel_x - dy - 1, scaled_pixel_y - dy), (255, 0, 0))

#     return image
def apply_visual_cues(image: Image, intrinsics: list, target_coords: tuple) -> Image:
    """
    Draws the yellow angle grids (radar), the specific target ring, 
    and the small 5-pixel red marker.
    """
    # --- PART 1: Draw Radar Grids (CV2) ---
    overlay = np.array(image)
    fx, fy, cx, cy = intrinsics[:4]
    w, h = image.size
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_grid = (255, 255, 0)  # Yellow

    standard_angles = [5, 10, 15, 20, 30, 45]
    for deg in standard_angles:
        theta_rad = np.deg2rad(deg)
        ax_x = int(fx * np.tan(theta_rad))
        ax_y = int(fy * np.tan(theta_rad))

        if ax_x < w * 2:
            cv2.ellipse(
                overlay, (int(cx), int(cy)), (ax_x, ax_y),
                0, 0, 360, color_grid, 1, cv2.LINE_AA
            )
            label_pos_y = int(cy) - ax_y + 10
            if 0 < label_pos_y < h:
                cv2.putText(
                    overlay, f"{deg}d", (int(cx) + 5, label_pos_y),
                    font, 0.4, color_grid, 1, cv2.LINE_AA
                )

    # --- NEW: Draw Custom Ring for Target Pixel ---
    scaled_pixel_x, scaled_pixel_y = target_coords
    dx = scaled_pixel_x - cx
    dy = scaled_pixel_y - cy
    
    # Calculate angular magnitude (rho_q) exactly as in Equation 2
    rho_q = np.sqrt((dx / fx)**2 + (dy / fy)**2)
    
    # Calculate the ellipse axes for this specific point
    target_ax_x = int(fx * rho_q)
    target_ax_y = int(fy * rho_q)
    
    # Draw the specific target ring (using red to match the point marker)
    color_target_ring = (255, 0, 0)  # Red
    cv2.ellipse(
        overlay, (int(cx), int(cy)), (target_ax_x, target_ax_y),
        0, 0, 360, color_target_ring, 1, cv2.LINE_AA
    )

    # --- PART 2: Draw Small Red Arrow/Marker (PIL) ---
    image = Image.fromarray(overlay)
    cross_size = 5

    if (cross_size <= scaled_pixel_x < image.width - cross_size and 
        cross_size <= scaled_pixel_y < image.height - cross_size):
        
        # Draw arrow shaft
        for i in range(1, cross_size + 1):
            image.putpixel((scaled_pixel_x - i, scaled_pixel_y), (255, 0, 0))
        # Draw arrow head
        for i in range(1, cross_size // 2 + 1):
            image.putpixel((scaled_pixel_x - i - 1, scaled_pixel_y + i), (255, 0, 0))
            image.putpixel((scaled_pixel_x - i - 1, scaled_pixel_y - i), (255, 0, 0))

    return image
# =====================================================================
# 2. CAMERA GEOMETRY SHARED LOGIC
# =====================================================================

def scale_image_by_factor(image: Image, intrinsics: list, scale_factor: float, apply_scaling: bool = False):
    """
    Scales the image and its intrinsics by a multiplication factor.
    Controlled by apply_scaling boolean (defaults to False).
    """
    if not apply_scaling or scale_factor == 1.0:
        # Ensure the list has 6 elements [fx, fy, cx, cy, w, h] for pipeline consistency
        if len(intrinsics) < 6:
             return image, intrinsics[:4] + [image.width, image.height]
        return image, intrinsics

    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    image = image.resize((new_width, new_height))

    intrinsics_new = [
        intrinsics[0] * scale_factor, 
        intrinsics[1] * scale_factor, 
        intrinsics[2] * scale_factor, 
        intrinsics[3] * scale_factor, 
        new_width,                     
        new_height                     
    ]
    return image, intrinsics_new

# =====================================================================
# 3. BASE DATASET (Shared Logic)
# =====================================================================

class BaseDepthDataset(Dataset):
    def __init__(self, data_path: str, image_folder: str):
        super().__init__()
        self.data_paths = data_path.split(";")
        self.image_folders = image_folder.split(";")
        self.list_data_dict = []

        for dp in self.data_paths:
            if ".jsonl" in dp:
                try:
                    self.list_data_dict.append(pd.read_json(dp, lines=True).to_dict(orient="records"))
                except Exception as e:
                    print(f"Pandas failed, falling back to JSON for {dp}: {e}")
                    self.list_data_dict.append(json.load(open(dp, "r")))
            else:
                self.list_data_dict.append(json.load(open(dp, "r")))
                
    def _process_sample(self, dataset_idx, img_idx, point_idx, scale_factor, apply_scaling, is_eval):
        current_dataset = self.list_data_dict[dataset_idx]
        current_img_folder = self.image_folders[dataset_idx] if dataset_idx < len(self.image_folders) else self.image_folders[0]
        
        sample_data = current_dataset[img_idx]
        
        # 1. Load Image
        fname = sample_data.get("image") or sample_data.get("file_name") or sample_data.get("filename")
        image_path = os.path.join(current_img_folder, fname.lstrip("/"))
        image = Image.open(image_path).convert("RGB")
        
        # 2. Extract Meta
        pixel_coords = sample_data.get("pixel_coords", [])
        intrinsics = sample_data.get("intrinsics", [1000, 1000, image.width/2, image.height/2])[:4]
        
        if intrinsics[0] == 0.0: intrinsics[0] = intrinsics[1]
        if intrinsics[1] == 0.0: intrinsics[1] = intrinsics[0]

        target_val = sample_data["depth"][point_idx]
        target_coord = pixel_coords[point_idx]

        # 3. Geometric Adjustments
        image, intrinsics = scale_image_by_factor(image, intrinsics, scale_factor, apply_scaling=apply_scaling)

        target_u = int((target_coord[0] - sample_data["intrinsics"][2]) * (intrinsics[0] / sample_data["intrinsics"][0]) + intrinsics[2])
        target_v = int((target_coord[1] - sample_data["intrinsics"][3]) * (intrinsics[1] / sample_data["intrinsics"][1]) + intrinsics[3])

        # 4. Apply Visual Cues (Grid + Small Arrow)
        image = apply_visual_cues(image, intrinsics, (target_u, target_v))

        # 5. Format Output
        problem, thinking, solution = generate_prompt_depth_sft(target_val, is_eval=is_eval)
        
        return {
            "vlm_image": image,
            "problem": problem,
            "thinking": thinking,
            "solution": solution,
            "pixel_coord": [target_u, target_v],
            "intrinsics": intrinsics,
            "system": "You are a helpful assistant.",
            "prompt": [{
                "content": [
                    {"image": image, "type": "image"},
                    {"text": problem, "type": "text"},
                ],
                "role": "user",
            }]
        }

# =====================================================================
# 4. TRAIN & EVAL DATASETS
# =====================================================================

class DatasetTrain(BaseDepthDataset):
    def __init__(self, data_path: str, image_folder: str, sample_weights=None, focal_length_range=(0.8, 1.2), apply_scaling=False):
        super().__init__(data_path, image_folder)
        self.apply_scaling = apply_scaling
        
        for i, dp in enumerate(self.data_paths):
            if "scannet" in dp:
                split_idx = int(len(self.list_data_dict[i]) * 0.98)
                self.list_data_dict[i] = self.list_data_dict[i][:split_idx]

        self.focal_length_range = focal_length_range
        self.sample_weights = [int(x) for x in sample_weights.split(";")] if sample_weights else [1] * len(self.list_data_dict)

    def __len__(self) -> int:
        return sum(len(d) * 100 for d in self.list_data_dict) 

    def __getitem__(self, index):
        dataset_idx = random.choices(range(len(self.list_data_dict)), weights=self.sample_weights, k=1)[0]
        img_idx = index % len(self.list_data_dict[dataset_idx])
        
        pixel_coords = self.list_data_dict[dataset_idx][img_idx].get("pixel_coords", [])
        if not pixel_coords:
            return self.__getitem__((index + 1) % self.__len__())
            
        point_idx = random.randint(0, len(pixel_coords) - 1)
        
        # Calculate scaling factor
        is_augmented = (index % 2 != 0)
        scale_factor = random.uniform(self.focal_length_range[0], self.focal_length_range[1]) if is_augmented else 1.0

        try:
            return self._process_sample(dataset_idx, img_idx, point_idx, scale_factor, self.apply_scaling, is_eval=False)
        except Exception as e:
            return self.__getitem__((index + 1) % self.__len__())


class DatasetEval(BaseDepthDataset):
    # Added points_per_image=10 (100 images * 10 points = 1000 evaluations)
    def __init__(self, data_path: str, image_folder: str, fixed_scale_factor=1.0, apply_scaling=False, points_per_image=100):
        super().__init__(data_path, image_folder)
        self.fixed_scale_factor = fixed_scale_factor
        self.apply_scaling = apply_scaling
        self.points_per_image = points_per_image
        
        for i, dp in enumerate(self.data_paths):
            if "scannet" in dp:
                split_idx = int(len(self.list_data_dict[i]) * 0.98)
                self.list_data_dict[i] = self.list_data_dict[i][split_idx:]
                
            random.seed(42)
            random.shuffle(self.list_data_dict[i])

        self.random_indices = []
        random.seed(42) 
        
        for d in self.list_data_dict:
            dataset_indices = []
            for item in d:
                num_points = len(item.get("pixel_coords", [0]))
                # Generate a list of multiple random points for this specific image
                points = [random.randint(0, max(0, num_points - 1)) for _ in range(self.points_per_image)]
                dataset_indices.append(points)
            self.random_indices.append(dataset_indices)

    def __len__(self) -> int:
        # Multiply by points_per_image to expand the dataset size
        return sum(len(d) * self.points_per_image for d in self.list_data_dict)

    def __getitem__(self, index):
        # Calculate which image and which sub-point this index belongs to
        img_index_flat = index // self.points_per_image
        point_sub_idx = index % self.points_per_image

        dataset_idx = 0
        img_idx = img_index_flat
        for i, d in enumerate(self.list_data_dict):
            if img_idx < len(d):
                dataset_idx = i
                break
            img_idx -= len(d)
            
        # Retrieve the specific pre-selected random point
        point_idx = self.random_indices[dataset_idx][img_idx][point_sub_idx]

        try:
            return self._process_sample(dataset_idx, img_idx, point_idx, scale_factor=self.fixed_scale_factor, apply_scaling=self.apply_scaling, is_eval=True)
        except Exception as e:
            return self.__getitem__((index + 1) % self.__len__())
