import cv2
import numpy as np
import cv2
import numpy as np

def generate_clean_visual_sample(
    image_path, output_path, fx, fy, cx, cy, target_u, target_v
):
    img = cv2.imread(image_path)
    if img is None: return None
    
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # --- IMPROVEMENT 1: The "Sniper Scope" (Non-intrusive Ruler) ---
    # Instead of circles everywhere, draw ticks on the axes passing through center
    color_ruler = (0, 255, 255) # Yellow
    
    # Draw Crosshair at Principal Point (Optical Center)
    cv2.line(overlay, (int(cx)-20, int(cy)), (int(cx)+20, int(cy)), color_ruler, 1)
    cv2.line(overlay, (int(cx), int(cy)-20), (int(cx), int(cy)+20), color_ruler, 1)
    
    # Draw Ticks for standard angles (5, 10, 15...) ONLY on the axes
    standard_angles = [5, 10, 15, 20, 30, 45]
    for deg in standard_angles:
        pixels = int(fx * np.tan(np.deg2rad(deg)))
        
        # Horizontal Axis Ticks (Left/Right of center)
        for sign in [-1, 1]:
            px = int(cx + (pixels * sign))
            if 0 <= px < w:
                # Small vertical tick
                cv2.line(overlay, (px, int(cy)-5), (px, int(cy)+5), color_ruler, 2)
                # Add text only for major angles (e.g., 20, 45) to reduce clutter
                if deg in [20, 45]:
                    cv2.putText(overlay, f"{deg}", (px-10, int(cy)+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_ruler, 1)

    # --- IMPROVEMENT 2: The "Focal Bar" (Direct f visualization) ---
    # Draw a bar in bottom-left that represents 45 degrees FoV.
    # Length = f * tan(45) = f pixels.
    # If the bar is long, model knows it's zoomed in. If short, wide angle.
    bar_len = int(fx) 
    # Clamp bar to image width so it doesn't crash, but typically f ~ w
    bar_len = min(bar_len, w - 50) 
    
    start_x, start_y = 20, h - 30
    end_x = start_x + bar_len
    
    # Draw the bar
    cv2.line(overlay, (start_x, start_y), (end_x, start_y), (0, 255, 0), 4)
    cv2.putText(overlay, "Ref: 45 deg", (start_x, start_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- PART 3: The Target (Keep your existing Red Ring/Arrow logic) ---
    # Your logic here was good. It explicitly tells the model "This is the pixel".
    # (Copying your arrow/ring code here...)
    color_target = (0, 0, 255)
    cv2.arrowedLine(overlay, (int(target_u)-50, int(target_v)-50), (int(target_u), int(target_v)), color_target, 3)
    cv2.circle(overlay, (int(target_u), int(target_v)), 5, color_target, -1)

    # Save
    cv2.imwrite(output_path, overlay)
    
def generate_training_sample(
    image_path, output_path, fx, fy, cx, cy, target_u, target_v
):
    """
    Generates a dataset image for MLLM Depth/Angle estimation.

    Features:
    1. CONTEXT GRID: Yellow rings at 5, 10, 15 deg (The "Ruler").
    2. TARGET RING: Red ring at the specific target angle.
    3. POINTER: Blocky arrow pointing radially inward to the pixel.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]

    # ---------------------------------------------------------
    # PART 1: Background Context Grid (The Reference Scale)
    # ---------------------------------------------------------
    color_grid = (0, 255, 255)  # Yellow
    standard_angles = [5, 10, 15, 20, 30, 45]

    for deg in standard_angles:
        theta_rad = np.deg2rad(deg)
        ax_x = int(fx * np.tan(theta_rad))
        ax_y = int(fy * np.tan(theta_rad))

        # Only draw if it fits reasonably within the view
        if ax_x < w * 2:
            cv2.ellipse(
                overlay,
                (int(cx), int(cy)),
                (ax_x, ax_y),
                0,
                0,
                360,
                color_grid,
                1,
                cv2.LINE_AA,
            )

            # We keep these labels because they are the "Ruler"
            # The AI uses these numbers to estimate the red ring's value
            label_pos_y = int(cy) - ax_y + 10
            if 0 < label_pos_y < h:
                cv2.putText(
                    overlay,
                    f"{deg}d",
                    (int(cx) + 5, label_pos_y),
                    font,
                    0.4,
                    color_grid,
                    1,
                    cv2.LINE_AA,
                )

    # ---------------------------------------------------------
    # PART 2: The Target Geometry
    # ---------------------------------------------------------
    dx = target_u - cx
    dy = target_v - cy

    # Calculate the exact angle (Ground Truth)
    # Note: We calculate this for the ring size, but we do NOT print it as text.
    dist_from_center = np.sqrt((dx / fx) ** 2 + (dy / fy) ** 2)
    tan_theta = dist_from_center
    theta_deg = np.degrees(np.arctan(tan_theta))

    target_ax_x = int(fx * tan_theta)
    target_ax_y = int(fy * tan_theta)

    color_target = (0, 0, 255)  # Red

    # Draw the specific target ring (The "Question")
    cv2.ellipse(
        overlay,
        (int(cx), int(cy)),
        (target_ax_x, target_ax_y),
        0,
        0,
        360,
        color_target,
        1,
        cv2.LINE_AA,
    )

    # ---------------------------------------------------------
    # PART 3: Dynamic Blocky Arrow (Radially Aligned)
    # ---------------------------------------------------------
    # We want the arrow to point FROM the outside IN towards the target pixel.
    # This ensures the arrow aligns with the "rays" of the camera.

    # 1. Calculate direction vector from Center -> Target
    vec_x, vec_y = dx, dy
    mag = np.sqrt(vec_x**2 + vec_y**2)
    if mag == 0:
        mag = 1

    # Unit vector pointing to target
    ux, uy = vec_x / mag, vec_y / mag

    # 2. Define Arrow Shape (pointing "Right" initially)
    # Tip is at (0,0)
    head_len = 50.0
    head_width_half = 25.0
    shaft_len = 60.0
    shaft_width_half = 12.0

    arrow_pts_base = np.array(
        [
            [0.0, 0.0],  # Tip
            [-head_len, -head_width_half],  # Head Top
            [-head_len, -shaft_width_half],  # Shaft Top
            [-head_len - shaft_len, -shaft_width_half],  # Shaft Tail Top
            [-head_len - shaft_len, shaft_width_half],  # Shaft Tail Bottom
            [-head_len, shaft_width_half],  # Shaft Neck Bottom
            [-head_len, head_width_half],  # Head Bottom
        ],
        dtype=np.float32,
    )

    # 3. Calculate Rotation Angle
    # We want the arrow pointing along the vector (ux, uy)
    angle_rad = np.arctan2(uy, ux)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]], dtype=np.float32)

    # 4. Transform Points
    # Rotate
    pts_rotated = arrow_pts_base @ rot_matrix.T

    # Translate to target pixel
    # Move the tip exactly to (target_u, target_v)
    pts_final = pts_rotated + np.array([target_u, target_v], dtype=np.float32)
    pts_final_int = pts_final.astype(np.int32)

    # Draw Filled Arrow
    cv2.fillPoly(overlay, [pts_final_int], color_target, lineType=cv2.LINE_AA)

    # ---------------------------------------------------------
    # PART 4: Output Generation
    # ---------------------------------------------------------
    # Blend: Overlay is opaque, but we want to see the background clearly.
    # Because we drew on a copy of the image, we can just save 'overlay'.
    # If you want the lines to be slightly transparent, use addWeighted.
    # For training data, clear, solid lines are usually better than transparent ones.

    cv2.imwrite(output_path, overlay)

    # Return the metadata for your JSONL training file
    return {
        "image_path": output_path,
        "ground_truth_angle": theta_deg,
        "prompt": "Estimate the angle of the object pointed to by the red arrow relative to the camera center, using the yellow rings as reference.",
    }