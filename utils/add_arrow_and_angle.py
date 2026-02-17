
import cv2
import numpy as np


def overlay_sniper_scope_blocky_arrow(
    image_path,
    output_path,
    fx,
    fy,
    cx,
    cy,
    target_u,
    target_v,
    alpha_bg=0.3,
):
    """
    Overlays a 'Sniper Scope' visualization with a specific thick,
    blocky red arrow pointing to the target.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    h, w = img.shape[:2]
    overlay = img.copy()

    # Define Colors
    color_bg_ring = (0, 255, 255)  # Yellow (BGR)
    color_target = (0, 0, 255)  # Red (BGR) - for the arrow
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---------------------------------------------------------
    # PART 1: Background Context Grid (The "Ruler")
    # ---------------------------------------------------------
    standard_angles = [5, 10, 15, 20, 30, 45]
    for deg in standard_angles:
        theta_rad = np.deg2rad(deg)
        ax_x = int(fx * np.tan(theta_rad))
        ax_y = int(fy * np.tan(theta_rad))
        if ax_x < w * 1.5:
            cv2.ellipse(
                overlay, (int(cx), int(cy)), (ax_x, ax_y), 0, 0, 360, color_bg_ring, 1
            )
            cv2.putText(
                overlay,
                f"{deg}d",
                (int(cx) + 5, int(cy) - ax_y + 5),
                font,
                0.4,
                color_bg_ring,
                1,
            )

    # ---------------------------------------------------------
    # PART 2: The Target "Sniper" Ring
    # ---------------------------------------------------------
    dx = target_u - cx
    dy = target_v - cy
    tan_theta = np.sqrt((dx / fx) ** 2 + (dy / fy) ** 2)
    theta_deg = np.degrees(np.arctan(tan_theta))

    # ---------------------------------------------------------
    # PART 3: Blocky Target Arrow (NEW Implementation)
    # ---------------------------------------------------------
    tx, ty = target_u, target_v

    # 1. Define Arrow Geometry (pointing RIGHT along X-axis, tip at 0,0)
    # Adjust these values to change the arrow's blockiness/size
    head_len = 50.0
    head_width_half = 30.0
    shaft_len = 70.0
    shaft_width_half = 12.0

    # Define the vertices of the arrow shape relative to its tip (0,0)
    arrow_pts_base = np.array(
        [
            [0.0, 0.0],  # Tip
            [-head_len, -head_width_half],  # Head Top Corner
            [-head_len, -shaft_width_half],  # Shaft Neck Top
            [-head_len - shaft_len, -shaft_width_half],  # Shaft Tail Top
            [-head_len - shaft_len, shaft_width_half],  # Shaft Tail Bottom
            [-head_len, shaft_width_half],  # Shaft Neck Bottom
            [-head_len, head_width_half],  # Head Bottom Corner
        ],
        dtype=np.float32,
    )

    # 2. Determine Rotation Angle
    # To mimic the reference image, let's make the arrow originate from
    # the top-left relative to the target and point down-right inward.
    # Vector pointing FROM tail TO tip:
    vec_x, vec_y = 1, 1  # A simple diagonal direction
    angle_rad = np.arctan2(vec_y, vec_x)

    # 3. Create Rotation Matrix
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]], dtype=np.float32)

    # 4. Rotate and Translate Points
    # Rotate the base shape
    pts_rotated = arrow_pts_base @ rot_matrix.T
    # Translate so the tip (index 0) sits at the target coordinate
    pts_final = pts_rotated + np.array([tx, ty], dtype=np.float32)
    # Convert to integer coordinates for drawing
    pts_final_int = pts_final.astype(np.int32)

    # 5. Draw Filled Polygon
    cv2.fillPoly(overlay, [pts_final_int], color_target)

    # Add Label near the tail of the arrow
    label_text = f"Angle: {theta_deg:.1f}d"
    # Find the tail point (index 3 or 4 in our definition) for label placement
    tail_pt = pts_final_int[3]
    text_pos = (tail_pt[0] - 120, tail_pt[1])  # Offset text to the left of tail

    (tw, th), _ = cv2.getTextSize(label_text, font, 0.6, 2)
    cv2.rectangle(
        overlay,
        (text_pos[0] - 2, text_pos[1] - th - 2),
        (text_pos[0] + tw + 2, text_pos[1] + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(overlay, label_text, text_pos, font, 0.6, color_bg_ring, 2)

    # ---------------------------------------------------------
    # PART 4: Blending
    # ---------------------------------------------------------
    final_img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    cv2.imwrite(output_path, final_img)
    print(f"[OK] Saved with blocky arrow: {output_path}")
