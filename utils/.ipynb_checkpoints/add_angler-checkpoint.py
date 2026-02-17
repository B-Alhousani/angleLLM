import cv2
import numpy as np


def overlay_angular_grid(
    image_path,
    output_path,
    fx,
    fy,
    cx,
    cy,
    dist_coeffs=None,
    ring_angles_deg=None,
    spoke_step_deg=30,
    alpha=0.6,
):
    """
    Overlays a geometrically-correct angular grid on an image using camera intrinsics.

    Purpose:
    - Visualize focal length and angular scale
    - Assist VLMs in understanding perspective and depth cues
    - NOT a depth estimator by itself

    Parameters
    ----------
    fx, fy : focal lengths in pixels
    cx, cy : principal point
    dist_coeffs : optional distortion coefficients (k1,k2,p1,p2[,k3])
    """

    # -----------------------------
    # 1. Load image
    # -----------------------------
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")

    h, w = img.shape[:2]

    # -----------------------------
    # 2. Optional undistortion
    # -----------------------------
    if dist_coeffs is not None:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
        img = cv2.undistort(img, K, dist_coeffs)

    overlay = img.copy()

    # -----------------------------
    # 3. Angle configuration
    # -----------------------------
    if ring_angles_deg is None:
        ring_angles_deg = [5, 10, 15, 20, 30, 45, 60]

    # Compute approximate FOV (rectilinear pinhole)
    fov_x = 2 * np.rad2deg(np.arctan(w / (2 * fx)))
    fov_y = 2 * np.rad2deg(np.arctan(h / (2 * fy)))
    max_fov = max(fov_x, fov_y)

    # Auto-sanity: large FOV warning zone
    safe_max_angle = min(0.45 * max_fov, 80)

    # -----------------------------
    # 4. Styling
    # -----------------------------
    color_ring = (0, 255, 255)  # Yellow
    color_spoke = (255, 255, 0)  # Cyan
    color_center = (0, 0, 255)  # Red
    font = cv2.FONT_HERSHEY_SIMPLEX

    # -----------------------------
    # 5. Draw angular rings (TRUE geometry)
    # -----------------------------
    for deg in ring_angles_deg:
        if deg > safe_max_angle:
            continue  # avoid pathological projection

        theta = np.deg2rad(deg)

        rx = int(fx * np.tan(theta))
        ry = int(fy * np.tan(theta))

        if rx < w * 1.5 and ry < h * 1.5:
            cv2.ellipse(overlay, (int(cx), int(cy)), (rx, ry), 0, 0, 360, color_ring, 2)

            label = str(deg)  # Just "5"

            # 1. Draw the number "5"
            (text_w, text_h), baseline = cv2.getTextSize(label, font, 0.55, 2)
            text_x = int(cx) + 6
            text_y = int(cy) - ry + 18
            cv2.putText(overlay, label, (text_x, text_y), font, 0.55, color_ring, 2)

            # 2. Draw a tiny circle for the degree symbol next to it
            degree_x = text_x + text_w + 3
            degree_y = text_y - text_h + 3  # Move up slightly
            cv2.circle(
                overlay, (degree_x, degree_y), 2, color_ring, 2
            )  # Radius 2 circle

    # -----------------------------
    # 6. Draw azimuth spokes
    # -----------------------------
    length = int(max(w, h) * 1.2)

    for angle in range(0, 360, spoke_step_deg):
        rad = np.deg2rad(angle)
        x2 = int(cx + length * np.cos(rad))
        y2 = int(cy + length * np.sin(rad))

        cv2.line(overlay, (int(cx), int(cy)), (x2, y2), color_spoke, 1)

    # -----------------------------
    # 7. Optical center marker
    # -----------------------------
    cv2.drawMarker(
        overlay,
        (int(cx), int(cy)),
        color_center,
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )

    # -----------------------------
    # 9. Blend overlay
    # -----------------------------
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # -----------------------------
    # 10. Save
    # -----------------------------
    cv2.imwrite(output_path, img)
    print(f"[OK] Angular grid saved to {output_path}")


# The Real Image path
img_path = "/Users/tehabsim/Downloads/corridor_01.png"

# 1. The "Wide Angle" Trap (Lies: says it's wide angle)
# We divide fx by 2. The grid will be small/steep.
overlay_angular_grid(img_path, "bench_wide.jpg", fx=280, fy=280, cx=361, cy=242)

# 2. The "True" Reference (Truth)
# The grid matches reality.
overlay_angular_grid(img_path, "bench_true.jpg", fx=559, fy=558, cx=361, cy=242)

# 3. The "Telephoto" Trap (Lies: says it's zoomed in)
# We multiply fx by 3. The grid will be massive/flat.
overlay_angular_grid(img_path, "bench_tele.jpg", fx=1500, fy=1500, cx=361, cy=242)

