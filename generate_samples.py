"""
generate_samples.py
-------------------
Generates realistic synthetic endoscopic frames saved to sample_data/.
Run once before launching the main application.
"""

import os
import numpy as np
import cv2

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

W, H = 640, 480
NUM_FRAMES = 80
np.random.seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: draw a circular vignette mask (mimics the endoscope barrel)
# ──────────────────────────────────────────────────────────────────────────────
def make_vignette(h, w, strength=0.85):
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    mask = 1.0 - strength * (dist / max_dist) ** 2
    return np.clip(mask, 0, 1).astype(np.float32)


VIGNETTE = make_vignette(H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Generate a single endoscopic-looking frame
# ──────────────────────────────────────────────────────────────────────────────
def generate_frame(frame_idx: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.float32)

    # --- base tissue colour (pinkish-red mucosa) ---
    base_r = np.random.uniform(160, 200)
    base_g = np.random.uniform(60, 100)
    base_b = np.random.uniform(60, 90)

    # Smooth noise base (Perlin-like via Gaussian blur of random noise)
    noise = np.random.rand(H, W).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (61, 61), 20)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)

    img[:, :, 2] = base_r + 30 * noise          # R
    img[:, :, 1] = base_g + 15 * noise          # G
    img[:, :, 0] = base_b + 10 * noise          # B

    # --- mucosal folds (sinusoidal ridges) ---
    shift = frame_idx * 3
    for k in range(1, 5):
        freq = 0.02 * k
        phase = np.pi * k / 3 + shift * 0.05
        fold = 20 * np.sin(freq * np.arange(W) + phase)
        for row in range(H):
            offset = int(fold[int(row * W / H) % W] if row < H else 0)
            col = (np.arange(W) + offset) % W
            img[row, :, 2] += 15 * np.sin(freq * col + phase)

    # --- polyp-like circular lesion (appears in ~40 % of frames) ---
    if frame_idx % 5 < 2:
        cx = int(W * 0.35 + 80 * np.sin(frame_idx * 0.1))
        cy = int(H * 0.45 + 40 * np.cos(frame_idx * 0.13))
        r  = int(30 + 10 * np.sin(frame_idx * 0.2))
        cv2.circle(img, (cx, cy), r,     (70,  90, 190), -1)   # base
        cv2.circle(img, (cx, cy), r - 5, (80, 100, 210), -1)   # highlight
        cv2.circle(img, (cx - 6, cy - 6), r // 4, (120, 140, 240), -1)  # specular

    # --- blood vessel lines ---
    for _ in range(np.random.randint(3, 8)):
        x0 = np.random.randint(0, W)
        y0 = np.random.randint(0, H)
        x1 = x0 + np.random.randint(-120, 120)
        y1 = y0 + np.random.randint(-120, 120)
        thickness = np.random.randint(1, 3)
        cv2.line(img, (x0, y0), (x1, y1), (30, 30, 150), thickness)

    # --- specular highlights ---
    for _ in range(np.random.randint(2, 6)):
        sx = np.random.randint(50, W - 50)
        sy = np.random.randint(50, H - 50)
        sr = np.random.randint(4, 12)
        cv2.ellipse(img, (sx, sy), (sr, sr // 2), np.random.randint(0, 360),
                    0, 360, (230, 230, 230), -1)

    # --- apply vignette ---
    for c in range(3):
        img[:, :, c] *= VIGNETTE

    # Clip & convert
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Optional mild Gaussian blur for realism
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[generate_samples] Writing {NUM_FRAMES} frames to '{OUTPUT_DIR}' ...")
    for i in range(NUM_FRAMES):
        frame = generate_frame(i)
        path = os.path.join(OUTPUT_DIR, f"endo_{i:04d}.png")
        cv2.imwrite(path, frame)
    print("[generate_samples] Done.")
