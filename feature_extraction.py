"""
feature_extraction.py
---------------------
Implements the BONUS Feature Extraction requirement for the
Smart Endoscopic Assistance System.

Extracts three feature categories (all mandatory for full bonus marks):

A) SHAPE FEATURES
   Area, perimeter, circularity, solidity, bounding-box aspect ratio,
   number of detected contours (potential lesions).

B) COLOR FEATURES
   Per-channel mean & std, dominant colour (K-Means in LAB space),
   HSV histogram (visualised as a bar chart).

C) TEXTURE FEATURES
   GLCM-based Haralick descriptors: contrast, dissimilarity, homogeneity,
   energy, correlation, ASM.
   Also: Local Binary Pattern (LBP) histogram + entropy.

Engineering rationale is included for oral defence.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure  import shannon_entropy
from utils import get_logger

logger = get_logger("Features")


# ══════════════════════════════════════════════════════════════════════════════
# A) SHAPE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_shape_features(frame: np.ndarray) -> dict:
    """
    Detect contours and compute morphological shape descriptors.

    Rationale
    ---------
    Polyps and lesions exhibit characteristic shapes (near-circular,
    elevated bumps). Area, perimeter, and circularity help distinguish
    pathological from normal mucosal folds.

    Circularity = 4π × Area / Perimeter²  (1.0 = perfect circle)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Preprocess: denoise + edge-preserve for robust contour detection
    blurred  = cv2.GaussianBlur(gray, (5, 5), 1.2)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    results = {
        "num_contours":    len(contours),
        "largest_area":    0.0,
        "largest_perim":   0.0,
        "circularity":     0.0,
        "solidity":        0.0,
        "aspect_ratio":    0.0,
    }

    if not contours:
        return results

    # Largest contour
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    perim   = cv2.arcLength(largest, True)
    circ    = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0

    hull    = cv2.convexHull(largest)
    hull_a  = cv2.contourArea(hull)
    solidity = (area / hull_a) if hull_a > 0 else 0.0

    x, y, bw, bh = cv2.boundingRect(largest)
    aspect       = (bw / bh) if bh > 0 else 0.0

    results.update({
        "largest_area":  round(area,     2),
        "largest_perim": round(perim,    2),
        "circularity":   round(circ,     4),
        "solidity":      round(solidity, 4),
        "aspect_ratio":  round(aspect,   4),
    })
    return results


def draw_contours(frame: np.ndarray, min_area: float = 300) -> np.ndarray:
    """
    Draw detected contours (potential lesion boundaries) on a copy of *frame*.
    Contours are colour-coded by circularity: green (circular) / cyan (other).
    """
    out  = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        perim = cv2.arcLength(c, True)
        circ  = (4 * np.pi * area / perim ** 2) if perim > 0 else 0
        colour = (0, 220, 80) if circ > 0.6 else (0, 200, 255)
        cv2.drawContours(out, [c], -1, colour, 2)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# B) COLOR FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_color_features(frame: np.ndarray) -> dict:
    """
    Per-channel statistics and dominant-colour extraction.

    Rationale
    ---------
    Colour is a primary diagnostic cue in endoscopy:
      - Healthy mucosa: pink/salmon
      - Inflammation:   brighter red (increased vascularity)
      - Ischaemia:      pale / whitish
      - Adenoma:        often brownish / darker
    Dominant colour via K-Means in LAB space is perceptually uniform.
    """
    b, g, r = cv2.split(frame)
    features = {}
    for name, ch in (("R", r), ("G", g), ("B", b)):
        features[f"mean_{name}"]  = round(float(ch.mean()), 2)
        features[f"std_{name}"]   = round(float(ch.std()),  2)

    # Dominant colour (K=3 in LAB)
    lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    data     = lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, labels, centres = cv2.kmeans(data, 3, None, criteria, 3,
                                        cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        dominant_lab = centres[np.argmax(counts)]
        dom_bgr = cv2.cvtColor(
            np.array([[dominant_lab]], dtype=np.float32), cv2.COLOR_LAB2BGR
        )[0, 0]
        features["dominant_B"] = round(float(dom_bgr[0]), 1)
        features["dominant_G"] = round(float(dom_bgr[1]), 1)
        features["dominant_R"] = round(float(dom_bgr[2]), 1)
    except Exception:
        features["dominant_B"] = features["dominant_G"] = features["dominant_R"] = 0.0

    # Mean saturation in HSV (inflammation indicator)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    features["mean_saturation"] = round(float(hsv[:, :, 1].mean()), 2)
    features["mean_value"]      = round(float(hsv[:, :, 2].mean()), 2)
    return features


def hsv_histogram(frame: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Compute HSV histogram (H and S channels) for colour analysis.
    Returns a 1-D concatenated feature vector of length 2×bins.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_h /= (hist_h.sum() + 1e-6)
    hist_s /= (hist_s.sum() + 1e-6)
    return np.concatenate([hist_h, hist_s])


# ══════════════════════════════════════════════════════════════════════════════
# C) TEXTURE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_texture_features(frame: np.ndarray) -> dict:
    """
    GLCM Haralick descriptors + LBP entropy.

    Rationale
    ---------
    Texture differentiates normal mucosa from dysplastic/hyperplastic
    tissue.  GLCM captures pixel-pair spatial relationships; high contrast
    or low energy may indicate irregular surface.  LBP is rotation-invariant
    and effective for detecting micro-texture changes characteristic of
    pit-pattern classification (Kudo classification).

    GLCM parameters
    ---------------
    distances = [1, 2, 4]   : captures micro- and meso-texture
    angles    = 4 directions : rotation-invariant aggregate
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── GLCM ─────────────────────────────────────────────────────────────────
    gray8 = (gray // 4).astype(np.uint8)         # quantise to 64 levels
    glcm  = graycomatrix(gray8,
                         distances=[1, 2, 4],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=64,
                         symmetric=True,
                         normed=True)

    def _prop(prop):
        return round(float(graycoprops(glcm, prop).mean()), 5)

    haralick = {
        "glcm_contrast":     _prop("contrast"),
        "glcm_dissimilarity":_prop("dissimilarity"),
        "glcm_homogeneity":  _prop("homogeneity"),
        "glcm_energy":       _prop("energy"),
        "glcm_correlation":  _prop("correlation"),
        "glcm_ASM":          _prop("ASM"),
    }

    # ── LBP ──────────────────────────────────────────────────────────────────
    lbp     = local_binary_pattern(gray, P=8, R=1.0, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    entropy = float(shannon_entropy(hist + 1e-10))

    haralick["lbp_entropy"] = round(entropy, 5)
    return haralick


# ══════════════════════════════════════════════════════════════════════════════
# Combined extractor
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_features(frame: np.ndarray) -> dict:
    """
    Run shape, color, and texture extraction on *frame*.
    Returns a flat dictionary of all features.
    """
    out = {}
    try:
        out.update(extract_shape_features(frame))
    except Exception as e:
        logger.warning("Shape extraction failed: %s", e)

    try:
        out.update(extract_color_features(frame))
    except Exception as e:
        logger.warning("Color extraction failed: %s", e)

    try:
        out.update(extract_texture_features(frame))
    except Exception as e:
        logger.warning("Texture extraction failed: %s", e)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helper – feature panel
# ══════════════════════════════════════════════════════════════════════════════

def render_feature_panel(features: dict,
                          width: int = 320,
                          height: int = 420) -> np.ndarray:
    """
    Render a dark-themed text panel showing extracted feature values.
    Returns a BGR image of size (height, width).
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 18, 35)

    # Title bar
    cv2.rectangle(panel, (0, 0), (width, 28), (40, 80, 120), -1)
    cv2.putText(panel, "FEATURE EXTRACTION", (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)

    sections = {
        "SHAPE":   ["num_contours", "largest_area", "circularity",
                    "solidity", "aspect_ratio"],
        "COLOR":   ["mean_R", "mean_G", "mean_B",
                    "dominant_R", "dominant_G", "dominant_B",
                    "mean_saturation"],
        "TEXTURE": ["glcm_contrast", "glcm_homogeneity", "glcm_energy",
                    "glcm_correlation", "lbp_entropy"],
    }
    section_colours = {
        "SHAPE":   (50, 220, 120),
        "COLOR":   (50, 180, 255),
        "TEXTURE": (255, 180, 50),
    }

    y = 40
    for sec, keys in sections.items():
        # Section header
        cv2.putText(panel, sec, (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    section_colours[sec], 1, cv2.LINE_AA)
        y += 16
        for k in keys:
            if k not in features:
                continue
            val = features[k]
            if isinstance(val, float):
                txt = f"  {k}: {val:.4f}"
            else:
                txt = f"  {k}: {val}"
            cv2.putText(panel, txt, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200),
                        1, cv2.LINE_AA)
            y += 14
            if y > height - 16:
                break
        y += 4
        if y > height - 16:
            break

    return panel
