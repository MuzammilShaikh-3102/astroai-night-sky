from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class StarPoint:
    x: float
    y: float
    score: float
    area: float


def load_rgb(image_bytes: bytes) -> Image.Image:
    import io
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def prepare_image(image: Image.Image, max_side: int = 1400) -> Image.Image:
    img = image.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def star_enhanced_gray(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Large-scale background subtraction suppresses sky glow while keeping point sources.
    background = cv2.GaussianBlur(gray, (0, 0), 21)
    high = cv2.subtract(gray, background)
    high = cv2.normalize(high, None, 0, 255, cv2.NORM_MINMAX)
    return high


def detect_stars(image: Image.Image, max_stars: int = 45) -> list[StarPoint]:
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    background = cv2.GaussianBlur(gray, (0, 0), 17)
    high = cv2.subtract(gray, background)
    high = cv2.GaussianBlur(high, (3, 3), 0)

    med = float(np.median(high))
    mad = float(np.median(np.abs(high.astype(np.float32) - med))) + 1e-6
    threshold = int(np.clip(med + 6.0 * mad, 18, 90))
    _, mask = cv2.threshold(high, threshold, 255, cv2.THRESH_BINARY)

    # Remove very thin structures and large saturated blobs.
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    h, w = gray.shape
    candidates: list[StarPoint] = []
    border = 0.02 * min(h, w)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cx, cy = centroids[i]
        if area < 1 or area > max(35, 0.0008 * h * w):
            continue
        if cx < border or cx > w - border or cy < border or cy > h - border:
            continue
        patch_x0 = max(0, int(cx - 2))
        patch_x1 = min(w, int(cx + 3))
        patch_y0 = max(0, int(cy - 2))
        patch_y1 = min(h, int(cy + 3))
        score = float(np.mean(gray[patch_y0:patch_y1, patch_x0:patch_x1]))
        # Reject obvious large landscape lights/lamps.
        if bw > 12 or bh > 12:
            continue
        candidates.append(StarPoint(float(cx), float(cy), score, float(area)))

    candidates.sort(key=lambda p: p.score, reverse=True)
    # Non-maximum suppression so one star is not represented by neighboring components.
    chosen: list[StarPoint] = []
    min_sep = max(4.0, 0.005 * min(w, h))
    for p in candidates:
        if all((p.x-q.x)**2 + (p.y-q.y)**2 >= min_sep**2 for q in chosen):
            chosen.append(p)
        if len(chosen) >= max_stars:
            break
    return chosen


def normalized_points(points: list[StarPoint], image: Image.Image) -> np.ndarray:
    w, h = image.size
    out = []
    for p in points:
        x = 2.0 * (p.x / max(w - 1, 1)) - 1.0
        y = 1.0 - 2.0 * (p.y / max(h - 1, 1))
        out.append([x, y, p.score])
    return np.asarray(out, dtype=np.float32) if out else np.empty((0, 3), dtype=np.float32)


def make_model_input(image: Image.Image, size: int = 64) -> np.ndarray:
    enhanced = star_enhanced_gray(image)
    enhanced = cv2.resize(enhanced, (size, size), interpolation=cv2.INTER_AREA)
    enhanced = enhanced.astype(np.float32) / 255.0
    return enhanced[None, ...]


def crop_patch(image: Image.Image, x: float, y: float, size: int = 32) -> np.ndarray:
    arr=np.asarray(image.convert("L"))
    half=size//2
    cx,cy=int(round(x)),int(round(y))
    patch=np.zeros((size,size),dtype=np.uint8)
    x0=max(0,cx-half); x1=min(arr.shape[1],cx+half)
    y0=max(0,cy-half); y1=min(arr.shape[0],cy+half)
    px0=half-(cx-x0); py0=half-(cy-y0)
    patch[py0:py0+(y1-y0),px0:px0+(x1-x0)]=arr[y0:y1,x0:x1]
    return patch
