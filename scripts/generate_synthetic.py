from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STARS = pd.read_csv(ROOT / "data" / "stars_seed.csv")
CONSTS = json.loads((ROOT / "data" / "constellations.json").read_text())


def render_constellation(name: str, size: int = 128, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    img = rng.normal(0, 4, (size, size)).astype(np.float32)
    star_names = sorted({x for edge in CONSTS[name] for x in edge})
    pts = STARS[STARS.name.isin(star_names)].copy()
    if len(pts) < 2:
        return np.clip(img, 0, 255).astype(np.uint8)

    # Build constellation coordinates from RA/Dec, normalize, random affine transform.
    x = pts.ra_h.to_numpy(float) * 15.0
    y = pts.dec_deg.to_numpy(float)
    x = (x - x.mean()) * np.cos(np.deg2rad(y.mean()))
    y = y - y.mean()
    scale = rng.uniform(2.3, 4.8)
    theta = rng.uniform(0, 2*np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = np.column_stack([x,y]) @ R.T * scale
    xy += rng.uniform(-12,12,size=2) + np.array([size/2,size/2])

    # Add unrelated background stars to avoid a trivial blank-background classification task.
    n_bg = rng.integers(10, 35)
    bgxy = rng.uniform(4, size-4, size=(n_bg,2))
    for bx,by in bgxy:
        r = rng.uniform(0.3,1.2)
        yy,xx=np.ogrid[:size,:size]
        img += rng.uniform(5,15)*np.exp(-((xx-bx)**2+(yy-by)**2)/(2*r*r))

    for (px,py), mag in zip(xy, pts.mag.to_numpy(float)):
        if px < -5 or py < -5 or px > size+5 or py > size+5:
            continue
        radius = max(0.7, min(2.2, 2.5 - 0.35*mag))
        yy,xx=np.ogrid[:size,:size]
        strength = 140*np.exp(-0.42*mag)
        img += strength*np.exp(-((xx-px)**2+(yy-py)**2)/(2*radius*radius))

    sigma = rng.uniform(0.4, 1.5)
    img = cv2.GaussianBlur(img, (0,0), sigma)
    if rng.random() < 0.5:
        img = np.clip(img + rng.normal(0,5,img.shape),0,255)
    return np.clip(img,0,255).astype(np.uint8)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT/"data"/"synthetic"))
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--size", type=int, default=128)
    args=ap.parse_args()
    out=Path(args.out)
    rng=np.random.default_rng(2026)
    for name in CONSTS:
        cdir=out/name.replace(" ","_")
        cdir.mkdir(parents=True,exist_ok=True)
        for i in range(args.per_class):
            arr=render_constellation(name,args.size,rng)
            cv2.imwrite(str(cdir/f"{i:05d}.png"),arr)
    print(f"Generated {len(CONSTS)*args.per_class} images in {out}")

if __name__ == "__main__":
    main()
