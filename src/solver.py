from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, pi, sin, tan

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .astronomy import altaz_to_vector


@dataclass
class SolveResult:
    success: bool
    center_az: float
    center_alt: float
    roll: float
    hfov: float
    score: float
    inliers: int
    tolerance: float
    projected: np.ndarray


def _camera_basis(az_deg: float, alt_deg: float, roll_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = np.deg2rad(az_deg)
    alt = np.deg2rad(alt_deg)
    f = np.array([cos(alt)*sin(az), cos(alt)*cos(az), sin(alt)], dtype=float)
    e = np.array([cos(az), -sin(az), 0.0], dtype=float)
    n = np.array([-sin(alt)*sin(az), -sin(alt)*cos(az), cos(alt)], dtype=float)
    r = np.deg2rad(roll_deg)
    right = cos(r) * e + sin(r) * n
    up = -sin(r) * e + cos(r) * n
    return f, right, up


def project_vectors(vectors: np.ndarray, az_deg: float, alt_deg: float, roll_deg: float, hfov: float, aspect: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    f, right, up = _camera_basis(az_deg, alt_deg, roll_deg)
    z = vectors @ f
    x = vectors @ right
    y = vectors @ up
    visible = z > 1e-5
    vfov = 2.0 * np.arctan(np.tan(np.deg2rad(hfov)/2.0) * aspect)
    px = (x / np.maximum(z, 1e-8)) / np.tan(np.deg2rad(hfov)/2.0)
    py = (y / np.maximum(z, 1e-8)) / np.tan(vfov/2.0)
    visible &= (np.abs(px) <= 1.0) & (np.abs(py) <= 1.0)
    return np.column_stack([px, py]), visible


def _match_score(detections: np.ndarray, projected: np.ndarray, tolerance: float) -> tuple[int, float]:
    if len(detections) == 0 or len(projected) == 0:
        return 0, 0.0
    dist = np.linalg.norm(detections[:, None, :] - projected[None, :, :], axis=2)
    large = 999.0
    cost = np.where(dist <= tolerance, dist, large)
    rows, cols = linear_sum_assignment(cost)
    good = cost[rows, cols] < large
    d = cost[rows, cols][good]
    inliers = int(np.sum(good))
    quality = float(np.sum(np.exp(-((d / max(tolerance, 1e-5)) ** 2)))) if len(d) else 0.0
    return inliers, quality


def _search(catalog_vectors: np.ndarray, detections: np.ndarray, candidates: list[tuple[float,float,float,float]], tolerance: float, aspect: float) -> tuple[tuple[float,float,float,float] | None, float, int]:
    best = None
    best_score = -1e9
    best_inliers = 0
    for az, alt, roll, hfov in candidates:
        projected, visible = project_vectors(catalog_vectors, az, alt, roll, hfov, aspect)
        projected = projected[visible]
        if len(projected) == 0:
            continue
        inliers, quality = _match_score(detections, projected, tolerance)
        # Reward many independent matches, but also retain a continuous quality term.
        score = 4.0 * inliers + quality
        if score > best_score:
            best_score = score
            best = (az, alt, roll, hfov)
            best_inliers = inliers
    return best, best_score, best_inliers


def solve_star_field(altaz_deg: np.ndarray, detections_xy: np.ndarray, image_aspect: float = 0.75, max_candidates: int = 120, hfov_hint: float = 65.0) -> SolveResult:
    visible = altaz_deg[:, 1] > 5.0
    altaz = altaz_deg[visible]
    vectors = altaz_to_vector(altaz[:, 0], altaz[:, 1])
    if len(detections_xy) < 3 or len(vectors) < 5:
        return SolveResult(False, 0, 0, 0, 60, 0, 0, 0.06, np.empty((0,2)))

    # Use the brightest catalog stars only for the pose search; full catalog is used afterward.
    vectors = vectors[:max_candidates]
    coarse = []
    for alt in range(15, 86, 10):
        for az in range(0, 360, 10):
            for roll in range(0, 360, 30):
                for hfov in (max(35.0, hfov_hint-15.0), max(35.0, hfov_hint-5.0), hfov_hint, min(120.0, hfov_hint+10.0), min(120.0, hfov_hint+15.0)):
                    coarse.append((float(az), float(alt), float(roll), float(hfov)))
    best, score, inliers = _search(vectors, detections_xy, coarse, 0.085, image_aspect)
    if best is None:
        return SolveResult(False, 0, 0, 0, 60, score, inliers, 0.085, np.empty((0,2)))

    az0, alt0, roll0, fov0 = best
    fine = []
    for alt in np.arange(max(5, alt0-10), min(90, alt0+10.1), 2.5):
        for az in np.arange(az0-10, az0+10.1, 2.5):
            for roll in np.arange(roll0-20, roll0+20.1, 5):
                for hfov in np.arange(max(35, fov0-7.5), min(110, fov0+7.6), 2.5):
                    fine.append((float(az % 360), float(alt), float(roll % 360), float(hfov)))
    best, score, inliers = _search(vectors, detections_xy, fine, 0.065, image_aspect)
    if best is None:
        return SolveResult(False, 0, 0, 0, 60, score, inliers, 0.065, np.empty((0,2)))
    az0, alt0, roll0, fov0 = best
    projected, _ = project_vectors(vectors, az0, alt0, roll0, fov0, image_aspect)
    return SolveResult(True, az0, alt0, roll0, fov0, score, inliers, 0.065, projected)


def project_catalog_to_image(altaz_deg: np.ndarray, result: SolveResult, image_size: tuple[int,int]) -> tuple[np.ndarray, np.ndarray]:
    vectors = altaz_to_vector(altaz_deg[:,0], altaz_deg[:,1])
    aspect = image_size[1] / max(image_size[0], 1)
    proj, visible = project_vectors(vectors, result.center_az, result.center_alt, result.roll, result.hfov, aspect)
    w, h = image_size
    if len(proj):
        px = (proj[:,0] + 1) * 0.5 * (w-1)
        py = (1 - proj[:,1]) * 0.5 * (h-1)
        proj_px = np.column_stack([px, py])
        proj_px[~visible] = np.nan
    else:
        proj_px = np.empty((0,2))
    return proj_px, visible


def match_detections_to_catalog(detections_xy: np.ndarray, catalog_points_px: np.ndarray, tolerance_px: float) -> list[tuple[int,int,float]]:
    if len(detections_xy) == 0 or len(catalog_points_px) == 0:
        return []
    dist = np.linalg.norm(detections_xy[:,None,:] - catalog_points_px[None,:,:], axis=2)
    dist[~np.isfinite(dist)] = 1e9
    rows, cols = linear_sum_assignment(dist)
    matches = []
    for r,c in zip(rows,cols):
        if dist[r,c] <= tolerance_px:
            matches.append((int(r),int(c),float(dist[r,c])))
    return matches
