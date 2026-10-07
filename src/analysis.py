from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .catalog import load_catalog, solver_catalog
from .vision import detect_stars, normalized_points
from .astronomy import star_altaz, planet_altaz, ensure_utc
from .solver import solve_star_field, project_catalog_to_image, match_detections_to_catalog
from .planets import planet_predictions_to_image
from .star_ml import load_model as load_star_model, score_patch
from .vision import crop_patch

ROOT = Path(__file__).resolve().parents[1]
CONST_FILE = ROOT / "data" / "constellations.json"
CON_MAP = {
    "CMa": "Canis Major", "CMi": "Canis Minor", "Ori": "Orion", "UMa": "Ursa Major", "UMi": "Ursa Minor",
    "Cas": "Cassiopeia", "Cyg": "Cygnus", "Lyr": "Lyra", "Aql": "Aquila", "Sco": "Scorpius", "Sgr": "Sagittarius",
    "Tau": "Taurus", "Gem": "Gemini", "Leo": "Leo", "And": "Andromeda", "Peg": "Pegasus", "Vir": "Virgo",
    "Boo": "Bootes", "Her": "Hercules", "Per": "Perseus", "Cet": "Cetus", "Phe": "Phoenix", "Cru": "Crux",
    "Car": "Carina", "Cen": "Centaurus", "Pav": "Pavo", "Gru": "Grus", "TrA": "Triangulum Australe", "Ser": "Serpens",
    "Oph": "Ophiuchus", "Lib": "Libra", "Dra": "Draco", "Lep": "Lepus", "Crv": "Corvus", "Cep": "Cepheus", "Pup": "Puppis",
    "Erid": "Eridanus", "Hya": "Hydra"
}


def constellation_edges() -> dict[str, list[tuple[str,str]]]:
    return json.loads(CONST_FILE.read_text(encoding="utf-8"))


def identify_constellation_from_matches(matches, catalog: pd.DataFrame) -> list[tuple[str,int]]:
    counts: dict[str,int] = {}
    for det_idx, cat_idx, dist in matches:
        row = catalog.iloc[cat_idx]
        name = str(row.get("display_name", ""))
        con = CON_MAP.get(str(row.get("constellation", "")))
        # Use the key-star map in the fallback catalog too.
        if not con:
            for cname, edges in constellation_edges().items():
                if any(name in pair for pair in edges):
                    con = cname
                    break
        if con:
            counts[con] = counts.get(con,0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


def analyze(image, dt_local: datetime, tz_offset_hours: float, lat: float, lon: float, catalog: pd.DataFrame | None = None, hfov_hint: float = 65.0, max_detected_stars: int = 45, use_ml: bool = True) -> dict:
    prepared = image.copy()
    detections = detect_stars(prepared, max_stars=max_detected_stars)
    star_model = load_star_model() if use_ml else None
    if star_model is not None:
        verified=[]
        for p in detections:
            sc=score_patch(star_model, crop_patch(prepared,p.x,p.y,32))
            if sc is None or sc >= 0.55:
                verified.append(p)
        detections = verified
    det_norm = normalized_points(detections, prepared)
    metadata = {"stars_detected": len(detections), "catalog_source": ""}
    if catalog is None:
        catalog, source = load_catalog(prefer_online=True)
    else:
        source = "provided"
    metadata["catalog_source"] = source

    result = {
        "detections": detections,
        "matches": [],
        "stars": [],
        "planets": [],
        "constellations": [],
        "solve": None,
        "metadata": metadata,
    }

    result["metadata"]["ml_star_verifier"] = star_model is not None
    if len(detections) < 4:
        result["warning"] = "Only a few bright points were detected. Try a darker sky or a longer night-mode exposure."
        return result

    dt_utc = ensure_utc(dt_local, tz_offset_hours)
    solve_cat = solver_catalog(catalog, max_stars=min(250, len(catalog)))
    altaz = star_altaz(solve_cat, dt_utc, lat, lon)
    solve = solve_star_field(altaz, det_norm[:, :2], image_aspect=prepared.size[1] / max(prepared.size[0], 1), max_candidates=min(120,len(solve_cat)), hfov_hint=float(hfov_hint))
    result["solve"] = solve
    if not solve.success or solve.inliers < 4:
        result["warning"] = "The sky pattern could not be solved confidently. Adjust the Field of View slider or use a photo with more visible stars."
    else:
        full_altaz = star_altaz(catalog, dt_utc, lat, lon)
        points_px, visible = project_catalog_to_image(full_altaz, solve, prepared.size)
        tolerance_px = max(7.0, 0.018 * max(prepared.size))
        matches = match_detections_to_catalog(np.asarray([[p.x,p.y] for p in detections]), points_px, tolerance_px)
        result["matches"] = matches
        for det_idx, cat_idx, dist in matches:
            row = catalog.iloc[cat_idx]
            if float(row["mag"]) > 5.5:
                continue
            confidence = float(np.clip(1.0 - dist / tolerance_px, 0.0, 1.0))
            result["stars"].append({
                "name": str(row["display_name"]),
                "magnitude": float(row["mag"]),
                "constellation": CON_MAP.get(str(row.get("constellation","")), str(row.get("constellation",""))),
                "distance_px": float(dist),
                "confidence": confidence,
                "x": float(detections[det_idx].x),
                "y": float(detections[det_idx].y),
            })
        result["stars"].sort(key=lambda x: (-x["confidence"], x["magnitude"]))
        result["constellations"] = identify_constellation_from_matches(matches, catalog)[:5]

        try:
            p_altaz = planet_altaz(dt_utc, lat, lon)
            p_points = planet_predictions_to_image(p_altaz, solve, prepared.size)
            known_star_xy = np.asarray([[s["x"],s["y"]] for s in result["stars"]], dtype=float)
            for name,(x,y,alt) in p_points.items():
                if alt <= 8:
                    continue
                if known_star_xy.size:
                    d = np.min(np.linalg.norm(known_star_xy - np.array([x,y]), axis=1))
                else:
                    d = 9999
                det_d = np.min(np.linalg.norm(np.asarray([[p.x,p.y] for p in detections]) - np.array([x,y]), axis=1))
                if det_d <= tolerance_px * 1.35 and d > tolerance_px * 0.55:
                    result["planets"].append({
                        "name": name, "confidence": float(np.clip(1.0-det_d/(tolerance_px*1.35),0,1)),
                        "x": float(x), "y": float(y), "altitude": float(alt)
                    })
        except Exception as exc:
            result["metadata"]["planet_error"] = str(exc)

    return result
