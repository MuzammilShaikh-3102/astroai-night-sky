from __future__ import annotations

import numpy as np

from .astronomy import altaz_to_vector
from .solver import project_vectors, SolveResult


def planet_predictions_to_image(predictions: dict[str, tuple[float,float]], result: SolveResult, image_size: tuple[int,int]) -> dict[str, tuple[float,float,float]]:
    names = list(predictions)
    if not names:
        return {}
    az = np.array([predictions[n][0] for n in names], dtype=float)
    alt = np.array([predictions[n][1] for n in names], dtype=float)
    vec = altaz_to_vector(az, alt)
    aspect = image_size[1] / max(image_size[0], 1)
    proj, visible = project_vectors(vec, result.center_az, result.center_alt, result.roll, result.hfov, aspect)
    w,h = image_size
    out = {}
    for i,name in enumerate(names):
        if not visible[i]:
            continue
        x = (proj[i,0] + 1) * 0.5 * (w-1)
        y = (1 - proj[i,1]) * 0.5 * (h-1)
        out[name] = (float(x), float(y), float(alt[i]))
    return out
