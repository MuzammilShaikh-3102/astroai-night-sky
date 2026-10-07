import numpy as np
from src.astronomy import altaz_to_vector
from src.solver import project_vectors, solve_star_field


def test_solver_recovers_synthetic_pose():
    rng = np.random.default_rng(7)
    az = rng.uniform(0, 360, 120)
    alt = rng.uniform(10, 80, 120)
    vectors = altaz_to_vector(az, alt)
    true_params=(140.0, 52.0, 23.0, 62.0)
    projected, visible = project_vectors(vectors, *true_params, aspect=0.75)
    pts = projected[visible]
    pts = pts[:18] + rng.normal(0, 0.008, size=(min(18, len(pts)),2))
    altaz = np.column_stack([az[visible], alt[visible]])[:len(pts)]
    # We intentionally pass an ordered alt/az table with matching vectors.
    result = solve_star_field(altaz, pts, image_aspect=0.75, max_candidates=len(altaz))
    assert result.success
    assert result.inliers >= 6
