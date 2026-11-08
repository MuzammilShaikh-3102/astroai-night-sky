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


def test_projection_uses_width_over_height_aspect_ratio():
    # A 60-degree horizontal field of view on a 2:1 landscape image must
    # have a narrower vertical field of view than horizontal field of view.
    hfov = 60.0
    aspect = 2.0
    vfov = 2.0 * np.arctan(np.tan(np.deg2rad(hfov) / 2.0) / aspect)
    assert 0 < np.rad2deg(vfov) < hfov


def test_catalog_projection_uses_same_aspect_convention():
    from src.solver import SolveResult, project_catalog_to_image
    from src.astronomy import altaz_to_vector

    # With a centered camera looking north at altitude 45°, a star at the
    # camera center should project to the image center regardless of aspect.
    result = SolveResult(True, 0.0, 45.0, 0.0, 60.0, 1.0, 5, 0.05, np.empty((0, 2)))
    altaz = np.array([[0.0, 45.0]])
    projected, visible = project_catalog_to_image(altaz, result, (1200, 600))
    assert visible[0]
    assert np.allclose(projected[0], [599.5, 299.5], atol=1e-5)
