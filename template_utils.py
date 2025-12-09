"""Shared utilities for template-aware geometry processing."""
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

DEFAULT_MAX_DIST_ON_TEMPLATE = 30  # mm


def filter_points_near_template(
    points: np.ndarray,
    template_points: np.ndarray,
    max_dist: float = DEFAULT_MAX_DIST_ON_TEMPLATE,
    return_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Filter ``points`` to those lying within ``max_dist`` of the template surface.

    Args:
        points: (N, 3) array of 3D points sampled from NURBS surfaces.
        template_points: (M, 3) reference points sampled from the average template surface.
        max_dist: Maximum allowed Euclidean distance (in mm) from each point to the
            nearest template point.
        return_mask: If True, also return the boolean mask of kept points.

    Returns:
        The filtered points (and optionally the boolean mask).

    Raises:
        RuntimeError: If no points remain after filtering.
        ValueError: If the provided point clouds are empty.
    """

    pts = np.asarray(points, dtype=float)
    tmpl = np.asarray(template_points, dtype=float)

    if pts.size == 0:
        raise ValueError("Cannot filter empty NURBS points array.")
    if tmpl.size == 0:
        raise ValueError("Template point cloud is empty; cannot build KDTree.")

    tree = cKDTree(tmpl)
    dists, _ = tree.query(pts, k=1)
    mask = dists < float(max_dist)
    filtered = pts[mask]

    if filtered.shape[0] == 0:
        raise RuntimeError(
            "All NURBS samples were filtered out by distance; "
            "try increasing max_dist_on_template or check alignment."
        )

    if return_mask:
        return filtered, mask
    return filtered


def sample_template_points(
    mesh: o3d.geometry.TriangleMesh | Path | str,
    n_points: int = 20000,
) -> np.ndarray:
    """Sample points uniformly from a template mesh.

    Args:
        mesh: An Open3D TriangleMesh or path to a mesh file.
        n_points: Number of points to sample uniformly.
    """

    if isinstance(mesh, (str, Path)):
        mesh = o3d.io.read_triangle_mesh(str(mesh))
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise TypeError("mesh must be a TriangleMesh or path to one.")
    if not mesh.has_vertices():
        raise ValueError("Template mesh is empty; cannot sample points.")

    pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
    return np.asarray(pcd.points)
