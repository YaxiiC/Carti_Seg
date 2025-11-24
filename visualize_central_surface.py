#!/usr/bin/env python3
r"""
visualize_central_surface.py

Visualize the central femoral cartilage surface with per-vertex thickness colors.

Usage example (Windows CMD):

  python visualize_central_surface.py ^
      --mesh_path C:\Users\chris\MICCAI2026\Carti_Seg\femoral_cartilage_template_central.ply ^
      --thickness_path C:\Users\chris\MICCAI2026\Carti_Seg\femoral_cartilage_template_thickness.npy

Dependencies:
  pip install open3d numpy
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def simple_colormap_grayscale(values: np.ndarray) -> np.ndarray:
    """
    Map scalar values in [0, 1] to grayscale RGB colors.
    """
    v = np.clip(values, 0.0, 1.0)
    colors = np.stack([v, v, v], axis=1)  # (N, 3)
    return colors


def simple_colormap_blue_red(values: np.ndarray) -> np.ndarray:
    """
    Simple blue -> cyan -> yellow -> red colormap without extra dependencies.
    Input: values in [0, 1]
    """
    v = np.clip(values, 0.0, 1.0)
    # Piecewise linear colormap
    colors = np.zeros((len(v), 3), dtype=np.float32)
    # 0   -> blue  (0,0,1)
    # 0.5 -> cyan  (0,1,1)
    # 1   -> red   (1,0,0)
    lower = v <= 0.5
    upper = v > 0.5

    # 0..0.5: blue -> cyan
    t = np.zeros_like(v)
    t[lower] = v[lower] / 0.5  # 0..1
    colors[lower, 0] = 0.0
    colors[lower, 1] = t[lower]
    colors[lower, 2] = 1.0

    # 0.5..1: cyan -> red
    t[upper] = (v[upper] - 0.5) / 0.5  # 0..1
    colors[upper, 0] = t[upper]
    colors[upper, 1] = 1.0 - t[upper]
    colors[upper, 2] = 1.0 - t[upper]

    return colors


def visualize_central_surface(mesh_path: Path,
                              thickness_path: Path,
                              colormap: str = "gray"):
    # Load mesh
    print(f"Loading mesh from: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertices():
        raise RuntimeError("Loaded mesh has no vertices. Check the mesh path.")

    # Load thickness
    print(f"Loading thickness from: {thickness_path}")
    thickness = np.load(str(thickness_path))
    thickness = thickness.astype(np.float32).ravel()

    n_vertices = len(mesh.vertices)
    if len(thickness) != n_vertices:
        raise RuntimeError(
            f"Thickness length ({len(thickness)}) does not match number of vertices ({n_vertices})."
        )

    # Normalize thickness to [0, 1] for coloring
    t_min = float(thickness.min())
    t_max = float(thickness.max())
    print(f"Thickness range: min = {t_min:.4f}, max = {t_max:.4f}")
    t_norm = (thickness - t_min) / (t_max - t_min + 1e-8)

    # Map to colors
    if colormap == "gray":
        colors = simple_colormap_grayscale(t_norm)
    elif colormap == "blue_red":
        colors = simple_colormap_blue_red(t_norm)
    else:
        raise ValueError(f"Unsupported colormap: {colormap}. Use 'gray' or 'blue_red'.")

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    print("Opening Open3D viewer. Use mouse to rotate/zoom, press 'Q' or ESC to quit.")
    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Femoral cartilage central surface (thickness-mapped)",
        width=1024,
        height=768,
        mesh_show_back_face=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize central femoral cartilage surface with thickness map."
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        required=True,
        help="Path to central surface .ply (e.g., femoral_cartilage_template_central.ply)",
    )
    parser.add_argument(
        "--thickness_path",
        type=str,
        required=True,
        help="Path to thickness .npy (e.g., femoral_cartilage_template_thickness.npy)",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="gray",
        choices=["gray", "blue_red"],
        help="Colormap for thickness visualization.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mesh_path = Path(args.mesh_path)
    thickness_path = Path(args.thickness_path)

    visualize_central_surface(
        mesh_path=mesh_path,
        thickness_path=thickness_path,
        colormap=args.colormap,
    )
