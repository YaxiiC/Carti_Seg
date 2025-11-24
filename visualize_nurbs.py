#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Visualize central template mesh and fitted NURBS surface (as point cloud),
optionally showing the NURBS control mesh, and filtering out NURBS points
far outside the anatomical region.

Usage example (Windows CMD):

  python visualize_nurbs.py ^
    --central_mesh  C:\Users\chris\MICCAI2026\Carti_Seg\femoral_cartilage_template_central.ply ^
    --template_npz  C:\Users\chris\MICCAI2026\Carti_Seg\femoral_template_surf.npz ^
    --bbox_margin   10.0 ^
    --n_u 150 --n_v 150 ^
    --show_ctrlmesh

Dependencies:
  pip install open3d numpy geomdl
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from geomdl import BSpline


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize central template mesh and fitted NURBS surface."
    )
    p.add_argument(
        "--central_mesh",
        type=str,
        required=True,
        help="Path to central template .ply (e.g., femoral_cartilage_template_central.ply)",
    )
    p.add_argument(
        "--template_npz",
        type=str,
        required=True,
        help="Path to femoral_template_surf.npz (NURBS parameters from fitting script).",
    )
    p.add_argument(
        "--bbox_margin",
        type=float,
        default=10.0,
        help="Margin (mm) around template bbox for keeping NURBS points.",
    )
    p.add_argument(
        "--n_u",
        type=int,
        default=120,
        help="Number of sampling points in u direction for NURBS visualization.",
    )
    p.add_argument(
        "--n_v",
        type=int,
        default=80,
        help="Number of sampling points in v direction for NURBS visualization.",
    )
    p.add_argument(
        "--show_ctrlmesh",
        action="store_true",
        help="Also visualize NURBS control mesh (quad wireframe).",
    )
    return p.parse_args()


def load_nurbs_from_npz(npz_path: Path):
    data = np.load(str(npz_path))
    ctrlpts = data["ctrlpts"]        # (nu_ctrl, nv_ctrl, 3)
    knots_u = data["knots_u"]
    knots_v = data["knots_v"]
    degree_u = int(data["degree_u"])
    degree_v = int(data["degree_v"])

    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    # geomdl expects ctrlpts2d as list[list[[x,y,z], ...], ...]
    surf.ctrlpts2d = ctrlpts.tolist()
    surf.knotvector_u = knots_u.tolist()
    surf.knotvector_v = knots_v.tolist()

    return surf, ctrlpts


def sample_nurbs_points(surf, n_u: int = 120, n_v: int = 80) -> np.ndarray:
    """
    Sample NURBS surface on (u,v) grid in its param range.
    Returns (n_u * n_v, 3) array.
    """
    u_min, u_max = surf.knotvector_u[0], surf.knotvector_u[-1]
    v_min, v_max = surf.knotvector_v[0], surf.knotvector_v[-1]

    u_vals = np.linspace(u_min, u_max, n_u)
    v_vals = np.linspace(v_min, v_max, n_v)

    pts = []
    for u in u_vals:
        for v in v_vals:
            p = np.array(surf.evaluate_single((float(u), float(v))), dtype=float)
            pts.append(p)
    pts = np.asarray(pts, dtype=float)
    return pts


def build_ctrlmesh_lines(ctrlpts_grid: np.ndarray) -> o3d.geometry.LineSet:
    """
    将控制点网格 (nu, nv, 3) 转成一个折线网格，用于可视化控制网拓扑。
    """
    nu, nv, _ = ctrlpts_grid.shape
    pts = ctrlpts_grid.reshape(-1, 3)
    lines = []

    # 连接 u 方向
    for i in range(nu):
        for j in range(nv - 1):
            idx0 = i * nv + j
            idx1 = idx0 + 1
            lines.append([idx0, idx1])
    # 连接 v 方向
    for j in range(nv):
        for i in range(nu - 1):
            idx0 = i * nv + j
            idx1 = (i + 1) * nv + j
            lines.append([idx0, idx1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.0, 0.0, 1.0]]), (len(lines), 1))
    )  # blue
    return line_set


def main():
    args = parse_args()
    central_path = Path(args.central_mesh)
    npz_path = Path(args.template_npz)

    # ---- 1. Load central mesh ----
    print(f"Loading central mesh: {central_path}")
    central = o3d.io.read_triangle_mesh(str(central_path))
    if not central.has_vertices():
        raise RuntimeError(f"Failed to load central mesh or mesh is empty: {central_path}")
    central.compute_vertex_normals()
    central.paint_uniform_color([0.7, 0.7, 0.7])  # gray

    # Template bbox (for filtering NURBS points)
    verts_template = np.asarray(central.vertices)
    bb_min = verts_template.min(axis=0)
    bb_max = verts_template.max(axis=0)
    print(f"Template bbox min: {bb_min}, max: {bb_max}")

    # ---- 2. Load NURBS parameters ----
    print(f"Loading NURBS template params from: {npz_path}")
    surf, ctrlpts_grid = load_nurbs_from_npz(npz_path)
    print(
        f"NURBS surface: degree_u={surf.degree_u}, degree_v={surf.degree_v}, "
        f"ctrlpts_size_u={surf.ctrlpts_size_u}, ctrlpts_size_v={surf.ctrlpts_size_v}"
    )

    # ---- 3. Sample NURBS ----
    print(f"Sampling NURBS surface: n_u={args.n_u}, n_v={args.n_v}")
    pts_nurbs = sample_nurbs_points(surf, n_u=args.n_u, n_v=args.n_v)

    # ---- 4. Filter NURBS points by bbox ± margin ----
    margin = float(args.bbox_margin)
    mask = np.all(
        (pts_nurbs >= bb_min - margin) & (pts_nurbs <= bb_max + margin),
        axis=1,
    )
    pts_nurbs_filt = pts_nurbs[mask]
    print(
        f"Kept {pts_nurbs_filt.shape[0]} / {pts_nurbs.shape[0]} NURBS samples within "
        f"bbox ± {margin} mm"
    )
    if pts_nurbs_filt.shape[0] == 0:
        raise RuntimeError("All NURBS points filtered out; try increasing --bbox_margin.")

    # ---- 5. Build NURBS point cloud ----
    nurbs_pcd = o3d.geometry.PointCloud()
    nurbs_pcd.points = o3d.utility.Vector3dVector(pts_nurbs_filt)
    nurbs_pcd.paint_uniform_color([1.0, 0.2, 0.2])  # red

    geometries = [central, nurbs_pcd]

    # ---- 6. (可选) 显示控制网格 ----
    if args.show_ctrlmesh:
        ctrlmesh_ls = build_ctrlmesh_lines(ctrlpts_grid)
        geometries.append(ctrlmesh_ls)

    print("Opening viewer: gray = template mesh, red = NURBS samples, blue = control mesh (optional)")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Template (gray) vs NURBS (red points, blue control mesh)",
        width=1024,
        height=768,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
