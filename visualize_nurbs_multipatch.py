#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Visualize central template mesh and multi-patch NURBS surfaces.

- 灰色：central template mesh
- 彩色点：各 NURBS patch 的采样点
- 彩色线框：各 patch 的控制网格（可选）

Usage example (Windows CMD):

  python visualize_nurbs_multipatch.py ^
    --central_mesh  C:\Users\chris\MICCAI2026\Carti_Seg\average_mesh.ply ^
    --out_dir       C:\Users\chris\MICCAI2026\Carti_Seg ^
    --n_patches     2 ^
    --bbox_margin   10.0 ^
    --n_u 40 --n_v 40 ^
    --show_ctrlmesh

要求：fit 脚本已经在 out_dir 下生成
  femoral_template_surf_patch0.npz
  femoral_template_surf_patch1.npz
  (如果 n_patches=3 则还有 femoral_template_surf_patch2.npz)
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from geomdl import BSpline


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize central mesh and multi-patch NURBS surfaces."
    )
    p.add_argument(
        "--central_mesh",
        type=str,
        required=True,
        help="Path to central template .ply (e.g., femoral_cartilage_template_central.ply)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory containing femoral_template_surf_patch*.npz and NURBS meshes.",
    )
    p.add_argument(
        "--n_patches",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of NURBS patches (must match fit_nurbs_from_central_template.py).",
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
        help="Number of sampling points in u direction for NURBS visualization (per patch).",
    )
    p.add_argument(
        "--n_v",
        type=int,
        default=80,
        help="Number of sampling points in v direction for NURBS visualization (per patch).",
    )
    p.add_argument(
        "--show_ctrlmesh",
        action="store_true",
        help="Also visualize control meshes (wireframe) for each patch.",
    )
    return p.parse_args()


# -------------------------------------------------------------------------
# NURBS helpers
# -------------------------------------------------------------------------
def load_nurbs_patch(npz_path: Path):
    """
    从 femoral_template_surf_patch*.npz 读取 NURBS patch 参数。
    """
    data = np.load(str(npz_path))
    ctrlpts = data["ctrlpts"]        # (nu_ctrl, nv_ctrl, 3)
    knots_u = data["knots_u"]
    knots_v = data["knots_v"]
    degree_u = int(data["degree_u"])
    degree_v = int(data["degree_v"])

    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v
    surf.ctrlpts2d = ctrlpts.tolist()
    surf.knotvector_u = knots_u.tolist()
    surf.knotvector_v = knots_v.tolist()

    return surf, ctrlpts


def sample_nurbs_points(surf, n_u: int = 120, n_v: int = 80) -> np.ndarray:
    """
    在 param 域上等间隔采样 NURBS patch。
    返回 (n_u*n_v, 3).
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


def build_ctrlmesh_lines(ctrlpts_grid: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
    """
    ctrlpts_grid: (nu, nv, 3)
    color: (3,) RGB in [0,1]
    """
    nu, nv, _ = ctrlpts_grid.shape
    pts = ctrlpts_grid.reshape(-1, 3)
    lines = []

    # u 方向
    for i in range(nu):
        for j in range(nv - 1):
            idx0 = i * nv + j
            idx1 = idx0 + 1
            lines.append([idx0, idx1])
    # v 方向
    for j in range(nv):
        for i in range(nu - 1):
            idx0 = i * nv + j
            idx1 = (i + 1) * nv + j
            lines.append([idx0, idx1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(color.reshape(1, 3), (len(lines), 1))
    )
    return line_set


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    central_path = Path(args.central_mesh)
    out_dir = Path(args.out_dir)

    # ---- 1. load central mesh ----
    print(f"Loading central mesh: {central_path}")
    central = o3d.io.read_triangle_mesh(str(central_path))
    if not central.has_vertices():
        raise RuntimeError(f"Failed to load central mesh or mesh is empty: {central_path}")
    central.compute_vertex_normals()
    central.paint_uniform_color([0.7, 0.7, 0.7])  # gray

    verts_template = np.asarray(central.vertices)
    bb_min = verts_template.min(axis=0)
    bb_max = verts_template.max(axis=0)
    print(f"Template bbox min: {bb_min}, max: {bb_max}")

    # 为不同 patch 准备几种颜色
    patch_colors = [
        np.array([1.0, 0.0, 0.0]),  # red
        np.array([0.0, 0.7, 0.0]),  # green
        np.array([0.0, 0.3, 1.0]),  # blue
    ]

    geometries = [central]

    # ---- 2. 对每个 patch 读取参数并可视化 ----
    for pid in range(args.n_patches):
        npz_path = out_dir / f"femoral_template_surf_patch{pid}.npz"
        print(f"\nLoading NURBS patch {pid} from: {npz_path}")
        if not npz_path.exists():
            raise RuntimeError(f"Patch npz not found: {npz_path}")

        surf, ctrlpts_grid = load_nurbs_patch(npz_path)
        print(
            f"  Patch {pid}: degree_u={surf.degree_u}, degree_v={surf.degree_v}, "
            f"ctrlpts_size_u={surf.ctrlpts_size_u}, ctrlpts_size_v={surf.ctrlpts_size_v}"
        )

        # 采样 NURBS 点
        print(f"  Sampling patch {pid}: n_u={args.n_u}, n_v={args.n_v}")
        pts_nurbs = sample_nurbs_points(surf, n_u=args.n_u, n_v=args.n_v)

        # bbox 过滤
        margin = float(args.bbox_margin)
        mask = np.all(
            (pts_nurbs >= bb_min - margin) & (pts_nurbs <= bb_max + margin),
            axis=1,
        )
        pts_nurbs_filt = pts_nurbs[mask]
        print(
            f"  Patch {pid}: kept {pts_nurbs_filt.shape[0]} / {pts_nurbs.shape[0]} points within bbox ± {margin} mm"
        )
        if pts_nurbs_filt.shape[0] == 0:
            print(f"  WARNING: all points of patch {pid} filtered out; consider increasing --bbox_margin.")

        # 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_nurbs_filt)
        color = patch_colors[pid % len(patch_colors)]
        pcd.paint_uniform_color(color.tolist())
        geometries.append(pcd)

        # 控制网格线框
        if args.show_ctrlmesh:
            ls = build_ctrlmesh_lines(ctrlpts_grid, color=color)
            geometries.append(ls)

    print("\nOpening viewer:")
    print("  - gray  = central template mesh")
    print("  - red/green/blue points = NURBS patches")
    print("  - colored wireframe     = control meshes (if --show_ctrlmesh)")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Multi-patch NURBS vs Template",
        width=1024,
        height=768,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
