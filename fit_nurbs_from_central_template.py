#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fit a cubic tensor-product B-spline (NURBS) surface to the central femoral
cartilage template mesh, using a quad-grid control mesh (no LSCM),
and verify the fitting accuracy in 3D via Chamfer distance.

核心思路（方法 B: Quad-remesh-based control mesh）:
  1. 对 central template 做 PCA，得到两个主方向 pc1, pc2；
  2. 把所有顶点投影到 (pc1, pc2) 平面上，得到 2D 坐标 (u', v')；
  3. 归一化到 [0, 1]^2；
  4. 在 [0,1]^2 上建立一个规则 M×N 网格 (size_u × size_v)；
  5. 对每个网格中心 (u, v)，在模板顶点中找“最近的投影点”，用它的 3D 坐标作为控制点；
     → 得到一个规则的 quad 控制网格 ctrlpts[size_u, size_v, 3]；
  6. 用这些控制点直接构造一个三次 B-spline 曲面（不再用 LSCM、不再做 least-squares fitting）；
  7. 在 NURBS 曲面上采样点，与原始模板表面做 Chamfer 距离评估拟合质量。

Inputs:
  - central surface mesh (.ply), e.g.
      femoral_cartilage_template_central.ply

Outputs (in out_dir):
  - femoral_template_surf.npz
      ctrlpts:     (nu_ctrl, nv_ctrl, 3)   control points in world coordinates
      knots_u:     knot vector in u
      knots_v:     knot vector in v
      degree_u/v:  spline degrees
  - femoral_template_nurbs_mesh.ply
      NURBS surface evaluated on a dense (u,v) grid (for visualization)
  - femoral_template_chamfer_stats.txt
      Chamfer distance stats (mm) between template mesh and NURBS surface


python fit_nurbs_from_central_template.py ^
--central_mesh C:\Users\chris\MICCAI2026\Carti_Seg\femoral_cartilage_template_central.ply ^
--out_dir C:\Users\chris\MICCAI2026\Carti_Seg ^
--size_u 150 --size_v 150
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from geomdl import BSpline, utilities
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit and verify a NURBS surface from central cartilage template (quad control mesh, no LSCM)."
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
        help="Output directory to save NURBS template and verification meshes.",
    )
    p.add_argument(
        "--size_u",
        type=int,
        default=60,
        help="Number of control points in u direction (quad grid).",
    )
    p.add_argument(
        "--size_v",
        type=int,
        default=40,
        help="Number of control points in v direction (quad grid).",
    )
    p.add_argument(
        "--degree_u",
        type=int,
        default=3,
        help="B-spline degree in u direction.",
    )
    p.add_argument(
        "--degree_v",
        type=int,
        default=3,
        help="B-spline degree in v direction.",
    )
    p.add_argument(
        "--n_sample_template",
        type=int,
        default=10000,
        help="Number of points sampled from template mesh for Chamfer.",
    )
    p.add_argument(
        "--n_sample_nurbs",
        type=int,
        default=10000,
        help="Target number of points sampled from NURBS surface grid for Chamfer.",
    )
    p.add_argument(
        "--bbox_margin",
        type=float,
        default=10.0,
        help="Margin (mm) around template bbox to keep NURBS points for Chamfer.",
    )
    return p.parse_args()


# -------------------------------------------------------------------------
# 1. 基于 PCA + 最近邻 在模板表面上构造规则 quad 控制网格
# -------------------------------------------------------------------------
def build_quad_control_mesh_from_template(
    mesh: o3d.geometry.TriangleMesh,
    size_u: int,
    size_v: int,
):
    """
    从 central template mesh 构造一个规则的 size_u x size_v 控制点网格（quad 控制网格）：
      1) PCA 得到两个主方向 pc1, pc2；
      2) 所有顶点投影到 (pc1, pc2) 平面上得到 (u', v')；
      3) 归一化到 [0,1]^2 得到 uv_norm；
      4) 对每个规则网格中心 (u, v) 在 uv_norm 中最近邻，取相应顶点的 3D 坐标作为控制点。

    返回:
      ctrlpts_grid: (size_u, size_v, 3) 世界坐标控制点
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)  # (N, 3)

    # ---- 1) PCA: 找主方向 ----
    center = verts.mean(axis=0, keepdims=True)  # (1,3)
    verts_centered = verts - center             # (N,3)
    U, S, Vt = np.linalg.svd(verts_centered, full_matrices=False)
    pc1 = Vt[0]   # 第一主方向
    pc2 = Vt[1]   # 第二主方向

    # ---- 2) 投影到 PCA 平面 ----
    u_prime = verts_centered @ pc1
    v_prime = verts_centered @ pc2
    uv_prime = np.stack([u_prime, v_prime], axis=1)  # (N,2)

    # ---- 3) 归一化到 [0,1]^2 ----
    min_uv = uv_prime.min(axis=0)
    max_uv = uv_prime.max(axis=0)
    span_uv = np.maximum(max_uv - min_uv, 1e-8)
    uv_norm = (uv_prime - min_uv) / span_uv  # (N,2) in [0,1]^2

    # ---- 4) 规则网格 + 最近邻采样 3D 控制点 ----
    u_vals = np.linspace(0.0, 1.0, size_u)
    v_vals = np.linspace(0.0, 1.0, size_v)

    ctrlpts_grid = np.zeros((size_u, size_v, 3), dtype=np.float64)

    # 用 KD-tree 加速最近邻
    tree_uv = cKDTree(uv_norm)

    print("Building quad control mesh by nearest neighbors on PCA-uv plane...")
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            query_uv = np.array([u, v], dtype=np.float64)
            _, idx = tree_uv.query(query_uv, k=1)
            ctrlpts_grid[i, j, :] = verts[idx]

    return ctrlpts_grid  # (size_u, size_v, 3)


# -------------------------------------------------------------------------
# 2. 用 quad 控制网格构造 NURBS/B-spline 曲面
# -------------------------------------------------------------------------
def build_nurbs_from_control_grid(
    ctrlpts_grid: np.ndarray,
    degree_u: int = 3,
    degree_v: int = 3,
):
    """
    从规则控制点网格 ctrlpts_grid[size_u, size_v, 3] 构造一个几何上光滑的
    三次 B-spline 曲面（Tensor-product surface），不做额外拟合。

    返回:
      surf: geomdl.BSpline.Surface 实例
    """
    size_u, size_v, _ = ctrlpts_grid.shape

    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v

    # geomdl 需要 2D list: ctrlpts2d[u][v] = [x,y,z]
    surf.ctrlpts2d = ctrlpts_grid.tolist()

    # 生成 open-uniform knot vector
    kv_u = utilities.generate_knot_vector(degree_u, size_u)
    kv_v = utilities.generate_knot_vector(degree_v, size_v)
    surf.knotvector_u = kv_u
    surf.knotvector_v = kv_v

    return surf


# -------------------------------------------------------------------------
# 3. 后续采样 / Chamfer / 保存，与之前版本类似
# -------------------------------------------------------------------------
def sample_nurbs_surface(surf, n_u: int = 120, n_v: int = 80) -> np.ndarray:
    """
    在 NURBS 曲面的参数域上均匀采样 n_u x n_v 个点, 返回 (n_u*n_v, 3).
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
    pts = np.asarray(pts, dtype=float)  # (n_u*n_v, 3)
    return pts


def build_mesh_from_grid_points(points: np.ndarray, n_u: int, n_v: int) -> o3d.geometry.TriangleMesh:
    verts = points
    faces = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            idx0 = i * n_v + j
            idx1 = idx0 + 1
            idx2 = (i + 1) * n_v + j
            idx3 = idx2 + 1
            faces.append([idx0, idx2, idx1])
            faces.append([idx1, idx2, idx3])
    faces = np.asarray(faces, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def chamfer_distance(pts_pred: np.ndarray, pts_gt: np.ndarray) -> dict:
    tree_gt = cKDTree(pts_gt)
    d_pred2gt, _ = tree_gt.query(pts_pred, k=1)

    tree_pred = cKDTree(pts_pred)
    d_gt2pred, _ = tree_pred.query(pts_gt, k=1)

    stats = {
        "pred2gt_mean": float(d_pred2gt.mean()),
        "pred2gt_median": float(np.median(d_pred2gt)),
        "pred2gt_95": float(np.percentile(d_pred2gt, 95)),
        "pred2gt_max": float(d_pred2gt.max()),
        "gt2pred_mean": float(d_gt2pred.mean()),
        "gt2pred_median": float(np.median(d_gt2pred)),
        "gt2pred_95": float(np.percentile(d_gt2pred, 95)),
        "gt2pred_max": float(d_gt2pred.max()),
        "chamfer": float(d_pred2gt.mean() + d_gt2pred.mean()),
    }
    return stats


def save_nurbs_template(surf, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    ctrlpts = np.array(surf.ctrlpts, dtype=float)  # (nu_ctrl*nv_ctrl, 3)
    nu_ctrl = surf.ctrlpts_size_u
    nv_ctrl = surf.ctrlpts_size_v
    ctrlpts_3d = ctrlpts.reshape(nu_ctrl, nv_ctrl, 3)

    knots_u = np.array(surf.knotvector_u, dtype=float)
    knots_v = np.array(surf.knotvector_v, dtype=float)
    degree_u = surf.degree_u
    degree_v = surf.degree_v

    np.savez(
        out_dir / "femoral_template_surf.npz",
        ctrlpts=ctrlpts_3d,
        knots_u=knots_u,
        knots_v=knots_v,
        degree_u=degree_u,
        degree_v=degree_v,
    )
    print(f"NURBS template parameters saved to: {out_dir / 'femoral_template_surf.npz'}")


def main():
    args = parse_args()
    central_mesh_path = Path(args.central_mesh)
    out_dir = Path(args.out_dir)

    print(f"Loading central template mesh from: {central_mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(central_mesh_path))
    if not mesh.has_vertices():
        raise RuntimeError("Central mesh has no vertices, please check the path.")

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # ---------- Step 1: quad 控制网格 ----------
    print(
        f"Building quad control mesh: size_u={args.size_u}, "
        f"size_v={args.size_v} (no LSCM, PCA-based uv)."
    )
    ctrl_grid = build_quad_control_mesh_from_template(
        mesh,
        size_u=args.size_u,
        size_v=args.size_v,
    )

    # ---------- Step 2: 用控制网格构造 NURBS 曲面 ----------
    print(
        f"Building NURBS surface from control grid (degree_u={args.degree_u}, degree_v={args.degree_v})..."
    )
    surf = build_nurbs_from_control_grid(
        ctrl_grid,
        degree_u=args.degree_u,
        degree_v=args.degree_v,
    )

    print(
        f"NURBS surface ready: degree_u={surf.degree_u}, "
        f"degree_v={surf.degree_v}, "
        f"ctrlpts_size_u={surf.ctrlpts_size_u}, "
        f"ctrlpts_size_v={surf.ctrlpts_size_v}"
    )

    print("Saving NURBS template parameters...")
    save_nurbs_template(surf, out_dir)

    # ---------- Step 3: Chamfer 验证 ----------
    print("Sampling points from template mesh...")
    template_pcd = mesh.sample_points_uniformly(number_of_points=args.n_sample_template)
    pts_template = np.asarray(template_pcd.points)

    print("Sampling points from NURBS surface...")
    pts_nurbs_dense = sample_nurbs_surface(
        surf,
        n_u=int(np.sqrt(args.n_sample_nurbs) * 2),
        n_v=int(np.sqrt(args.n_sample_nurbs) * 2),
    )

    bb_min = pts_template.min(axis=0)
    bb_max = pts_template.max(axis=0)
    margin = float(args.bbox_margin)
    mask = np.all(
        (pts_nurbs_dense >= bb_min - margin) & (pts_nurbs_dense <= bb_max + margin),
        axis=1,
    )
    pts_nurbs_filtered = pts_nurbs_dense[mask]
    print(
        f"Filtered NURBS points: {pts_nurbs_filtered.shape[0]} / "
        f"{pts_nurbs_dense.shape[0]} kept within bbox ± {margin} mm"
    )

    if pts_nurbs_filtered.shape[0] == 0:
        raise RuntimeError("All NURBS samples were filtered out; try increasing bbox_margin.")

    if pts_nurbs_filtered.shape[0] > args.n_sample_nurbs:
        idx = np.random.choice(pts_nurbs_filtered.shape[0], args.n_sample_nurbs, replace=False)
        pts_nurbs = pts_nurbs_filtered[idx]
    else:
        pts_nurbs = pts_nurbs_filtered

    print("Computing Chamfer distance between NURBS surface and template mesh...")
    stats = chamfer_distance(pts_nurbs, pts_template)

    print("Chamfer stats (all in mm):")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    stats_path = out_dir / "femoral_template_chamfer_stats.txt"
    with open(stats_path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Chamfer stats saved to: {stats_path}")

    # ---------- Step 4: 保存 NURBS 网格用于可视化 ----------
    print("Saving NURBS surface mesh for visualization...")
    n_u_vis, n_v_vis = 120, 80
    pts_vis = sample_nurbs_surface(surf, n_u=n_u_vis, n_v=n_v_vis)
    nurbs_mesh = build_mesh_from_grid_points(pts_vis, n_u_vis, n_v_vis)
    nurbs_mesh_path = out_dir / "femoral_template_nurbs_mesh.ply"
    o3d.io.write_triangle_mesh(str(nurbs_mesh_path), nurbs_mesh)
    print(f"NURBS mesh saved to: {nurbs_mesh_path}")

    print("Done.")


if __name__ == "__main__":
    main()
