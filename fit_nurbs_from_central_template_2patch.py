#!/usr/bin/env python 
# -*- coding: utf-8 -*-
r"""
Fit cubic tensor-product B-spline (NURBS) surfaces to the central femoral
cartilage template mesh, using quad-grid control meshes (no LSCM),
and verify the fitting accuracy in 3D via Chamfer distance.

结构性改进：多 patch NURBS 模板
----------------------------------------------------
1. 用固定世界坐标 Y 轴作为“长轴方向”（与现有实现一致）；
2. 沿 Y 轴方向，用投影值把三角面片划分为 2 或 3 段（patch）：
   - n_patches=2: 左半 / 右半
   - n_patches=3: 左 / 中 / 右
3. 对每个 patch 的子网格：
   - 再做局部 PCA 得到两个主方向 pc1, pc2（在该 patch 内）；
   - 把 patch 顶点投影到 (pc1, pc2) 平面，归一化到 [0,1]^2；
   - 在 [0,1]^2 上建立规则 size_u × size_v 网格；
   - 每个 (u,v) 网格中心在该 patch 顶点中找最近邻，取其 3D 坐标作为控制点；
   - 用这些控制点构造一个三次 B-spline 曲面（不做额外拟合）。
4. 在每个 patch 的 NURBS 曲面上采样点，把所有 patch 的采样点并在一起，
   与原始 central template 表面做 Chamfer 距离评估整体拟合质量。
  5. 输出：
     - <roi_name>_template_surf_patch0.npz / patch1.npz (/ patch2.npz)
     - <roi_name>_template_nurbs_mesh_multi.ply   （所有 patch 的 NURBS 网格合在一起）
     - <roi_name>_template_chamfer_stats.txt      （Chamfer 统计）

Example:

python fit_nurbs_from_central_template_2patch.py ^
  --roi 2 ^
  --central_mesh C:\Users\chris\MICCAI2026\Carti_Seg\femoral_cartilage_average_mesh.ply ^
  --out_dir      C:\Users\chris\MICCAI2026\Carti_Seg ^
  --n_patches 2
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from geomdl import BSpline, utilities
from scipy.spatial import cKDTree

from template_utils import DEFAULT_MAX_DIST_ON_TEMPLATE, filter_points_near_template

ROI_FEMUR = 1
ROI_FEMORAL_CARTILAGE = 2
ROI_TIBIA = 3
ROI_MEDIAL_TIBIAL_CARTILAGE = 4
ROI_LATERAL_TIBIAL_CARTILAGE = 5

ROI_ID_TO_NAME = {
    ROI_FEMUR: "femur",
    ROI_FEMORAL_CARTILAGE: "femoral_cartilage",
    ROI_TIBIA: "tibia",
    ROI_MEDIAL_TIBIAL_CARTILAGE: "medial_tibial_cartilage",
    ROI_LATERAL_TIBIAL_CARTILAGE: "lateral_tibial_cartilage",
}


def _parse_roi(roi_value: str):
    """Parse ROI from CLI (accepts id or anatomical name)."""

    if roi_value.isdigit():
        roi_id = int(roi_value)
        if roi_id not in ROI_ID_TO_NAME:
            raise ValueError(f"Unsupported ROI id: {roi_id}")
        return roi_id, ROI_ID_TO_NAME[roi_id]

    roi_key = roi_value.strip().lower()
    for k, v in ROI_ID_TO_NAME.items():
        if roi_key == v:
            return k, v

    raise ValueError(
        f"Unsupported ROI '{roi_value}'. Use one of: "
        f"{', '.join(ROI_ID_TO_NAME.values())} or ids {list(ROI_ID_TO_NAME.keys())}"
    )


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Fit multi-patch NURBS surfaces from central cartilage template (quad control meshes, no LSCM)."
    )
    p.add_argument(
        "--roi",
        type=str,
        default=str(ROI_FEMORAL_CARTILAGE),
        help=(
            "ROI id or anatomical name. Supported: "
            f"{', '.join(f'{k}:{v}' for k, v in ROI_ID_TO_NAME.items())}."
        ),
    )
    p.add_argument(
        "--central_mesh",
        type=str,
        default=None,
        help=(
            "Path to ROI central template .ply. If omitted, defaults to "
            "<out_dir>/<roi_name>_average_mesh.ply."
        ),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Output directory to save NURBS template and verification meshes. "
            "Defaults to ./<roi_name>_template."
        ),
    )
    p.add_argument(
        "--n_patches",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of longitudinal patches (2 or 3).",
    )
    p.add_argument(
        "--size_u",
        type=int,
        default=60,
        help="Number of control points in u direction (per patch).",
    )
    p.add_argument(
        "--size_v",
        type=int,
        default=40,
        help="Number of control points in v direction (per patch).",
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
        default=20000,
        help="Number of points sampled from template mesh for Chamfer.",
    )
    p.add_argument(
        "--n_sample_nurbs",
        type=int,
        default=20000,
        help="Target number of points sampled from all NURBS patches for Chamfer.",
    )
    p.add_argument(
        "--bbox_margin",
        type=float,
        default=10.0,
        help="Margin (mm) around template bbox to keep NURBS points for Chamfer.",
    )
    p.add_argument(
        "--max_dist_on_template",
        type=float,
        default=DEFAULT_MAX_DIST_ON_TEMPLATE,
        help="Maximum distance (mm) from template surface to keep NURBS samples.",
    )
    return p.parse_args()


# -------------------------------------------------------------------------
# 工具：mesh 划分为多个 patch（使用固定 Y 轴）
# -------------------------------------------------------------------------
def split_mesh_longitudinal(
    mesh: o3d.geometry.TriangleMesh,
    n_patches: int = 2,
):
    """
    沿固定世界坐标 Y 轴将 mesh 按三角面片划分为 2 或 3 个子 mesh。

    返回:
      patch_meshes: List[TriangleMesh], 长度 = n_patches
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)   # (N,3)
    faces = np.asarray(mesh.triangles, dtype=np.int32)    # (M,3)

    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise RuntimeError("Mesh is empty, cannot split.")

    # 用顶点中心化，仅影响一个常数偏移，不影响根据投影排序/分段
    center = verts.mean(axis=0, keepdims=True)
    verts_centered = verts - center

    # 固定“长轴方向”为世界坐标 Y 轴（与现有实现保持一致）
    pc_long = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # 顶点在固定长轴上的投影
    proj = verts_centered @ pc_long  # (N,)

    # 用面心的投影值决定每个 face 属于哪个 patch
    face_proj = proj[faces].mean(axis=1)  # (M,)

    if n_patches == 2:
        t = np.median(face_proj)
        thresholds = [t]
    elif n_patches == 3:
        t1, t2 = np.quantile(face_proj, [1.0 / 3.0, 2.0 / 3.0])
        thresholds = [t1, t2]
    else:
        raise ValueError("n_patches must be 2 or 3.")

    # 根据阈值分配面
    patch_face_indices = []
    if n_patches == 2:
        patch_face_indices.append(np.where(face_proj <= thresholds[0])[0])
        patch_face_indices.append(np.where(face_proj > thresholds[0])[0])
    else:  # 3 patches
        patch_face_indices.append(np.where(face_proj <= thresholds[0])[0])
        patch_face_indices.append(
            np.where((face_proj > thresholds[0]) & (face_proj <= thresholds[1]))[0]
        )
        patch_face_indices.append(np.where(face_proj > thresholds[1])[0])

    patch_meshes = []
    for k, idx_faces in enumerate(patch_face_indices):
        if idx_faces.size == 0:
            raise RuntimeError(f"Patch {k} has no faces, splitting failed.")

        faces_k = faces[idx_faces]  # (Mk,3)
        used_verts = np.unique(faces_k.reshape(-1))
        verts_k = verts[used_verts]

        # 旧 -> 新索引映射
        old2new = -np.ones(verts.shape[0], dtype=np.int32)
        old2new[used_verts] = np.arange(used_verts.shape[0], dtype=np.int32)
        faces_k_new = old2new[faces_k]

        mesh_k = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts_k),
            triangles=o3d.utility.Vector3iVector(faces_k_new),
        )
        mesh_k.compute_vertex_normals()
        mesh_k.compute_triangle_normals()
        patch_meshes.append(mesh_k)

        print(
            f"Patch {k}: {verts_k.shape[0]} vertices, {faces_k_new.shape[0]} faces."
        )

    return patch_meshes, pc_long



# -------------------------------------------------------------------------
# 1. 基于 PCA + 最近邻 在“单个 patch”表面上构造 quad 控制网格
# -------------------------------------------------------------------------
def build_quad_control_mesh_from_template(
    mesh: o3d.geometry.TriangleMesh,
    size_u: int,
    size_v: int,
    long_axis: np.ndarray | None = None,
):
    """
    从 patch mesh 构造一个规则的 size_u x size_v 控制点网格（quad 控制网格）：
      1) PCA 得到局部主方向 pc1, pc2（在该 patch 内）；
      2) 所有顶点投影到 (pc1, pc2) 平面上得到 (u', v')；
      3) 归一化到 [0,1]^2 得到 uv_norm；
      4) 对每个规则网格中心 (u, v) 在 uv_norm 中最近邻，取相应顶点的 3D 坐标作为控制点。

    返回:
      ctrlpts_grid: (size_u, size_v, 3) 世界坐标控制点
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)  # (N, 3)

    # ---- 1) PCA: 找局部主方向 ----
    center = verts.mean(axis=0, keepdims=True)  # (1,3)
    verts_centered = verts - center             # (N,3)
    U, S, Vt = np.linalg.svd(verts_centered, full_matrices=False)
    pc1 = Vt[0]
    pc2 = Vt[1]

    if long_axis is not None:
        ref_u = long_axis.astype(np.float64)
        ref_u /= np.linalg.norm(ref_u) + 1e-12
    else:
        # 如果没给，就退回到原来的全局 +X
        ref_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if np.dot(pc1, ref_u) < 0:
        pc1 = -pc1

    # 保持右手系，并让法向尽量朝向全局 +Z
    pc3 = np.cross(pc1, pc2)
    ref_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.dot(pc3, ref_normal) < 0:
        pc2 = -pc2
        pc3 = -pc3

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

    tree_uv = cKDTree(uv_norm)

    print("  Building quad control mesh (per patch) by nearest neighbors on PCA-uv plane...")
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
    三次 B-spline 曲面（Tensor-product surface）。
    """
    size_u, size_v, _ = ctrlpts_grid.shape

    surf = BSpline.Surface()
    surf.degree_u = degree_u
    surf.degree_v = degree_v

    # geomdl 需要 2D list: ctrlpts2d[u][v] = [x,y,z]
    surf.ctrlpts2d = ctrlpts_grid.tolist()

    # open-uniform knot vectors
    kv_u = utilities.generate_knot_vector(degree_u, size_u)
    kv_v = utilities.generate_knot_vector(degree_v, size_v)
    surf.knotvector_u = kv_u
    surf.knotvector_v = kv_v

    return surf


# -------------------------------------------------------------------------
# 3. 采样 / Chamfer / 保存
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


def save_nurbs_template(surf, out_dir: Path, roi_name: str, patch_id: int):
    """
    每个 patch 单独保存一个 npz:
      femoral_template_surf_patch{patch_id}.npz
    """
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
        out_dir / f"{roi_name}_template_surf_patch{patch_id}.npz",
        ctrlpts=ctrlpts_3d,
        knots_u=knots_u,
        knots_v=knots_v,
        degree_u=degree_u,
        degree_v=degree_v,
    )
    print(
        "NURBS template parameters saved to: "
        f"{out_dir / f'{roi_name}_template_surf_patch{patch_id}.npz'}"
    )


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    roi_id, roi_name = _parse_roi(args.roi)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path(f"{roi_name}_template")
    )

    default_central = out_dir / f"{roi_name}_average_mesh.ply"
    central_mesh_path = Path(args.central_mesh) if args.central_mesh else default_central

    print(
        f"ROI {roi_id} ({roi_name}). Loading central template mesh from: {central_mesh_path}"
    )
    mesh = o3d.io.read_triangle_mesh(str(central_mesh_path))
    if not mesh.has_vertices():
        raise RuntimeError(
            f"Central mesh has no vertices, please check the path: {central_mesh_path}"
        )

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # ---------- Step 0: 沿固定 Y 轴切成多个 patch ----------
    print(f"Splitting mesh into {args.n_patches} longitudinal patches (fixed Y axis)...")
    patch_meshes, pc_long = split_mesh_longitudinal(mesh, n_patches=args.n_patches)

    # 用于汇总的 NURBS 点 & NURBS 网格
    all_nurbs_pts_dense = []
    nurbs_patch_meshes = []

    # ---------- 对每个 patch 分别拟合 NURBS ----------
    for pid, patch_mesh in enumerate(patch_meshes):
        print(f"\n=== Processing patch {pid} ===")
        print(
            f"  Building quad control mesh: size_u={args.size_u}, size_v={args.size_v}"
        )
        ctrl_grid = build_quad_control_mesh_from_template(
            patch_mesh,
            size_u=args.size_u,
            size_v=args.size_v,
            long_axis=pc_long,      # 统一的“解剖长轴正方向”（此处为固定 Y 轴）
        )

        print(
            f"  Building NURBS surface from control grid (degree_u={args.degree_u}, degree_v={args.degree_v})..."
        )
        surf = build_nurbs_from_control_grid(
            ctrl_grid,
            degree_u=args.degree_u,
            degree_v=args.degree_v,
        )

        print(
            f"  NURBS surface (patch {pid}): degree_u={surf.degree_u}, "
            f"degree_v={surf.degree_v}, "
            f"ctrlpts_size_u={surf.ctrlpts_size_u}, "
            f"ctrlpts_size_v={surf.ctrlpts_size_v}"
        )

        print(f"  Saving NURBS template parameters for patch {pid}...")
        save_nurbs_template(surf, out_dir, roi_name=roi_name, patch_id=pid)

        # 为 Chamfer 和可视化采样
        pts_nurbs_dense = sample_nurbs_surface(
            surf,
            n_u=int(np.sqrt(args.n_sample_nurbs) * 2),
            n_v=int(np.sqrt(args.n_sample_nurbs) * 2),
        )
        all_nurbs_pts_dense.append(pts_nurbs_dense)

        # 可视化网格（每个 patch 单独一个 mesh）
        n_u_vis, n_v_vis = 80, 60
        pts_vis = sample_nurbs_surface(surf, n_u=n_u_vis, n_v=n_v_vis)
        nurbs_mesh = build_mesh_from_grid_points(pts_vis, n_u_vis, n_v_vis)
        nurbs_patch_meshes.append(nurbs_mesh)

    # ---------- Step 3: Chamfer 验证（所有 patch 一起） ----------
    print("\nSampling points from template mesh...")
    template_pcd = mesh.sample_points_uniformly(number_of_points=args.n_sample_template)
    pts_template = np.asarray(template_pcd.points)

    print("Concatenating NURBS samples from all patches...")
    pts_nurbs_dense_all = np.concatenate(all_nurbs_pts_dense, axis=0)

    bb_min = pts_template.min(axis=0)
    bb_max = pts_template.max(axis=0)
    margin = float(args.bbox_margin)
    mask = np.all(
        (pts_nurbs_dense_all >= bb_min - margin)
        & (pts_nurbs_dense_all <= bb_max + margin),
        axis=1,
    )
    pts_nurbs_filtered = pts_nurbs_dense_all[mask]
    print(
        f"Filtered NURBS points (bbox): {pts_nurbs_filtered.shape[0]} / "
        f"{pts_nurbs_dense_all.shape[0]} kept within bbox ± {margin} mm"
    )

    if pts_nurbs_filtered.shape[0] == 0:
        raise RuntimeError("All NURBS samples were filtered out; try increasing bbox_margin.")

    print(
        f"Applying template distance filter (max_dist_on_template={args.max_dist_on_template} mm)..."
    )
    pts_nurbs_filtered = filter_points_near_template(
        pts_nurbs_filtered, pts_template, max_dist=args.max_dist_on_template
    )
    print(
        f"Filtered NURBS points (distance): {pts_nurbs_filtered.shape[0]} / "
        f"{mask.sum()} kept within {args.max_dist_on_template} mm of template"
    )

    if pts_nurbs_filtered.shape[0] > args.n_sample_nurbs:
        idx = np.random.choice(
            pts_nurbs_filtered.shape[0], args.n_sample_nurbs, replace=False
        )
        pts_nurbs = pts_nurbs_filtered[idx]
    else:
        pts_nurbs = pts_nurbs_filtered

    print("Computing Chamfer distance between multi-patch NURBS and template mesh...")
    stats = chamfer_distance(pts_nurbs, pts_template)

    print("Chamfer stats (all in mm):")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    stats_path = out_dir / f"{roi_name}_template_chamfer_stats.txt"
    with open(stats_path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Chamfer stats saved to: {stats_path}")

    # ---------- Step 4: 保存合并后的 NURBS 网格用于可视化 ----------
    print("Saving combined multi-patch NURBS surface mesh for visualization...")
    # 把每个 patch 的 nurbs_mesh 合并到一个 mesh 里
    all_vertices = []
    all_triangles = []
    v_offset = 0
    for m in nurbs_patch_meshes:
        v = np.asarray(m.vertices)
        t = np.asarray(m.triangles)
        all_vertices.append(v)
        all_triangles.append(t + v_offset)
        v_offset += v.shape[0]
    all_vertices = np.concatenate(all_vertices, axis=0)
    all_triangles = np.concatenate(all_triangles, axis=0)

    combined_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(all_vertices),
        triangles=o3d.utility.Vector3iVector(all_triangles),
    )
    combined_mesh.compute_vertex_normals()
    combined_mesh.compute_triangle_normals()

    nurbs_mesh_path = out_dir / f"{roi_name}_template_nurbs_mesh_multi.ply"
    o3d.io.write_triangle_mesh(str(nurbs_mesh_path), combined_mesh)
    print(f"Combined multi-patch NURBS mesh saved to: {nurbs_mesh_path}")

    print("Done.")


if __name__ == "__main__":
    main()
