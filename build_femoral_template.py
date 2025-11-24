#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Template builder for ROI=2 (Femoral Cartilage) from OAI-ZIB-CM with
NURBS (cubic tensor-product B-spline) template fitting.

Pipeline (per your description):
1. Randomly select 20–30 femoral cartilage masks from OAI-ZIB-CM/labelsTr.
2. Extract ROI=2 binary masks.
3. Marching Cubes → surface meshes; smooth (Taubin / Laplacian).
4. Rigid ICP to align all meshes to a common space.
5. On a reference topology, compute an average cartilage surface.
6. Fit a cubic tensor-product B-spline (NURBS) central template surface S_c(u,v).
7. On the same param domain (u,v), define a thickness field T(u,v),
   initialize T0(u,v) ≈ 2 mm, and save as a thickness map.

Outputs:
- average_mesh.ply              (Open3D mesh of average surface)
- femoral_template_surf.npz     (control points, knots, degrees, etc.)
- femoral_template_thickness.npy (thickness map on (u,v) grid)

python build_femoral_template.py ^
    --data_root C:\Users\chris\MICCAI2026\OAI-ZIB-CM ^
    --output_dir C:\Users\chris\MICCAI2026\Carti_Seg ^
    --num_cases 25


Author: (you + ChatGPT)
"""

import os
import glob
import argparse
import random

import numpy as np
import nibabel as nib
from skimage import measure
import open3d as o3d

# For NURBS / B-spline surface fitting
from geomdl import fitting


ROI_FEMORAL_CARTILAGE = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a femoral cartilage NURBS template (ROI=2) from OAI-ZIB-CM."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder of OAI-ZIB-CM (containing imagesTr, labelsTr, ...).",
    )
    parser.add_argument(
        "--num_cases",
        type=int,
        default=25,
        help="Number of training cases to sample (20–30 recommended).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./femoral_template",
        help="Directory to store the template outputs.",
    )
    parser.add_argument(
        "--min_voxels",
        type=int,
        default=500,
        help="Skip masks smaller than this voxel count.",
    )
    parser.add_argument(
        "--num_sample_points",
        type=int,
        default=10000,
        help="Number of points sampled per mesh for ICP.",
    )
    parser.add_argument(
        "--smooth_iters",
        type=int,
        default=10,
        help="Number of smoothing iterations per mesh.",
    )
    parser.add_argument(
        "--uv_size_u",
        type=int,
        default=60,
        help="Number of control grid samples in u direction for NURBS fitting.",
    )
    parser.add_argument(
        "--uv_size_v",
        type=int,
        default=40,
        help="Number of control grid samples in v direction for NURBS fitting.",
    )
    parser.add_argument(
        "--thickness_init",
        type=float,
        default=2.0,
        help="Initial constant thickness T0(u,v) in mm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for case selection.",
    )
    return parser.parse_args()


def list_label_files(labels_tr_dir):
    # Accept .nii or .nii.gz
    files = glob.glob(os.path.join(labels_tr_dir, "*.nii")) + glob.glob(
        os.path.join(labels_tr_dir, "*.nii.gz")
    )
    files = sorted(files)
    return files


def load_nifti(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    # voxel spacing (mm)
    if "pixdim" in nii.header:
        # nibabel pixdim: [0, sx, sy, sz, ...]
        spacing = nii.header["pixdim"][1:4].astype(float)
    else:
        spacing = np.array([1.0, 1.0, 1.0], dtype=float)
    return data, spacing


def extract_roi_mask(label_vol, roi_id=ROI_FEMORAL_CARTILAGE):
    return (label_vol == roi_id).astype(np.uint8)


def mask_to_mesh(mask, spacing, level=0.5):
    """
    mask: 3D binary numpy array
    spacing: (sx, sy, sz)
    Returns Open3D TriangleMesh
    """
    if mask.max() == 0:
        return None

    # Marching cubes expects z, y, x, but nibabel gives data in x, y, z order.
    # To stay consistent, we keep nib order but pass spacing accordingly.
    verts, faces, normals, _ = measure.marching_cubes(
        volume=mask, level=level, spacing=spacing
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.compute_triangle_normals()
    return mesh


def smooth_mesh(mesh, n_iters=10):
    """
    Use Open3D Taubin smoothing; fallback to Laplacian if needed.
    """
    if mesh is None:
        return None

    try:
        mesh_smoothed = mesh.filter_smooth_taubin(number_of_iterations=n_iters)
    except Exception:
        mesh_smoothed = mesh.filter_smooth_simple(number_of_iterations=n_iters)

    mesh_smoothed.compute_vertex_normals()
    mesh_smoothed.compute_triangle_normals()
    return mesh_smoothed


def sample_mesh_points(mesh, num_points):
    """
    Uniformly sample points from mesh surface.
    Returns (N, 3) numpy array.
    """
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(pcd.points)


def icp_align(source_mesh, target_mesh, num_points=5000):
    """
    Align source_mesh to target_mesh using ICP (rigid).
    Returns a new transformed copy of source_mesh.
    """
    src_pts = sample_mesh_points(source_mesh, num_points)
    tgt_pts = sample_mesh_points(target_mesh, num_points)

    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_pts)
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)

    # Rough initial alignment: center the clouds
    src_center = src_pts.mean(axis=0)
    tgt_center = tgt_pts.mean(axis=0)
    init_transform = np.eye(4)
    init_transform[:3, 3] = tgt_center - src_center

    threshold = 5.0  # distance threshold (mm) for ICP correspondence
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    aligned_mesh = source_mesh.transform(reg_p2p.transformation.copy())
    aligned_mesh.compute_vertex_normals()
    aligned_mesh.compute_triangle_normals()
    return aligned_mesh


def compute_average_mesh(ref_mesh, aligned_meshes):
    """
    Build an average surface using the topology of ref_mesh.

    For each vertex of the ref_mesh, we find the nearest point on each aligned mesh
    (via KD-tree on sampled points) and average coordinates.

    ref_mesh: Open3D TriangleMesh (reference topology)
    aligned_meshes: list of Open3D TriangleMesh, all in the same coordinate frame

    Returns: Open3D TriangleMesh with same topology as ref_mesh.
    """
    ref_vertices = np.asarray(ref_mesh.vertices)
    faces = np.asarray(ref_mesh.triangles)

    # Precompute KD-trees of each aligned mesh point cloud
    kdtrees = []
    point_sets = []
    for m in aligned_meshes:
        pts = sample_mesh_points(m, num_points=max(5000, len(ref_vertices)))
        point_sets.append(pts)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        kdtrees.append(o3d.geometry.KDTreeFlann(pcd))

    avg_vertices = np.zeros_like(ref_vertices)
    n_meshes = len(aligned_meshes)

    for i, v in enumerate(ref_vertices):
        accum = np.zeros(3, dtype=float)
        for k in range(n_meshes):
            kdtree = kdtrees[k]
            pts = point_sets[k]
            # nearest neighbor
            _, idxs, _ = kdtree.search_knn_vector_3d(v, 1)
            accum += pts[idxs[0]]
        avg_vertices[i] = accum / float(n_meshes)

    avg_mesh = o3d.geometry.TriangleMesh()
    avg_mesh.vertices = o3d.utility.Vector3dVector(avg_vertices)
    avg_mesh.triangles = o3d.utility.Vector3iVector(faces.copy())
    avg_mesh.compute_vertex_normals()
    avg_mesh.compute_triangle_normals()
    return avg_mesh


def fit_nurbs_surface_from_mesh(avg_mesh, size_u=60, size_v=40, degree_u=3, degree_v=3):
    """
    Approximate a NURBS surface S_c(u,v) from the average mesh.

    Changes vs previous version:
    - Normalize vertices into a unit-ish cube before fitting (better conditioning).
    - Fit NURBS in normalized space.
    - Rescale control points back to original coordinates.
    """
    verts = np.asarray(avg_mesh.vertices)           # (N, 3)

    # --- 1) Normalize geometry for numerical stability ---
    center = verts.mean(axis=0, keepdims=True)      # (1, 3)
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    span = (bb_max - bb_min).max()                  # single scale
    span = float(span) if span > 0 else 1.0

    verts_norm = (verts - center) / span            # roughly in [-0.5, 0.5]^3

    # --- 2) PCA in normalized space to define 2D parameterization ---
    verts_centered = verts_norm  # already centered
    U, S, Vt = np.linalg.svd(verts_centered, full_matrices=False)
    pc1 = Vt[0]
    pc2 = Vt[1]

    # Project vertices onto (pc1, pc2) plane
    u_prime = verts_centered @ pc1
    v_prime = verts_centered @ pc2
    uv_prime = np.stack([u_prime, v_prime], axis=1)

    # Normalize to [0, 1] x [0, 1] parameter domain
    min_uv = uv_prime.min(axis=0)
    max_uv = uv_prime.max(axis=0)
    span_uv = np.maximum(max_uv - min_uv, 1e-6)
    uv_norm = (uv_prime - min_uv) / span_uv  # (N, 2) in [0,1]^2

    # --- 3) Build regular (u, v) grid ---
    u_vals = np.linspace(0.0, 1.0, size_u)
    v_vals = np.linspace(0.0, 1.0, size_v)
    uv_grid = np.zeros((size_u, size_v, 2), dtype=float)

    # --- 4) For each (u,v) grid point, pick nearest vertex in param space ---
    points_grid = []
    N = uv_norm.shape[0]
    for i, u in enumerate(u_vals):
        row_pts = []
        for j, v in enumerate(v_vals):
            uv_grid[i, j, :] = [u, v]
            query = np.array([u, v], dtype=float)   # (2,)

            # squared distances in parameter space
            diff = uv_norm - query[None, :]         # (N, 2)
            d2 = np.sum(diff * diff, axis=1)        # (N,)
            idx = np.argmin(d2)

            # use normalized 3D position!
            row_pts.append(verts_norm[idx].tolist())
        points_grid.append(row_pts)

    # Flatten to 1D list, as expected by geomdl.approximate_surface
    points_flat = [points_grid[i][j] for i in range(size_u) for j in range(size_v)]

    # --- 5) Fit cubic tensor-product B-spline / NURBS in normalized space ---
    surf = fitting.approximate_surface(
        points_flat,
        size_u,
        size_v,
        degree_u=degree_u,
        degree_v=degree_v,
    )
    surf.delta = 0.01

    # --- 6) Rescale control points back to original coordinate system ---
    ctrlpts = np.array(surf.ctrlpts, dtype=float)      # (U*V, 3) in normalized coords
    ctrlpts = ctrlpts * span + center                  # back to world coords
    size_ctrl_u = surf.ctrlpts_size_u
    size_ctrl_v = surf.ctrlpts_size_v
    surf.set_ctrlpts(ctrlpts.tolist(), size_ctrl_u, size_ctrl_v)

    return surf, uv_grid




def save_nurbs_template(surf, uv_grid, thickness_init_mm, output_dir):
    """
    Save NURBS template parameters and initial thickness map.

    surf: geomdl surface
    uv_grid: (size_u, size_v, 2)
    thickness_init_mm: scalar (e.g. 2.0)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Control points (list of [x,y,z])
    ctrlpts = np.array(surf.ctrlpts, dtype=float)
    # ctrlpts are flattened; geomdl stores grid size in surf.ctrlpts_size_u/v
    size_u = surf.ctrlpts_size_u
    size_v = surf.ctrlpts_size_v
    ctrlpts_3d = ctrlpts.reshape(size_u, size_v, 3)

    knots_u = np.array(surf.knotvector_u, dtype=float)
    knots_v = np.array(surf.knotvector_v, dtype=float)
    degree_u = surf.degree_u
    degree_v = surf.degree_v

    # Save geometry
    np.savez(
        os.path.join(output_dir, "femoral_template_surf.npz"),
        ctrlpts=ctrlpts_3d,
        knots_u=knots_u,
        knots_v=knots_v,
        degree_u=degree_u,
        degree_v=degree_v,
        uv_grid=uv_grid,
    )

    # Thickness map T(u,v); here just initialize with constant 2.0 mm
    thickness_map = np.full(
        (uv_grid.shape[0], uv_grid.shape[1]), float(thickness_init_mm), dtype=np.float32
    )
    np.save(
        os.path.join(output_dir, "femoral_template_thickness.npy"), thickness_map
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    labels_tr_dir = os.path.join(args.data_root, "labelsTr")
    if not os.path.isdir(labels_tr_dir):
        raise RuntimeError(f"labelsTr directory not found: {labels_tr_dir}")

    label_files = list_label_files(labels_tr_dir)
    if len(label_files) == 0:
        raise RuntimeError(f"No label files found in {labels_tr_dir}")

    print(f"Found {len(label_files)} label files.")
    num_cases = min(args.num_cases, len(label_files))
    selected_files = random.sample(label_files, num_cases)
    print(f"Randomly selected {len(selected_files)} cases for template building.")

    meshes = []
    spacings = []

    for lf in selected_files:
        print(f"Loading label: {os.path.basename(lf)}")
        lbl_vol, spacing = load_nifti(lf)
        roi_mask = extract_roi_mask(lbl_vol, ROI_FEMORAL_CARTILAGE)
        voxel_count = roi_mask.sum()
        if voxel_count < args.min_voxels:
            print(
                f"  -> Skipped (ROI=2 too small: {voxel_count} voxels < {args.min_voxels})"
            )
            continue

        mesh = mask_to_mesh(roi_mask, spacing)
        if mesh is None:
            print("  -> Skipped (no mesh generated)")
            continue

        mesh = smooth_mesh(mesh, n_iters=args.smooth_iters)
        meshes.append(mesh)
        spacings.append(spacing)

    if len(meshes) < 2:
        raise RuntimeError(
            f"Not enough valid meshes for template building (got {len(meshes)})."
        )

    print(f"Using {len(meshes)} meshes after filtering for template building.")

    # Use the first mesh as reference
    ref_mesh = meshes[0]
    aligned_meshes = [ref_mesh]
    for idx, m in enumerate(meshes[1:], start=1):
        print(f"Aligning mesh {idx+1}/{len(meshes)} to reference via ICP...")
        aligned = icp_align(m, ref_mesh, num_points=args.num_sample_points)
        aligned_meshes.append(aligned)

    print("Computing average mesh on reference topology...")
    avg_mesh = compute_average_mesh(ref_mesh, aligned_meshes)

    os.makedirs(args.output_dir, exist_ok=True)
    avg_mesh_path = os.path.join(args.output_dir, "average_mesh.ply")
    o3d.io.write_triangle_mesh(avg_mesh_path, avg_mesh)
    print(f"Average mesh saved to: {avg_mesh_path}")

    print("Fitting NURBS (cubic tensor-product B-spline) central template surface...")
    surf, uv_grid = fit_nurbs_surface_from_mesh(
        avg_mesh,
        size_u=args.uv_size_u,
        size_v=args.uv_size_v,
        degree_u=3,
        degree_v=3,
    )
    print(
        f"Fitted NURBS surface: degree_u={surf.degree_u}, degree_v={surf.degree_v}, "
        f"ctrlpts_size_u={surf.ctrlpts_size_u}, ctrlpts_size_v={surf.ctrlpts_size_v}"
    )

    print("Saving NURBS template and initial thickness map...")
    save_nurbs_template(
        surf,
        uv_grid,
        thickness_init_mm=args.thickness_init,
        output_dir=args.output_dir,
    )

    print("Done. Template: central surface + thickness map ready for downstream use.")


if __name__ == "__main__":
    main()
