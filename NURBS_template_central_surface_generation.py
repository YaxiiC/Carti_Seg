#!/usr/bin/env python3
r'''
NURBS_template_central_surface_generation.py

Template builder for ROI=2 (Femoral Cartilage) from OAI-ZIB-CM with
a cubic tensor-product B-spline (NURBS) central-surface template.

This script implements the template generation stage using a
**central-surface + thickness** representation:

  - Randomly sample knee MRI scans and corresponding cartilage
    annotations (OAI-ZIB-CM).

  - Extract cartilage surfaces from binary masks using Marching Cubes
    and smooth them (Taubin or Laplacian) to remove noise and local
    irregularities.

  - Rigidly register all surfaces into a common coordinate system
    via ICP, then compute a mean cartilage surface on the reference
    topology.

  - Fit a cubic tensor-product B-spline (NURBS) surface to the mean
    surface. This NURBS is interpreted as a **central cartilage
    surface** S_c(u,v) with fixed topology and explicit control-point
    layout + knot vectors.

  - Define a default thickness field T^0(u,v) (in mm) over the same
    (u,v) parameter domain, initialized here as a constant value
    (offset_outer_mm, e.g. 2.0 mm). Inner/outer surfaces can later be
    reconstructed as:
        S_inner(u,v) = S_c(u,v) - 0.5 * T(u,v) * n(u,v)
        S_outer(u,v) = S_c(u,v) + 0.5 * T(u,v) * n(u,v),
    where n(u,v) is the unit normal of the central surface.

The outputs are:
  - central NURBS template mesh (.ply),
  - default thickness map on the (u,v) grid (.npy),
  - NURBS parameters (.npz: ctrl_pts, weights, knots, degrees, eval res),
  - a JSON manifest describing the central-surface + thickness model.

Workflow (code-level):
  multi-label NIfTI -> (mask==2) -> marching cubes -> smoothing -> rigid ICP
  -> mean surface (central_mesh_mc) -> PCA-based (u,v) parameterization
  -> least-squares fit cubic tensor-product B-spline surface
  -> evaluate NURBS on regular (u,v) grid -> central template mesh
  -> define default thickness T^0(u,v) (constant)
  -> save mesh (.ply) + thickness (.npy) + NURBS params (.npz) + manifest (.json)

Run (Windows CMD, one line):
  python NURBS_template_central_surface_generation.py ^
      --data_root C:\Users\chris\MICCAI2026\OAI-ZIB-CM ^
      --out_dir C:\Users\chris\MICCAI2026\Carti_Seg ^
      --split Tr --n_cases 25 --roi_id 2 ^
      --keep_largest_cc --offset_outer_mm 2.0 ^
      --nurbs_ctrl 10 8 --nurbs_eval 80 64





Dependencies:
  pip install numpy nibabel scikit-image open3d tqdm scipy
'''

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes, label as cc_label
from tqdm import tqdm

LABELS_MAP = {
    1: "Femur",
    2: "Femoral Cartilage",
    3: "Tibia",
    4: "Medial Tibial Cartilage",
    5: "Lateral Tibial Cartilage",
}

# ----------------------------
# Utilities
# ----------------------------

def nii_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    """Derive voxel spacing from a NIfTI affine (assumes orthogonal axes)."""
    return np.linalg.norm(affine[:3, :3], axis=0)

def load_roi_binary(mask_path: Path, roi_id: int = 2, keep_largest_cc: bool = True):
    """Load multi-label mask, extract ROI==roi_id as binary float32 array, optionally keep largest CC."""
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    spacing = nii_spacing_from_affine(img.affine)

    roi_mask = (np.round(data).astype(np.int32) == int(roi_id)).astype(np.uint8)

    if roi_mask.max() == 0:
        return roi_mask.astype(np.float32), spacing  # empty

    if keep_largest_cc:
        lab = cc_label(roi_mask, connectivity=1)
        if lab.max() > 0:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            roi_mask = (lab == np.argmax(sizes)).astype(np.uint8)

    return roi_mask.astype(np.float32), spacing

def mesh_from_mask(mask: np.ndarray, spacing, level: float = 0.5) -> o3d.geometry.TriangleMesh:
    """Marching Cubes -> Open3D TriangleMesh in physical space (uses NIfTI spacing)."""
    if mask.max() <= 0:
        raise ValueError("ROI mask is empty (no voxels for the requested label).")
    verts, faces, norms, _ = marching_cubes(mask, level=level, spacing=spacing)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces.astype(np.int32))
    )
    if norms is None or len(norms) != len(verts):
        mesh.compute_vertex_normals()
    else:
        mesh.vertex_normals = o3d.utility.Vector3dVector(norms)
    return mesh

def smooth_mesh(mesh: o3d.geometry.TriangleMesh, method: str = "taubin",
                iterations: int = 30, lambda_: float = 0.5, mu: float = -0.53):
    """Taubin (default) or Laplacian smoothing; cleans degeneracies and recomputes normals."""
    if method == "taubin":
        mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations, lambda_filter=lambda_, mu=mu)
    else:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh

def to_pointcloud(mesh: o3d.geometry.TriangleMesh, n_samples: int = 50000):
    """Uniformly sample mesh surface points for ICP."""
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    return mesh.sample_points_uniformly(number_of_points=n_samples)

def rigid_icp(source_mesh: o3d.geometry.TriangleMesh,
              target_mesh: o3d.geometry.TriangleMesh,
              voxel_down: float = 0.8,
              max_iters: int = 100,
              threshold: float = 5.0):
    """Rigid point-to-point ICP (source->target). Returns transformed mesh and 4x4 transform."""
    src = to_pointcloud(source_mesh)
    tgt = to_pointcloud(target_mesh)
    if voxel_down and voxel_down > 0:
        src = src.voxel_down_sample(voxel_down)
        tgt = tgt.voxel_down_sample(voxel_down)
    init = np.eye(4)
    init[:3, 3] = np.array(tgt.get_center()) - np.array(src.get_center())
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, threshold, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters),
    )
    T = reg.transformation
    out = o3d.geometry.TriangleMesh(source_mesh)
    out.transform(T)
    out.compute_vertex_normals()
    return out, T

def average_on_reference_topology(ref_mesh: o3d.geometry.TriangleMesh,
                                  aligned_meshes,
                                  k: int = 1) -> o3d.geometry.TriangleMesh:
    """Mean surface on the reference topology via nearest-neighbor vertex correspondence."""
    ref_v = np.asarray(ref_mesh.vertices)
    accum = np.zeros_like(ref_v)
    counts = np.zeros((ref_v.shape[0], 1))
    for m in aligned_meshes:
        mv = np.asarray(m.vertices)
        tree = cKDTree(mv)
        _, idx = tree.query(ref_v, k=k)
        if k == 1:
            matched = mv[idx]
        else:
            matched = mv[idx].mean(axis=1)
        accum += matched
        counts += 1.0
    mean_v = accum / np.clip(counts, 1.0, None)
    mean_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mean_v),
        triangles=o3d.utility.Vector3iVector(np.asarray(ref_mesh.triangles)),
    )
    mean_mesh.remove_degenerate_triangles()
    mean_mesh.compute_vertex_normals()
    return mean_mesh

def offset_surface_along_normals(mesh: o3d.geometry.TriangleMesh, offset_mm: float = 2.0):
    """Create an offset surface by displacing vertices along vertex normals."""
    m = o3d.geometry.TriangleMesh(mesh)
    m.compute_vertex_normals()
    v = np.asarray(m.vertices)
    n = np.asarray(m.vertex_normals)
    m.vertices = o3d.utility.Vector3dVector(v + offset_mm * n)
    m.compute_vertex_normals()
    return m

# ------- filename pairing (handles _0000 vs no-suffix) -------

def _canonical_stem(p: Path) -> str:
    """
    Returns a stem that ignores channel suffixes like _0000 and the extension.
    Examples:
      oaizib_001_0000.nii.gz -> oaizib_001
      oaizib_001_0001.nii    -> oaizib_001
      oaizib_001.nii         -> oaizib_001
    """
    name = p.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
        name = "_".join(parts[:-1])
    return name

def gather_pairs(oai_root: Path, split: str = "Tr"):
    """
    Build (image,label) pairs by canonical stem:
      imagesTr/oaizib_001_0000.nii  <-> labelsTr/oaizib_001.nii
    If multiple image channels exist (e.g., _0000, _0001), it picks _0000 by default.
    """
    img_dir = oai_root / f"images{split}"
    lab_dir = oai_root / f"labels{split}"
    if not img_dir.exists() or not lab_dir.exists():
        raise FileNotFoundError(f"Expected {img_dir} and {lab_dir}")

    img_files = sorted(list(img_dir.glob("*.nii")) + list(img_dir.glob("*.nii.gz")))
    lab_files = sorted(list(lab_dir.glob("*.nii")) + list(lab_dir.glob("*.nii.gz")))

    imgs_by_stem = defaultdict(list)
    for ip in img_files:
        imgs_by_stem[_canonical_stem(ip)].append(ip)

    labs_by_stem = {}
    for lp in lab_files:
        labs_by_stem[_canonical_stem(lp)] = lp

    pairs = []
    missing_labels = []
    for stem, candidates in imgs_by_stem.items():
        lp = labs_by_stem.get(stem, None)
        if lp is None:
            missing_labels.append(stem)
            continue
        preferred = None
        for c in candidates:
            if c.name.endswith("_0000.nii") or c.name.endswith("_0000.nii.gz"):
                preferred = c
                break
        if preferred is None:
            preferred = candidates[0]
        pairs.append((preferred, lp))

    if len(pairs) == 0:
        raise RuntimeError(
            "No (image,label) pairs found after stem matching.\n"
            f"Found {len(img_files)} images, {len(lab_files)} labels.\n"
            f"Example images: {[p.name for p in img_files[:3]]}\n"
            f"Example labels: {[p.name for p in lab_files[:3]]}\n"
        )

    print(f"Matched {len(pairs)} image/label pairs by stem. Missing labels for {len(missing_labels)} stems.")
    if missing_labels[:5]:
        print("First missing-label stems:", missing_labels[:5])
    return pairs

# -------------------------------------------------
# NURBS / B-spline utilities (cubic tensor-product)
# -------------------------------------------------

def pca_parameterize_vertices(verts: np.ndarray) -> np.ndarray:
    """
    Simple 2D parameterization of vertices via PCA:
      - center on mean
      - project to first two principal components
      - rescale to [0,1] x [0,1]
    Returns (N,2) array of (u,v) in [0,1].
    """
    v = verts.astype(np.float64)
    v_centered = v - v.mean(axis=0, keepdims=True)
    cov = v_centered.T @ v_centered / max(v.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    e1 = eigvecs[:, idx[0]]
    e2 = eigvecs[:, idx[1]]
    uv = np.stack([v_centered @ e1, v_centered @ e2], axis=1)
    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    uv = (uv - uv_min) / (uv_max - uv_min + 1e-8)
    return uv.astype(np.float32)

def open_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    """
    Open uniform knot vector in [0,1] for n_ctrl control points and given degree.
    Length = n_ctrl + degree + 1.
    """
    m = n_ctrl + degree + 1
    U = np.zeros(m, dtype=np.float32)
    U[-(degree + 1):] = 1.0
    n_inner = n_ctrl - degree - 1
    if n_inner > 0:
        for j in range(1, n_inner + 1):
            U[degree + j] = j / (n_inner + 1)
    return U

def bspline_basis_vector(u: float, degree: int, knots: np.ndarray, n_ctrl: int) -> np.ndarray:
    """
    Evaluate all B-spline basis functions N_i,p(u) for i=0..n_ctrl-1 at scalar u.

    Implementation: standard Cox–de Boor, using local basis and placing it
    into the global vector according to knot span.
    """
    u = float(np.clip(u, 0.0, 1.0))
    # Special-case right boundary
    if np.isclose(u, knots[-1]):
        span = n_ctrl - 1
    else:
        span = max(degree, min(np.searchsorted(knots, u) - 1, n_ctrl - 1))

    left = np.zeros(degree + 1, dtype=np.float64)
    right = np.zeros(degree + 1, dtype=np.float64)
    N = np.zeros(degree + 1, dtype=np.float64)
    N[0] = 1.0

    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if denom == 0.0:
                temp = 0.0
            else:
                temp = N[r] / denom
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    out = np.zeros(n_ctrl, dtype=np.float32)
    i0 = span - degree
    out[i0:i0 + degree + 1] = N.astype(np.float32)
    return out

def fit_bspline_surface(
    verts: np.ndarray,
    uv: np.ndarray,
    degree_u: int,
    degree_v: int,
    n_ctrl_u: int,
    n_ctrl_v: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Least-squares fit of a tensor-product B-spline surface:

      S(u,v) = sum_{i,j} N_i,p(u) M_j,q(v) P_ij

    Inputs:
      verts : (N,3) points in R^3
      uv    : (N,2) params in [0,1]^2
      degrees, number of control points in u/v

    Returns:
      ctrl_pts: (n_ctrl_u, n_ctrl_v, 3)
      weights : (n_ctrl_u, n_ctrl_v)  (all ones -> non-rational B-spline)
      knots_u : (n_ctrl_u + degree_u + 1,)
      knots_v : (n_ctrl_v + degree_v + 1,)
    """
    N = verts.shape[0]
    n_ctrl_u = int(n_ctrl_u)
    n_ctrl_v = int(n_ctrl_v)

    # Knot vectors
    knots_u = open_uniform_knots(n_ctrl_u, degree_u)
    knots_v = open_uniform_knots(n_ctrl_v, degree_v)

    # Basis matrices
    Bu = np.zeros((N, n_ctrl_u), dtype=np.float32)
    Bv = np.zeros((N, n_ctrl_v), dtype=np.float32)
    for k in range(N):
        Bu[k] = bspline_basis_vector(float(uv[k, 0]), degree_u, knots_u, n_ctrl_u)
        Bv[k] = bspline_basis_vector(float(uv[k, 1]), degree_v, knots_v, n_ctrl_v)

    # Full tensor-product basis: B[k, i,j] = Bu[k,i] * Bv[k,j]
    # -> reshape to (N, n_ctrl_u * n_ctrl_v)
    B = np.einsum("ki,kj->kij", Bu, Bv, optimize=True).reshape(N, n_ctrl_u * n_ctrl_v)

    # Solve three LS systems (x,y,z)
    # B @ P_flat = verts
    P_flat, *_ = np.linalg.lstsq(B, verts.astype(np.float64), rcond=None)
    ctrl_pts = P_flat.reshape(n_ctrl_u, n_ctrl_v, 3).astype(np.float32)

    # weights = 1 (ordinary B-spline, but we keep NURBS semantics)
    weights = np.ones((n_ctrl_u, n_ctrl_v), dtype=np.float32)

    return ctrl_pts, weights, knots_u, knots_v

def evaluate_bspline_surface_grid(
    ctrl_pts: np.ndarray,
    knots_u: np.ndarray,
    knots_v: np.ndarray,
    degree_u: int,
    degree_v: int,
    res_u: int,
    res_v: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate tensor-product B-spline surface on regular (u,v) grid:

      u in [0,1], res_u samples
      v in [0,1], res_v samples

    Returns:
      verts: (res_u * res_v, 3)
      faces: (2*(res_u-1)*(res_v-1), 3)
    """
    n_ctrl_u, n_ctrl_v, _ = ctrl_pts.shape
    us = np.linspace(0.0, 1.0, res_u, dtype=np.float32)
    vs = np.linspace(0.0, 1.0, res_v, dtype=np.float32)

    Bu = np.stack([bspline_basis_vector(float(u), degree_u, knots_u, n_ctrl_u) for u in us], axis=0)  # (res_u,n_ctrl_u)
    Bv = np.stack([bspline_basis_vector(float(v), degree_v, knots_v, n_ctrl_v) for v in vs], axis=0)  # (res_v,n_ctrl_v)

    # For each coord c:
    #   S_c(u,v) = Bu @ ctrl_pts[:,:,c] @ Bv^T
    verts_grid = np.zeros((res_u, res_v, 3), dtype=np.float32)
    for c in range(3):
        verts_grid[..., c] = Bu @ ctrl_pts[:, :, c] @ Bv.T

    verts = verts_grid.reshape(-1, 3)

    # Regular quad mesh triangulated into two triangles per cell
    faces = []
    for i in range(res_u - 1):
        for j in range(res_v - 1):
            idx0 = i * res_v + j
            idx1 = idx0 + 1
            idx2 = idx0 + res_v
            idx3 = idx2 + 1
            faces.append([idx0, idx1, idx2])
            faces.append([idx1, idx3, idx2])
    faces = np.asarray(faces, dtype=np.int32)

    return verts, faces

# ----------------------------
# Main pipeline
# ----------------------------

def build_template_for_roi(
    data_root,
    out_dir,
    roi_id: int = 2,                    # Femoral Cartilage
    n_cases: int = 25,
    split: str = "Tr",
    seed: int = 0,
    keep_largest_cc: bool = True,
    mc_level: float = 0.5,
    smooth_method: str = "taubin",
    smooth_iters: int = 30,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
    icp_down_voxel: float = 0.8,
    icp_max_iters: int = 100,
    icp_threshold: float = 5.0,
    offset_outer_mm: float = 2.0,       # interpreted as default thickness (mm)
    save_aligned: bool = True,
    # ---- NURBS-specific args ----
    nurbs_degree_u: int = 3,
    nurbs_degree_v: int = 3,
    nurbs_ctrl_u: int = 10,
    nurbs_ctrl_v: int = 8,
    nurbs_eval_u: int = 80,
    nurbs_eval_v: int = 64,
):
    """
    Build a NURBS-based **central cartilage surface + thickness** template
    for a given ROI (default: femoral cartilage).

    The fitted NURBS surface is interpreted as a central surface S_c(u,v).
    The scalar parameter `offset_outer_mm` is used as a default constant
    thickness T^0(u,v) (in mm) over the evaluation grid.
    """
    random.seed(seed)
    np.random.seed(seed)

    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aligned").mkdir(exist_ok=True)

    label_name = LABELS_MAP.get(int(roi_id), f"ROI_{roi_id}")
    pairs = gather_pairs(data_root, split=split)
    if not pairs:
        raise RuntimeError("No (image,label) pairs found.")

    sampled = random.sample(pairs, min(n_cases, len(pairs)))
    meshes = []
    names = []

    print(f"Extracting ROI={roi_id} ({label_name}) from {len(sampled)} masks...")
    for img_path, lab_path in tqdm(sampled):
        roi_mask, spacing = load_roi_binary(lab_path, roi_id=roi_id, keep_largest_cc=keep_largest_cc)
        if roi_mask.max() == 0:
            # Skip empty ROI (some cases may lack the label)
            continue
        m = mesh_from_mask(roi_mask, spacing, level=mc_level)
        m = smooth_mesh(m, method=smooth_method, iterations=smooth_iters,
                        lambda_=taubin_lambda, mu=taubin_mu)
        meshes.append(m)
        names.append(img_path.name)

    if len(meshes) == 0:
        raise RuntimeError(f"No non-empty meshes for ROI={roi_id}. Check labels and ROI id.")

    # Reference topology: first mesh
    ref_mesh = meshes[0]
    aligned = [ref_mesh]
    transforms = [np.eye(4)]

    print("Rigid ICP alignment to reference...")
    for m in tqdm(meshes[1:]):
        am, T = rigid_icp(m, ref_mesh, voxel_down=icp_down_voxel,
                          max_iters=icp_max_iters, threshold=icp_threshold)
        aligned.append(am)
        transforms.append(T)

    if save_aligned:
        for nm, m in zip(names, aligned):
            outp = out_dir / "aligned" / (nm.replace(".nii", "").replace(".gz", "") + f"_roi{roi_id}_aligned.ply")
            o3d.io.write_triangle_mesh(str(outp), m)

    print("Computing mean template on reference topology...")
    mean_mesh_mc = average_on_reference_topology(ref_mesh, aligned, k=1)

    # Additional smoothing on mean mesh before NURBS fit
    central_mesh_mc = smooth_mesh(mean_mesh_mc, method=smooth_method,
                                  iterations=max(1, smooth_iters // 2),
                                  lambda_=taubin_lambda, mu=taubin_mu)

    # -------------------------------------------------
    # NURBS fitting on mean mesh (central surface)
    # -------------------------------------------------
    print("Fitting cubic tensor-product B-spline surface (central NURBS template)...")
    central_verts = np.asarray(central_mesh_mc.vertices)   # (V,3)

    # 1) Parameterize vertices -> (u,v) in [0,1]^2
    uv = pca_parameterize_vertices(central_verts)          # (V,2)

    # 2) Fit B-spline surface (degrees typically 3,3)
    ctrl_pts, weights, knots_u, knots_v = fit_bspline_surface(
        central_verts,
        uv,
        degree_u=nurbs_degree_u,
        degree_v=nurbs_degree_v,
        n_ctrl_u=nurbs_ctrl_u,
        n_ctrl_v=nurbs_ctrl_v,
    )

    # 3) Evaluate surface on regular (u,v) grid to build central template mesh
    central_eval_verts, central_eval_faces = evaluate_bspline_surface_grid(
        ctrl_pts,
        knots_u,
        knots_v,
        degree_u=nurbs_degree_u,
        degree_v=nurbs_degree_v,
        res_u=nurbs_eval_u,
        res_v=nurbs_eval_v,
    )

    central_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(central_eval_verts),
        triangles=o3d.utility.Vector3iVector(central_eval_faces.astype(np.int32)),
    )
    central_mesh.remove_degenerate_triangles()
    central_mesh.compute_vertex_normals()

    # -------------------------------------------------
    # Thickness field on (u,v): default constant thickness in mm
    # -------------------------------------------------
    thickness_default = None
    thickness_path = None
    if offset_outer_mm and offset_outer_mm > 0:
        print(f"Creating default thickness field T^0(u,v) = {offset_outer_mm:.2f} mm...")
        thickness_default = np.full(
            (nurbs_eval_u, nurbs_eval_v),
            float(offset_outer_mm),
            dtype=np.float32,
        )
        thickness_path = out_dir / f"{label_name.lower().replace(' ', '_')}_template_thickness_default.npy"
        np.save(thickness_path, thickness_default)

    # ----------------------
    # Save mesh & NURBS params
    # ----------------------
    base = f"{label_name.lower().replace(' ', '_')}_template"
    central_path = out_dir / f"{base}_central.ply"
    o3d.io.write_triangle_mesh(str(central_path), central_mesh)

    manifest = {
        "data_root": str(data_root),
        "split": split,
        "roi_id": int(roi_id),
        "roi_name": label_name,
        "n_used": len(aligned),
        "sampled_cases": names,
        "preproc": {
            "keep_largest_cc": keep_largest_cc,
            "mc_level": mc_level
        },
        "smoothing": {
            "method": smooth_method,
            "iterations": smooth_iters,
            "taubin_lambda": taubin_lambda,
            "taubin_mu": taubin_mu
        },
        "icp": {
            "voxel_down": icp_down_voxel,
            "max_iters": icp_max_iters,
            "threshold": icp_threshold
        },
        # Interpreted as default physiological thickness in mm
        "default_thickness_mm": float(offset_outer_mm),
        "representation": {
            "type": "central_surface_plus_thickness",
            "description": (
                "Central NURBS cartilage surface S_c(u,v) with a default "
                "thickness field T^0(u,v) in mm. Inner/outer surfaces can be "
                "reconstructed as S_inner = S_c - 0.5*T*n, "
                "S_outer = S_c + 0.5*T*n."
            ),
            "thickness_grid_shape": [int(nurbs_eval_u), int(nurbs_eval_v)],
        },
        "nurbs": {
            "degree_u": int(nurbs_degree_u),
            "degree_v": int(nurbs_degree_v),
            "n_ctrl_u": int(nurbs_ctrl_u),
            "n_ctrl_v": int(nurbs_ctrl_v),
            "eval_res_u": int(nurbs_eval_u),
            "eval_res_v": int(nurbs_eval_v),
        },
        "outputs": {
            "central_mesh": str(central_path),
        }
    }

    if thickness_path is not None:
        manifest["outputs"]["thickness_default"] = str(thickness_path)

    # NURBS parameters -> .npz
    # This file fully specifies the central surface template in the NURBS parameter space.
    nurbs_npz_path = out_dir / f"{base}_nurbs_params.npz"
    np.savez(
        nurbs_npz_path,
        ctrl_pts=ctrl_pts,          # (n_ctrl_u,n_ctrl_v,3)
        weights=weights,            # (n_ctrl_u,n_ctrl_v)
        knots_u=knots_u,            # (n_ctrl_u+degree_u+1,)
        knots_v=knots_v,            # (n_ctrl_v+degree_v+1,)
        degree_u=np.int32(nurbs_degree_u),
        degree_v=np.int32(nurbs_degree_v),
        eval_res_u=np.int32(nurbs_eval_u),
        eval_res_v=np.int32(nurbs_eval_v),
    )
    manifest["outputs"]["nurbs_params"] = str(nurbs_npz_path)

    with open(out_dir / f"{base}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(f"Saved central NURBS template mesh: {central_path}")
    if thickness_path is not None:
        print(f"Saved default thickness field: {thickness_path}")
    print(f"Saved NURBS parameters: {nurbs_npz_path}")
    print(f"Manifest: {out_dir / (base + '_manifest.json')}")

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build femoral cartilage (ROI=2) central NURBS template from OAI-ZIB-CM")
    p.add_argument("--data_root", type=str, required=True, help="Path to OAI-ZIB-CM")
    p.add_argument("--out_dir", type=str, default="outputs/template_femoral_cartilage")
    p.add_argument("--split", type=str, default="Tr", choices=["Tr", "Ts"])
    p.add_argument("--n_cases", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--roi_id", type=int, default=2, help="Use 2 for Femoral Cartilage")
    p.add_argument("--keep_largest_cc", action="store_true", help="Keep only largest connected component of ROI")
    p.add_argument("--mc_level", type=float, default=0.5, help="Marching Cubes isovalue for binary mask")
    p.add_argument("--smooth_method", type=str, default="taubin", choices=["taubin", "laplacian"])
    p.add_argument("--smooth_iters", type=int, default=30)
    p.add_argument("--taubin_lambda", type=float, default=0.5)
    p.add_argument("--taubin_mu", type=float, default=-0.53)
    p.add_argument("--icp_down_voxel", type=float, default=0.8, help="Voxel size for ICP downsample (mm)")
    p.add_argument("--icp_max_iters", type=int, default=100)
    p.add_argument("--icp_threshold", type=float, default=5.0, help="ICP correspondence threshold (mm)")
    p.add_argument(
        "--offset_outer_mm",
        type=float,
        default=2.0,
        help="Default cartilage thickness (mm) used to initialize T^0(u,v) for the central-surface model",
    )
    p.add_argument("--no_save_aligned", action="store_true", help="Skip saving per-case aligned meshes")
    # NURBS-specific
    p.add_argument("--nurbs_degree_u", type=int, default=3)
    p.add_argument("--nurbs_degree_v", type=int, default=3)
    p.add_argument("--nurbs_ctrl", type=int, nargs=2, default=[10, 8],
                   help="Number of control points in (u,v), e.g., --nurbs_ctrl 10 8")
    p.add_argument("--nurbs_eval", type=int, nargs=2, default=[80, 64],
                   help="Evaluation resolution (u,v) for template mesh, e.g., --nurbs_eval 80 64")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    nurbs_ctrl_u, nurbs_ctrl_v = a.nurbs_ctrl
    nurbs_eval_u, nurbs_eval_v = a.nurbs_eval

    build_template_for_roi(
        data_root=a.data_root,
        out_dir=a.out_dir,
        roi_id=a.roi_id,
        n_cases=a.n_cases,
        split=a.split,
        seed=a.seed,
        keep_largest_cc=a.keep_largest_cc,
        mc_level=a.mc_level,
        smooth_method=a.smooth_method,
        smooth_iters=a.smooth_iters,
        taubin_lambda=a.taubin_lambda,
        taubin_mu=a.taubin_mu,
        icp_down_voxel=a.icp_down_voxel,
        icp_max_iters=a.icp_max_iters,
        icp_threshold=a.icp_threshold,
        offset_outer_mm=a.offset_outer_mm,
        save_aligned=not a.no_save_aligned,
        nurbs_degree_u=a.nurbs_degree_u,
        nurbs_degree_v=a.nurbs_degree_v,
        nurbs_ctrl_u=nurbs_ctrl_u,
        nurbs_ctrl_v=nurbs_ctrl_v,
        nurbs_eval_u=nurbs_eval_u,
        nurbs_eval_v=nurbs_eval_v,
    )
